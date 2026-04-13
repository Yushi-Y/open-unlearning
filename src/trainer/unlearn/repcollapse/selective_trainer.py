"""SelectiveCollapse: activation-only collapse with selectivity-aware cutoff.

Based on RepCollapse but with two key changes:
1. Uses SelectiveCollapser (PCA + ratio-based auto n_pcs) instead of CovCollapser
2. Activation-only collapse — gradients are not collapsed (fixes DISCO issue #1)
3. Retain forward pass in epoch 0 to collect retain activations for ratio computation
"""
import logging
import math
import random

import torch as pt
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise

from data.utils import batched, prep_batch
from evals.kl_eval import KLComputor
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repcollapse.selective_collapser import SelectiveCollapser
from trainer.unlearn.repcollapse.utils import get_banned_tokens, ManualLoRA
from trainer.utils import normalize_grads

logging.basicConfig(level=logging.INFO)


class SelectiveCollapse(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.use_hooks = False
        self.recording_retain = False
        self.batch_idx = 0
        self.recalc_every = math.ceil(
            len(self.train_dataset) / self.args.per_device_train_batch_size
        )
        logging.info(f"{self.recalc_every=}")
        assert self.args.gradient_accumulation_steps == 1

        max_pcs = cfg.get("max_pcs", 400)
        threshold = cfg.get("selectivity_threshold", 1.5)
        reg_eps = cfg.get("reg_eps", 1e-4)
        pca_source = cfg.get("pca_source", "forget")
        contrastive_alpha = cfg.get("contrastive_alpha", 0.5)
        boost_beta = cfg.get("boost_beta", 0.0)
        collapse_rule = cfg.get("collapse_rule", "mahal")
        cr_args = (max_pcs, threshold, reg_eps, pca_source,
                   contrastive_alpha, boost_beta, collapse_rule)

        collapse_attn = cfg.get("collapse_attn", False)

        self.model.requires_grad_(False)
        self.lora_params = []
        self.base_trainable_params = []
        for layer_num in range(len(self.model.model.layers)):
            layer = self.model.model.layers[layer_num]

            # MLP modules
            mlp = layer.mlp
            shared_collapser = SelectiveCollapser(*cr_args)
            down_collapser = SelectiveCollapser(*cr_args)
            for module in [mlp.gate_proj, mlp.up_proj, mlp.down_proj]:
                module.weight.requires_grad = True
                self.base_trainable_params.append(module.weight)
                module.register_forward_hook(self.save_act_input_hook)
                module.register_full_backward_hook(self.collapse_hook)

                if module is mlp.down_proj:
                    module.act_collapser = down_collapser
                else:
                    module.act_collapser = shared_collapser

                if "lora_lr" in cfg:
                    module.lora_module = ManualLoRA(
                        module.weight.shape[1],
                        module.weight.shape[0],
                        cfg.lora_rank,
                    ).to(self.model.device, dtype=self.model.dtype)
                    self.lora_params.extend(module.lora_module.parameters())
                    module.register_forward_hook(self.lora_forward_hook)
                if "noise_std" in cfg:
                    module.register_forward_hook(self.noise_forward_hook)

            # Attention modules (optional)
            if collapse_attn:
                attn = layer.self_attn
                # q/k/v share input space, o_proj has different input
                qkv_collapser = SelectiveCollapser(*cr_args)
                o_collapser = SelectiveCollapser(*cr_args)
                for module in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
                    module.weight.requires_grad = True
                    self.base_trainable_params.append(module.weight)
                    module.register_forward_hook(self.save_act_input_hook)
                    module.register_full_backward_hook(self.collapse_hook)

                    if module is attn.o_proj:
                        module.act_collapser = o_collapser
                    else:
                        module.act_collapser = qkv_collapser

                    if "noise_std" in cfg:
                        module.register_forward_hook(self.noise_forward_hook)

        # KL masking: pre-cache retain batches
        self.kl_computor = None
        if "retain_momentum" in self.cfg:
            self.retain_batches = [
                self.data_collator(r)
                for r in batched(
                    self.train_dataset.retain, self.args.per_device_train_batch_size
                )
            ]

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # Lazy init KLComputor (only needed when retain_grad_source=kl, the default)
        retain_grad_source = self.cfg.get("retain_grad_source", "kl")
        if (self.kl_computor is None and "retain_momentum" in self.cfg
                and retain_grad_source == "kl"):
            self.kl_computor = KLComputor(self.model, self.retain_batches)

        # Retain activation collection (epoch 0 only)
        if self.batch_idx < self.recalc_every:
            r_batch = inputs["retain"]
            self.retain_token_mask = r_batch["attention_mask"].bool().clone()
            self.retain_token_mask[:, 0] = False
            if self.processing_class.chat_template is not None:
                for banned_token in get_banned_tokens(self.processing_class):
                    self.retain_token_mask &= r_batch["input_ids"] != banned_token

            # Forward pass: collect retain activations
            self.recording_retain = True
            with pt.no_grad():
                model(**prep_batch(r_batch, self.model.device))
            self.recording_retain = False

        # Frozen retain grad (interp simplification): compute ref_grad ONCE at the
        # first real training step from the average over all retain batches, then
        # reuse forever. No momentum, no per-step recomputation, no KLComputor.
        frozen_retain = self.cfg.get("frozen_retain_grad", False)
        if (frozen_retain and "retain_momentum" in self.cfg
                and self.batch_idx == self.recalc_every):
            for param in self.base_trainable_params:
                param.ref_grad_frozen = pt.zeros_like(param)
            for r_batch in self.retain_batches:
                model.zero_grad(set_to_none=True)
                ref_loss = model(**prep_batch(r_batch, self.model.device)).loss
                ref_loss.backward()
                for param in self.base_trainable_params:
                    if param.grad is not None:
                        param.ref_grad_frozen += param.grad.detach() / len(self.retain_batches)
            for param in self.base_trainable_params:
                param.ref_grad = quantize_blockwise(param.ref_grad_frozen)
                del param.ref_grad_frozen

        # KL masking: compute retain reference gradient (KL or plain CE).
        if ("retain_momentum" in self.cfg and self.batch_idx >= self.recalc_every
                and not frozen_retain):
            r_batch = random.choice(self.retain_batches)
            model.zero_grad(set_to_none=True)
            if retain_grad_source == "ce":
                # Simplification #2: plain retain CE loss instead of KL(p||p_frozen).
                # Removes the cached hidden states + cached lm_head machinery.
                ref_loss = model(**prep_batch(r_batch, self.model.device)).loss
            else:
                ref_loss, _, _ = self.kl_computor.get_kl(r_batch)
            ref_loss.backward()
            for param in self.base_trainable_params:
                if hasattr(param, "ref_grad"):
                    ref = dequantize_blockwise(*param.ref_grad)
                else:
                    ref = pt.zeros_like(param)
                if param.grad is not None:
                    momentum = self.cfg.retain_momentum
                    ref = ref * momentum + param.grad * (1 - momentum)
                param.ref_grad = quantize_blockwise(ref)

        # Forget pass
        batch = inputs["forget"]
        self.token_mask = batch["attention_mask"].bool().clone()
        self.token_mask[:, 0] = False
        if self.processing_class.chat_template is not None:
            for banned_token in get_banned_tokens(self.processing_class):
                self.token_mask &= batch["input_ids"] != banned_token

        self.use_hooks = True
        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, self.model.device))
        forget_loss = -output.loss

        for p in self.base_trainable_params:
            p.requires_grad_(False)
        forget_loss.backward()
        for p in self.base_trainable_params:
            p.requires_grad_(True)
        self.use_hooks = False

        # LoRA adversarial update
        if "lora_lr" in self.cfg:
            normalize_grads(self.lora_params)
            for p in self.lora_params:
                p.data += self.cfg.lora_lr * self.args.learning_rate * p.grad
                p.grad = None

        self.batch_idx += 1
        if self.batch_idx % self.recalc_every == 0:
            for module in model.modules():
                if hasattr(module, "act_collapser"):
                    module.act_collapser.process_saved_vecs()

        normalize_grads(self.base_trainable_params)
        return forget_loss.detach()

    def save_act_input_hook(self, module, args, output):
        # Collect retain activations during epoch 0
        if self.recording_retain:
            retain_acts = args[0].detach()
            retain_acts = retain_acts[self.retain_token_mask]
            module.act_collapser.add_retain_vecs(retain_acts)
            return
        if not self.use_hooks:
            return
        module.last_act_input = args[0].detach()

    def collapse_hook(self, module, grad_input, grad_output):
        if not self.use_hooks:
            return
        acts = module.last_act_input
        grads = grad_output[0]
        module.last_act_input = None

        acts = acts[self.token_mask]
        grads = grads[self.token_mask]

        # Collect forget activations (for next epoch's PCA)
        module.act_collapser.add_forget_vecs(acts)

        if self.batch_idx < self.recalc_every:
            return  # too early, only collect activations

        # Save raw activations (pre-collapse) for token-level filters that need
        # the original (centered) activation, not the post-collapse version.
        raw_acts = acts

        # Activation-only collapse (no gradient collapse)
        acts = module.act_collapser.collapse(acts)

        # KL masking
        if "retain_momentum" in self.cfg:
            ref_grad = dequantize_blockwise(*module.weight.ref_grad)
            ref_grad = ref_grad.to(module.weight.dtype)
            token_disr = pt.einsum("ij,ti,tj->t", ref_grad, grads, acts)
            kl_mask = token_disr > 0
            acts = acts[kl_mask]
            grads = grads[kl_mask]

        # Novel token filters (§8): retain-shape-based rejection, no retain grads.
        token_filter = self.cfg.get("token_filter", None)
        if token_filter is not None:
            collapser = module.act_collapser
            tau = self.cfg.get("token_filter_tau", 0.5)
            mu = collapser.mean.to(raw_acts.dtype)
            centered = raw_acts - mu
            denom = (centered * centered).sum(dim=1).clamp(min=1e-8)

            if token_filter == "retain_conf":
                # A: skip token if most of its centered activation lives in V_r.
                V_r = collapser.V_r.to(raw_acts.dtype)
                proj = centered @ V_r  # (t, k_r)
                num = (proj * proj).sum(dim=1)
                r_t = num / denom
                keep = r_t <= tau
            elif token_filter == "forget_conf":
                # B: keep token only if centered activation is mostly in V.
                V = collapser.eig_vec.to(raw_acts.dtype)
                proj = centered @ V
                num = (proj * proj).sum(dim=1)
                f_t = num / denom
                keep = f_t >= tau
            elif token_filter == "retain_cov_reject":
                # C: reject tokens whose rank-1 update has large retain-cov mass.
                V_r = collapser.V_r.to(raw_acts.dtype)
                sig2 = collapser.retain_var_r.to(raw_acts.dtype)  # (k_r,)
                proj = centered @ V_r  # (t, k_r)
                quad = ((proj * proj) * sig2).sum(dim=1)  # (t,) = a^T P_r a
                gnorm = grads.norm(dim=1)
                delta = gnorm * quad
                # tau here is the top-fraction to reject (default 0.5 → reject top 50%)
                frac_reject = tau
                t = delta.numel()
                n_keep = max(1, int((1.0 - frac_reject) * t))
                _, idx = pt.topk(delta, n_keep, largest=False)
                keep = pt.zeros_like(delta, dtype=pt.bool)
                keep[idx] = True
            elif token_filter == "soft_C":
                # Soft reweighting variant of C: instead of hard rejection, multiply
                # each token's grad contribution by w_t = 1 / (1 + delta_t / median(delta)).
                # High-delta (retain-risky) tokens are attenuated, not removed.
                V_r = collapser.V_r.to(raw_acts.dtype)
                sig2 = collapser.retain_var_r.to(raw_acts.dtype)
                proj = centered @ V_r
                quad = ((proj * proj) * sig2).sum(dim=1)
                gnorm = grads.norm(dim=1)
                delta = gnorm * quad
                med = delta.median().clamp(min=1e-8)
                w = 1.0 / (1.0 + delta / med)
                grads = grads * w.unsqueeze(1).to(grads.dtype)
                keep = pt.ones_like(delta, dtype=pt.bool)
            elif token_filter == "hybrid_AC":
                # A ∪ C: reject if A rejects (r_t > tau_A) OR if C rejects
                # (top frac_C of gnorm*quad). Two independent criteria combined.
                tau_A = self.cfg.get("hybrid_tau_A", 0.3)
                frac_C = self.cfg.get("hybrid_frac_C", 0.5)
                V_r = collapser.V_r.to(raw_acts.dtype)
                sig2 = collapser.retain_var_r.to(raw_acts.dtype)
                proj = centered @ V_r
                # A component
                num = (proj * proj).sum(dim=1)
                r_t = num / denom
                keep_A = r_t <= tau_A
                # C component
                quad = ((proj * proj) * sig2).sum(dim=1)
                gnorm = grads.norm(dim=1)
                delta = gnorm * quad
                t = delta.numel()
                n_keep_C = max(1, int((1.0 - frac_C) * t))
                _, idx = pt.topk(delta, n_keep_C, largest=False)
                keep_C = pt.zeros_like(delta, dtype=pt.bool)
                keep_C[idx] = True
                keep = keep_A & keep_C  # both must accept
            else:
                raise ValueError(f"unknown token_filter {token_filter}")

            acts = acts[keep]
            grads = grads[keep]

        module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)

    def lora_forward_hook(self, module, args, output):
        if self.use_hooks:
            return output + module.lora_module(args[0])

    def noise_forward_hook(self, module, args, output):
        # Parameter-free adversary: inject Gaussian noise scaled by per-layer std.
        if self.use_hooks:
            noise_std = self.cfg.get("noise_std", 0.1)
            return output + pt.randn_like(output) * noise_std * output.std()
