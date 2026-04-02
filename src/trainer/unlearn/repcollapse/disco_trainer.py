# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=DISCO task_name=test_disco
import logging
import math
import random

import torch as pt
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise

from data.utils import batched, prep_batch
from evals.kl_eval import KLComputor
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repcollapse.disco_collapser import DiscoCollapser
from trainer.unlearn.repcollapse.utils import get_banned_tokens, ManualLoRA
from trainer.utils import normalize_grads

logging.basicConfig(level=logging.INFO)


class DISCO(UnlearnTrainer):
    """Discriminative Collapse: activation DISCO + post-hoc weight projection.

    First-principles approach:
    1. Activation DISCO identifies selective INPUT directions (high forget/retain ratio)
    2. Standard backprop computes the raw weight gradient
    3. Post-hoc projection constrains ΔW to only modify selective input directions
    4. KL masking filters disruptive tokens
    5. LoRA adversary forces robustness
    """

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

        self.model.requires_grad_(False)
        self.base_trainable_params = []
        self.lora_params = []
        n_pcs = cfg.n_pcs_select
        eps = cfg.get("reg_eps", 1e-4)

        for layer_num in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[layer_num].mlp
            shared_collapser = DiscoCollapser(n_pcs, eps)
            down_collapser = DiscoCollapser(n_pcs, eps)
            for module in [mlp.gate_proj, mlp.up_proj, mlp.down_proj]:
                module.weight.requires_grad = True
                self.base_trainable_params.append(module.weight)
                module.register_forward_hook(self.save_act_input_hook)
                module.register_full_backward_hook(self.collapse_hook)
                module.disco_collapser = down_collapser if module is mlp.down_proj else shared_collapser

                # LoRA adversary
                if "lora_lr" in cfg:
                    module.lora_module = ManualLoRA(
                        module.weight.shape[1],
                        module.weight.shape[0],
                        cfg.lora_rank,
                    ).to(self.model.device, dtype=self.model.dtype)
                    self.lora_params.extend(module.lora_module.parameters())
                    module.register_forward_hook(self.lora_forward_hook)

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

        # Lazy init KLComputor
        if self.kl_computor is None and "retain_momentum" in self.cfg:
            self.kl_computor = KLComputor(self.model, self.retain_batches)

        # Retain activation collection (epoch 0 only, forward only)
        if self.batch_idx < self.recalc_every:
            r_batch = inputs["retain"]
            self.retain_token_mask = r_batch["attention_mask"].bool().clone()
            self.retain_token_mask[:, 0] = False
            if self.processing_class.chat_template is not None:
                for banned_token in get_banned_tokens(self.processing_class):
                    self.retain_token_mask &= r_batch["input_ids"] != banned_token

            self.recording_retain = True
            with pt.no_grad():
                model(**prep_batch(r_batch, model.device))
            self.recording_retain = False

        # KL masking: compute retain KL gradient (momentum-averaged)
        if "retain_momentum" in self.cfg and self.batch_idx >= self.recalc_every:
            r_batch = random.choice(self.retain_batches)
            model.zero_grad(set_to_none=True)
            kl, _, _ = self.kl_computor.get_kl(r_batch)
            kl.backward()
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
        output = model(**prep_batch(batch, model.device))
        forget_loss = -output.loss

        for p in self.base_trainable_params:
            p.requires_grad_(False)
        forget_loss.backward()
        for p in self.base_trainable_params:
            p.requires_grad_(True)
        self.use_hooks = False

        # LoRA adversarial update (gradient ascent — adversary tries to relearn)
        if "lora_lr" in self.cfg:
            normalize_grads(self.lora_params)
            for p in self.lora_params:
                p.data += self.cfg.lora_lr * self.args.learning_rate * p.grad
                p.grad = None

        self.batch_idx += 1
        if self.batch_idx % self.recalc_every == 0:
            for module in model.modules():
                if hasattr(module, "disco_collapser"):
                    module.disco_collapser.process_saved_vecs()

        normalize_grads(self.base_trainable_params)
        return forget_loss.detach()

    def save_act_input_hook(self, module, args, output):
        if self.recording_retain:
            retain_acts = args[0].detach()
            retain_acts = retain_acts[self.retain_token_mask]
            module.disco_collapser.add_retain_vecs(retain_acts)
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

        # Collect forget activations for DISCO covariance
        module.disco_collapser.add_forget_vecs(acts)

        if self.batch_idx < self.recalc_every:
            return

        # KL masking: filter disruptive tokens BEFORE computing gradient
        if "retain_momentum" in self.cfg:
            ref_grad = dequantize_blockwise(*module.weight.ref_grad)
            ref_grad = ref_grad.to(module.weight.dtype)
            token_disr = pt.einsum("ij,ti,tj->t", ref_grad, grads, acts)
            kl_mask = token_disr > 0
            acts = acts[kl_mask]
            grads = grads[kl_mask]

        # Standard weight gradient from surviving tokens
        W_grad = pt.einsum("ti,tj->ij", grads, acts)  # (D_out, D_in)

        # Post-hoc DISCO projection: constrain ΔW to selective input dirs
        V = module.disco_collapser.eig_vec    # (D_in, m)
        w = module.disco_collapser.weights     # (m,)
        G_proj = W_grad @ V                    # (D_out, m)
        G_weighted = G_proj * w                # reweight by selectivity
        module.weight.grad = G_weighted @ V.T  # (D_out, D_in)

    def lora_forward_hook(self, module, args, output):
        if self.use_hooks:
            return output + module.lora_module(args[0])
