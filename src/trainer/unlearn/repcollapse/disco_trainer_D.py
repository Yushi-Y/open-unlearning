# Approach D: Standard gradient ascent + post-hoc weight projection
# Normal training, but after each step remove ΔW components along shared directions
import logging
import math
import copy

import torch as pt

from data.utils import prep_batch
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repcollapse.disco_collapser import DiscoCollapser
from trainer.unlearn.repcollapse.utils import get_banned_tokens
from trainer.utils import normalize_grads

logging.basicConfig(level=logging.INFO)


class DISCO_D(UnlearnTrainer):
    """Approach D: Normal gradient ascent, then clean the weight update."""

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

        # Store original weights for computing ΔW
        self.model.requires_grad_(False)
        self.base_trainable_params = []
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
                module.disco_collapser = down_collapser if module is mlp.down_proj else shared_collapser
                # Save original weight
                module.original_weight = module.weight.data.clone()

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # Retain activation collection (epoch 1 only)
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

        # Standard forget gradient ascent (NO collapse on gradient)
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
        forget_loss.backward()
        self.use_hooks = False

        # Collect forget activations for covariance
        # (already done in hook)

        self.batch_idx += 1
        if self.batch_idx % self.recalc_every == 0:
            for module in model.modules():
                if hasattr(module, "disco_collapser"):
                    module.disco_collapser.process_saved_vecs()

        # Post-hoc: clean each weight's accumulated ΔW
        if self.batch_idx >= self.recalc_every:
            for module in model.modules():
                if hasattr(module, "disco_collapser") and hasattr(module.disco_collapser, "weights"):
                    with pt.no_grad():
                        delta_W = module.weight.data - module.original_weight
                        V = module.disco_collapser.eig_vec  # (D_in, m)
                        w = module.disco_collapser.weights    # (m,)

                        # Project ΔW onto DISCO directions (row space)
                        proj = delta_W @ V           # (D_out, m)
                        # Reweight: keep selective, remove shared
                        cleaned_proj = proj * w      # (D_out, m)
                        # Reconstruct: ΔW_clean = ΔW - proj @ V.T + cleaned @ V.T
                        delta_clean = delta_W + (cleaned_proj - proj) @ V.T

                        module.weight.data = module.original_weight + delta_clean

        normalize_grads(self.base_trainable_params)
        return forget_loss.detach()

    def save_act_input_hook(self, module, args, output):
        if self.recording_retain:
            retain_acts = args[0].detach()
            retain_acts = retain_acts[self.retain_token_mask]
            module.disco_collapser.add_retain_vecs(retain_acts)
            return
        if self.use_hooks:
            acts = args[0].detach()
            acts = acts[self.token_mask]
            module.disco_collapser.add_forget_vecs(acts)
