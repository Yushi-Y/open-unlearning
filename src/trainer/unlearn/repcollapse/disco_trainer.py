# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=DISCO task_name=test_disco
import logging
import math

import torch as pt

from data.utils import batched, prep_batch
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repcollapse.disco_collapser import DiscoCollapser
from trainer.unlearn.repcollapse.utils import get_banned_tokens
from trainer.utils import normalize_grads

logging.basicConfig(level=logging.INFO)


class DISCO(UnlearnTrainer):
    """Discriminative Collapse with DISCO directions on both acts and grads.

    Uses generalised eigenvalue (forget/retain variance ratio) to identify
    selective directions in BOTH activation and gradient space.  Collapses
    both sides of the weight gradient for focused unlearning.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.use_hooks = False
        self.recording_retain = False
        self.recording_retain_grads = False
        self.batch_idx = 0
        self.recalc_every = math.ceil(
            len(self.train_dataset) / self.args.per_device_train_batch_size
        )
        logging.info(f"{self.recalc_every=}")
        assert self.args.gradient_accumulation_steps == 1

        self.model.requires_grad_(False)
        self.base_trainable_params = []
        n_pcs = cfg.n_pcs_select
        eps = cfg.get("reg_eps", 1e-4)

        for layer_num in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[layer_num].mlp
            shared_act_collapser = DiscoCollapser(n_pcs, eps)
            down_act_collapser = DiscoCollapser(n_pcs, eps)
            shared_grad_collapser = DiscoCollapser(n_pcs, eps)
            down_grad_collapser = DiscoCollapser(n_pcs, eps)
            for module in [mlp.gate_proj, mlp.up_proj, mlp.down_proj]:
                module.weight.requires_grad = True
                self.base_trainable_params.append(module.weight)
                module.register_forward_hook(self.save_act_input_hook)
                module.register_full_backward_hook(self.collapse_hook)
                if module is mlp.down_proj:
                    module.act_collapser = down_act_collapser
                    module.grad_collapser = down_grad_collapser
                else:
                    module.act_collapser = shared_act_collapser
                    module.grad_collapser = shared_grad_collapser

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # Retain data collection (epoch 0 only)
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
                model(**prep_batch(r_batch, model.device))
            self.recording_retain = False

            # Backward pass: collect retain gradients
            self.recording_retain_grads = True
            model.zero_grad(set_to_none=True)
            output = model(**prep_batch(r_batch, model.device))
            for p in self.base_trainable_params:
                p.requires_grad_(False)
            output.loss.backward()
            for p in self.base_trainable_params:
                p.requires_grad_(True)
            self.recording_retain_grads = False

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

        self.batch_idx += 1
        if self.batch_idx % self.recalc_every == 0:
            for module in model.modules():
                if hasattr(module, "act_collapser"):
                    module.act_collapser.process_saved_vecs()
                if hasattr(module, "grad_collapser"):
                    module.grad_collapser.process_saved_vecs()

        normalize_grads(self.base_trainable_params)
        return forget_loss.detach()

    def save_act_input_hook(self, module, args, output):
        if self.recording_retain:
            retain_acts = args[0].detach()
            retain_acts = retain_acts[self.retain_token_mask]
            module.act_collapser.add_retain_vecs(retain_acts)
            return
        if not self.use_hooks:
            return
        module.last_act_input = args[0].detach()

    def collapse_hook(self, module, grad_input, grad_output):
        # Collect retain gradient statistics during epoch 0
        if self.recording_retain_grads:
            grads = grad_output[0].detach()
            grads = grads[self.retain_token_mask]
            module.grad_collapser.add_retain_vecs(grads)
            return

        if not self.use_hooks:
            return
        acts = module.last_act_input
        grads = grad_output[0]
        module.last_act_input = None

        acts = acts[self.token_mask]
        grads = grads[self.token_mask]

        # Collect forget statistics for both act and grad collapsers
        module.act_collapser.add_forget_vecs(acts)
        module.grad_collapser.add_forget_vecs(grads)

        if self.batch_idx < self.recalc_every:
            return

        # Collapse acts via DISCO; collapse grads only if selective dirs found
        acts = module.act_collapser.collapse(acts)
        if hasattr(module.grad_collapser, "n_active") and module.grad_collapser.n_active > 0:
            grads = module.grad_collapser.collapse(grads)

        module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
