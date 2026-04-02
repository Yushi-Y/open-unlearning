# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=GradOrth task_name=test
"""RepCollapse with gradient orthogonalization instead of binary KL masking.

Instead of dropping disruptive tokens entirely, projects out the disruptive
component from the gradient. Preserves more signal per step → more unlearning
within the same disruption budget.
"""
import torch as pt
from bitsandbytes.functional import dequantize_blockwise

from trainer.unlearn.repcollapse.repcollapse_trainer import RepCollapse


class GradOrth(RepCollapse):
    """RepCollapse with continuous disruption control."""

    def collapse_hook(self, module, grad_input, grad_output):
        if not self.use_hooks:
            return
        acts = module.last_act_input
        grads = grad_output[0]
        module.last_act_input = None

        if self.is_moe:
            token_mask = grads.norm(dim=1) != 0
            acts = acts[token_mask]
            grads = grads[token_mask]
            if acts.shape[0] == 0:
                return
        else:
            acts = acts[self.token_mask]
            grads = grads[self.token_mask]

        if "n_pcs" in self.cfg:
            module.act_collapser.add_vecs(acts)
            module.grad_collapser.add_vecs(grads)

        if self.batch_idx < self.recalc_every:
            return

        if "n_pcs" in self.cfg:
            acts = module.act_collapser.collapse(acts)
            grads = module.grad_collapser.collapse(grads)

        # Gradient orthogonalization instead of binary KL masking
        if "retain_momentum" in self.cfg:
            ref_grad = dequantize_blockwise(*module.weight.ref_grad)
            ref_grad = ref_grad.to(module.weight.dtype)

            # Compute per-token disruption direction: how each token's update
            # projects onto the retain gradient
            disr_grad = acts @ ref_grad.T  # (T, D_out) — disruption direction per token
            disr_grad = disr_grad / (disr_grad.norm(dim=1, keepdim=True) + 1e-8)

            # Project out the disruptive component from grads
            projections = pt.einsum("tg,tg->t", disr_grad, grads).unsqueeze(1)  # (T, 1)
            projections = projections.clamp(max=0)  # only remove positive disruption
            grads = grads - projections * disr_grad  # orthogonalize

        module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
