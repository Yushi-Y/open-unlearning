# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=TwoPhase task_name=test
"""Two-phase DISCO unlearning: eigenvalue surgery + DISCO collapse fine-tuning.

Phase 1 (epoch 0): Collect stats, compute DISCO directions, perform surgery
Phase 2 (epochs 1+): Full DISCO iter8 (collapse + KL masking + LoRA adversary)

The surgery removes bulk selective weight components in one shot.
DISCO fine-tuning then cleans up remaining knowledge with disruption control.
"""
from trainer.unlearn.repcollapse.disco_trainer import DISCO

import logging
import torch as pt

logging.basicConfig(level=logging.INFO)


class TwoPhase(DISCO):
    """EigenSurgery initialization + DISCO collapse fine-tuning."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.surgery_done = False

    def _do_surgery(self, model):
        alpha = self.cfg.get("surgery_alpha", 0.1)
        logging.info(f"Phase 1: Eigenvalue surgery with α={alpha}")
        with pt.no_grad():
            for module in model.modules():
                if hasattr(module, "act_collapser") and hasattr(module.act_collapser, "eig_vec"):
                    V = module.act_collapser.eig_vec.to(module.weight.dtype)
                    w = module.act_collapser.weights.to(module.weight.dtype)
                    proj = module.weight.data @ V
                    weighted = proj * w
                    W_selective = weighted @ V.T
                    module.weight.data -= alpha * W_selective
        self.surgery_done = True
        logging.info("Phase 1 complete. Switching to DISCO fine-tuning.")

    def training_step(self, model, inputs, num_items_in_batch=None):
        result = super().training_step(model, inputs, num_items_in_batch)

        # After first epoch's process_saved_vecs, do surgery once
        if not self.surgery_done and self.batch_idx >= self.recalc_every:
            # Check if collapsers have been processed
            for module in model.modules():
                if hasattr(module, "act_collapser") and hasattr(module.act_collapser, "eig_vec"):
                    self._do_surgery(model)
                    break

        return result
