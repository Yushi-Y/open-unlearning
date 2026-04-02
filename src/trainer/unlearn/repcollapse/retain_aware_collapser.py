"""Retain-aware CovCollapser: PCA directions ranked by forget/retain ratio."""
import torch as pt
from trainer.unlearn.repcollapse.collapsers import CovCollapser, _proj_to_mahal_dirs
from trainer.unlearn.repcollapse.online_covariance import OnlineCovariance


class RetainAwareCovCollapser(CovCollapser):
    """Drop-in replacement for CovCollapser.

    Same PCA directions as CovCollapser, but eigenvalues are reweighted
    by the forget/retain variance ratio.  Directions that are selective
    (high forget, low retain variance) are preserved more; directions that
    are shared (similar variance in both) are suppressed more.
    """

    def __init__(self, PCs_to_use: int, reg_eps: float = 1e-4):
        super().__init__(PCs_to_use)
        self.reg_eps = reg_eps
        self.retain_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_retain_data = False

    def add_retain_vecs(self, vecs):
        self.retain_cov.add_all(vecs)
        self._has_retain_data = True

    def _reset_vecs(self):
        super()._reset_vecs()
        self.retain_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_retain_data = False

    def process_saved_vecs(self):
        """Compute PCA directions, then reweight by forget/retain ratio."""
        # Standard PCA (same as CovCollapser)
        super().process_saved_vecs()

        if not self._has_retain_data:
            return  # fall back to standard PCA ranking

        # Compute retain variance along each PCA direction
        Sigma_r = self.retain_cov.cov().to(pt.float32)
        V = self.eig_vec  # (D, k) — PCA directions from forget data

        # retain_var_i = V_i^T @ Sigma_r @ V_i for each direction i
        # Efficient: (V.T @ Sigma_r @ V).diagonal()
        retain_var = (V.T @ Sigma_r @ V).diagonal().clamp(min=self.reg_eps)

        # Forget variance per direction = original eigenvalues (before normalization)
        # eig_val was normalized by min, so recover: forget_var = eig_val * min_val
        # But we only need the RATIO, and eig_val is monotonic with forget_var
        # So: ratio = eig_val * const / retain_var ∝ eig_val / retain_var
        ratio = self.eig_val / retain_var

        # Re-normalize so min = 1 (same convention as CovCollapser)
        self.eig_val = ratio / ratio.min()

        self.retain_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_retain_data = False
