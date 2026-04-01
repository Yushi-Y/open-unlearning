import logging

import torch as pt
from trainer.unlearn.repcollapse.online_covariance import OnlineCovariance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscoCollapser:
    """Discriminative Collapse.

    Step 1: Compute generalised eigenvectors (DISCO directions, ranked by λ)
    Step 2: Suppress SHARED directions (low λ) via reweighting
    Step 3: Project centered data onto the resulting cleaned direction (1D per token)

    This matches RepCollapse's double-projection structure:
    - whitening step controls WHICH directions are suppressed
    - projection step controls MAGNITUDE (bounded by ||centered||)
    """

    mean: pt.Tensor
    eig_val: pt.Tensor
    eig_vec: pt.Tensor

    def __init__(self, n_pcs_select: int, reg_eps: float = 1e-4, n_pcs_exclude: int = 0):
        self.n_pcs_select = n_pcs_select
        self.reg_eps = reg_eps
        self._reset_vecs()

    def _reset_vecs(self):
        self.forget_cov = OnlineCovariance(dtype=pt.bfloat16)
        self.retain_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_forget_data = False
        self._has_retain_data = False

    def add_forget_vecs(self, vecs):
        self.forget_cov.add_all(vecs)
        self._has_forget_data = True

    def add_retain_vecs(self, vecs):
        self.retain_cov.add_all(vecs)
        self._has_retain_data = True

    def process_saved_vecs(self):
        if not self._has_forget_data or not self._has_retain_data:
            return

        self.mean = self.forget_cov.mean.to(pt.float32)
        Sigma_f = self.forget_cov.cov().to(pt.float32)
        Sigma_r = self.retain_cov.cov().to(pt.float32)
        D = Sigma_f.shape[0]

        Sigma_f = (Sigma_f + Sigma_f.T) / 2
        Sigma_r = (Sigma_r + Sigma_r.T) / 2

        # Generalised eigenvalue: Sigma_f v = λ Sigma_r v
        k_r = min(self.n_pcs_select * 2, D)
        _, S_r, V_r = pt.svd_lowrank(Sigma_r, q=k_r)
        inv_sqrt_S = 1.0 / S_r.clamp(min=self.reg_eps).sqrt()
        W = V_r * inv_sqrt_S.unsqueeze(0)

        M_reduced = W.T @ Sigma_f @ W
        M_reduced = (M_reduced + M_reduced.T) / 2
        _, eigenvalues, U_reduced = pt.svd_lowrank(M_reduced, q=self.n_pcs_select)

        eigenvectors = W @ U_reduced
        eigenvectors = eigenvectors / eigenvectors.norm(dim=0, keepdim=True)

        # Per-direction weight: suppress shared (low λ), keep selective (high λ)
        # w(λ) = max(0, 1 - 1/λ): λ≤1 → 0, λ=2 → 0.5, λ=93 → 0.99
        self.weights = (1.0 - 1.0 / eigenvalues.clamp(min=1e-6)).clamp(min=0)
        self.eig_vec = eigenvectors  # (D, m)
        self.eigenvalues = eigenvalues

        n_active = (self.weights > 0.01).sum().item()
        logger.info(
            f"DISCO: {n_active}/{eigenvectors.shape[1]} active dirs (dim={D}), "
            f"top-5 λ: {[f'{v:.1f}' for v in eigenvalues[:5].tolist()]}"
        )
        self._reset_vecs()

    def collapse(self, vecs):
        """
        1. Project centered data onto DISCO directions
        2. Reweight: suppress shared, keep selective
        3. Reconstruct a "cleaned direction" per token
        4. Project centered data onto that cleaned direction (1D, bounded magnitude)
        """
        centered = vecs - self.mean

        # Project onto DISCO directions and reweight
        projected = centered @ self.eig_vec              # (T, m)
        weighted = projected * self.weights               # suppress shared
        # Cleaned direction = reconstruction from weighted components
        cleaned_dir = weighted @ self.eig_vec.T           # (T, D)

        # Double projection: project centered onto cleaned direction (like RepCollapse)
        # This bounds the output magnitude
        cleaned_norm = cleaned_dir / (cleaned_dir.norm(dim=1, keepdim=True) + 1e-8)
        proj_strength = (cleaned_norm * centered).sum(dim=1, keepdim=True)
        return (proj_strength * cleaned_norm).to(vecs.dtype)
