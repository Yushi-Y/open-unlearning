"""DISCO v2: Orthogonalized directions + selectivity filter."""
import logging

import torch as pt
from trainer.unlearn.repcollapse.online_covariance import OnlineCovariance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscoV2Collapser:
    """DISCO with orthogonalized eigenvectors and per-token selectivity scoring.

    Fix 1: QR-orthogonalize DISCO directions after computing them, then
           recompute per-direction λ in the orthonormal basis. Clean subtraction.
    Fix 2: Provides token_selectivity() for DISCO-native disruption control.
    """

    def __init__(self, n_pcs_select: int, reg_eps: float = 1e-4):
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

        self.mean = self.forget_cov.mean().to(pt.float32)
        Sigma_f = self.forget_cov.cov().to(pt.float32)
        Sigma_r = self.retain_cov.cov().to(pt.float32)

        Sigma_f = (Sigma_f + Sigma_f.T) / 2
        Sigma_r = (Sigma_r + Sigma_r.T) / 2

        # Step 1: Compute DISCO subspace via diagonal whitening
        diag_r = Sigma_r.diagonal().clamp(min=self.reg_eps)
        inv_sqrt_diag = 1.0 / diag_r.sqrt()

        whitened = Sigma_f * inv_sqrt_diag.unsqueeze(0) * inv_sqrt_diag.unsqueeze(1)
        whitened = (whitened + whitened.T) / 2

        _, _, V = pt.svd_lowrank(whitened, q=self.n_pcs_select)

        # Transform back to original space
        eigenvectors = V * inv_sqrt_diag.unsqueeze(1)

        # Step 2: QR-orthogonalize — same subspace, orthonormal basis
        Q, _ = pt.linalg.qr(eigenvectors)

        # Step 3: Recompute per-direction λ in orthonormal basis
        # λ_i = q_i^T Σ_f q_i / q_i^T Σ_r q_i
        forget_var = (Q.T @ Sigma_f @ Q).diagonal()
        retain_var = (Q.T @ Sigma_r @ Q).diagonal().clamp(min=self.reg_eps)
        eigenvalues = forget_var / retain_var

        # Sort by eigenvalue (descending) since QR may change ordering
        sort_idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[sort_idx]
        Q = Q[:, sort_idx]

        # Mahalanobis weights: same formula as before
        self.weights = (1.0 - 1.0 / eigenvalues.clamp(min=1e-6)).clamp(min=0)
        self.eig_vec = Q  # (D, m) — ORTHONORMAL
        self.eigenvalues = eigenvalues

        n_active = (self.weights > 0.01).sum().item()
        self.n_active = n_active
        logger.info(
            f"DISCOv2: {n_active}/{Q.shape[1]} active dirs (dim={Q.shape[0]}), "
            f"top-5 λ: {[f'{v:.1f}' for v in eigenvalues[:5].tolist()]}"
        )
        self._reset_vecs()

    def collapse(self, vecs):
        """Mahalanobis with orthonormal DISCO directions. Clean subtraction."""
        centered = vecs - self.mean

        projected = centered @ self.eig_vec  # (T, m) — orthonormal, clean projection
        removal = projected * (1.0 - self.weights)
        mahal_dirs = centered - removal @ self.eig_vec.T  # clean, no leakage

        # Double projection
        mahal_norm = mahal_dirs / (mahal_dirs.norm(dim=1, keepdim=True) + 1e-8)
        proj_strength = (mahal_norm * centered).sum(dim=1, keepdim=True)
        return (proj_strength * mahal_norm).to(vecs.dtype)

    def token_selectivity(self, vecs):
        """Per-token selectivity: fraction of energy in selective vs shared dirs."""
        centered = (vecs - self.mean).to(pt.float32)
        proj = centered @ self.eig_vec  # (T, m)
        selective_energy = (proj ** 2 * self.weights).sum(dim=1)
        shared_energy = (proj ** 2 * (1.0 - self.weights)).sum(dim=1)
        # Ratio: high = token is mostly selective, low = mostly shared
        return selective_energy / (shared_energy + 1e-8)
