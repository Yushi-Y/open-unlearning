"""Selective Collapser: PCA directions + selectivity-aware cutoff.

Combines PCA stability (forget-only directions) with DISCO's selectivity insight
(forget/retain variance ratio). Replaces fixed n_pcs with a data-driven threshold.

Key differences from CovCollapser:
  - Tracks both forget and retain covariance
  - Auto-selects number of PCs based on selectivity ratio λ_i > threshold
  - Uses ratio (not raw eigenvalue) for Mahalanobis scaling

Key differences from DiscoCollapser:
  - Uses PCA directions (stable, forget-only) not generalised eigenvectors
  - Uses diagonal Σ_r (stable) only for computing per-direction ratio
  - Activation-only — no gradient collapse
"""
import logging

import torch as pt
from trainer.unlearn.repcollapse.online_covariance import OnlineCovariance

logger = logging.getLogger(__name__)


def _get_mahal_dirs(centered, eig_val, eig_vec):
    """Mahalanobis-style suppression: suppress high-eigenvalue directions."""
    projected = centered @ eig_vec  # (N, k)
    proj_diff = projected - projected / (eig_val / eig_val.min())
    return centered - proj_diff @ eig_vec.T


def _proj_to_mahal_dirs(centered, mahal_dirs):
    """Double projection: project centered onto Mahalanobis direction (bounded magnitude)."""
    mahal_dirs_norm = mahal_dirs / (mahal_dirs.norm(dim=1, keepdim=True) + 1e-8)
    proj_strengths = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
    return proj_strengths * mahal_dirs_norm


class SelectiveCollapser:
    """PCA directions with selectivity-aware cutoff.

    process_saved_vecs():
      1. PCA on forget covariance → stable directions
      2. Measure retain variance along each PCA direction (diagonal)
      3. Compute ratio λ_i = forget_var_i / retain_var_i
      4. Keep only directions with λ > threshold (auto n_pcs)
      5. Use ratio as eigenvalue for Mahalanobis scaling

    collapse(): Same Mahalanobis + double projection as CovCollapser.
    """

    mean: pt.Tensor
    eig_val: pt.Tensor
    eig_vec: pt.Tensor

    def __init__(self, max_pcs: int = 400, selectivity_threshold: float = 1.5,
                 reg_eps: float = 1e-4, pca_source: str = "forget"):
        self.max_pcs = max_pcs
        self.selectivity_threshold = selectivity_threshold
        self.reg_eps = reg_eps
        self.pca_source = pca_source  # "forget", "retain", or "eigenvalue"
        self._reset_vecs()

    def _reset_vecs(self):
        self.forget_cov = OnlineCovariance(dtype=pt.bfloat16)
        self.retain_cov = OnlineCovariance(dtype=pt.bfloat16)
        self._has_forget_data = False
        self._has_retain_data = False

    def add_forget_vecs(self, vecs):
        self.forget_cov.add_vecs(vecs)
        self._has_forget_data = True

    def add_retain_vecs(self, vecs):
        self.retain_cov.add_vecs(vecs)
        self._has_retain_data = True

    def process_saved_vecs(self):
        if not self._has_forget_data or not self._has_retain_data:
            return

        self.mean = self.forget_cov.mean.to(pt.float32)
        Sigma_f = self.forget_cov.get_cov().to(pt.float32)
        Sigma_f = (Sigma_f + Sigma_f.T) / 2
        Sigma_r = self.retain_cov.get_cov().to(pt.float32)
        Sigma_r = (Sigma_r + Sigma_r.T) / 2
        D = Sigma_f.shape[0]

        if self.pca_source == "diagonal":
            # Diagonal Mahalanobis: per-dimension forget variance, no eigenvectors
            # Use standard basis (top max_pcs dims by variance)
            diag_var = Sigma_f.diagonal().clamp(min=self.reg_eps)
            topk = min(self.max_pcs, D)
            idx = diag_var.argsort(descending=True)[:topk]
            V = pt.zeros(D, topk, device=Sigma_f.device)
            V[idx, pt.arange(topk)] = 1.0  # standard basis vectors
            S = diag_var[idx]
            self.eig_vec = V
            self.eig_val = S / S.min()
            logger.info(
                f"SelectiveCollapser[diagonal]: {topk} dims, "
                f"var range [{S.min():.2f}, {S.max():.2f}], "
                f"dynamic range {S.max()/S.min():.0f}x"
            )

        elif self.pca_source == "random":
            # Random orthogonal directions + forget variance along them
            k = self.max_pcs
            Q, _ = pt.linalg.qr(pt.randn(D, k, device=Sigma_f.device))
            S = (Q.T @ Sigma_f @ Q).diagonal().clamp(min=self.reg_eps)
            self.eig_vec = Q
            self.eig_val = S / S.min()
            logger.info(
                f"SelectiveCollapser[random]: {k} dirs, "
                f"var range [{S.min():.2f}, {S.max():.2f}], "
                f"dynamic range {S.max()/S.min():.0f}x"
            )

        elif self.pca_source == "retain":
            self.mean = self.retain_cov.mean.to(pt.float32)
            _, S, V = pt.svd_lowrank(Sigma_r, q=self.max_pcs)
            self._set_ratio_scaling(V, Sigma_f, Sigma_r, "retain")

        elif self.pca_source == "eigenvalue":
            _, S, V = pt.svd_lowrank(Sigma_f, q=self.max_pcs)
            self.eig_vec = V
            self.eig_val = S / S.min()
            logger.info(
                f"SelectiveCollapser[eigenvalue]: {self.max_pcs} dirs, "
                f"eig range [{S.min():.2f}, {S.max():.2f}], "
                f"dynamic range {S.max()/S.min():.0f}x"
            )

        else:  # "forget" — ratio-based scaling
            _, S, V = pt.svd_lowrank(Sigma_f, q=self.max_pcs)
            self._set_ratio_scaling(V, Sigma_f, Sigma_r, "forget")

        self._reset_vecs()

    def _set_ratio_scaling(self, V, Sigma_f, Sigma_r, label):
        """Ratio-based Mahalanobis scaling with optional threshold cutoff."""
        forget_var = (V.T @ Sigma_f @ V).diagonal().clamp(min=self.reg_eps)
        retain_var = (V.T @ Sigma_r @ V).diagonal().clamp(min=self.reg_eps)
        ratio = forget_var / retain_var

        if self.selectivity_threshold > 0:
            mask = ratio > self.selectivity_threshold
            n_kept = mask.sum().item()
            if n_kept == 0:
                n_kept = min(10, self.max_pcs)
                mask[:n_kept] = True
                logger.warning(
                    f"SelectiveCollapser: no dirs above threshold "
                    f"{self.selectivity_threshold}, fallback to {n_kept}"
                )
            self.eig_vec = V[:, mask]
            kept_ratio = ratio[mask]
        else:
            self.eig_vec = V
            kept_ratio = ratio
            n_kept = self.max_pcs

        self.eig_val = kept_ratio / kept_ratio.min()
        logger.info(
            f"SelectiveCollapser[{label}]: {n_kept}/{self.max_pcs} dirs, "
            f"ratio range [{kept_ratio.min():.2f}, {kept_ratio.max():.2f}], "
            f"top-5: {[f'{v:.2f}' for v in ratio[:5].tolist()]}"
            )

    def collapse(self, vecs):
        centered = vecs - self.mean
        mahal_dirs = _get_mahal_dirs(centered, self.eig_val, self.eig_vec)
        return _proj_to_mahal_dirs(centered, mahal_dirs).to(vecs.dtype)
