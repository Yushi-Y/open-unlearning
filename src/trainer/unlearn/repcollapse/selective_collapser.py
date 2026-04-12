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
                 reg_eps: float = 1e-4, pca_source: str = "forget",
                 contrastive_alpha: float = 0.5, boost_beta: float = 0.0):
        self.max_pcs = max_pcs
        self.selectivity_threshold = selectivity_threshold
        self.reg_eps = reg_eps
        self.pca_source = pca_source
        self.contrastive_alpha = contrastive_alpha
        self.boost_beta = boost_beta
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

        if self.pca_source in ("whiten", "whiten_power", "whiten_contrastive", "whiten_ratio"):
            self.mean = self.forget_cov.mean.to(pt.float32)
            sigma_f = Sigma_f.diagonal().clamp(min=self.reg_eps).sqrt()
            sigma_r = Sigma_r.diagonal().clamp(min=self.reg_eps).sqrt()

            if self.pca_source == "whiten":
                # Standard: w_i = 1/σ_f
                self.weights = 1.0 / sigma_f
                label = "standard"
            elif self.pca_source == "whiten_power":
                # Power: w_i = 1/σ_f^β (β>1 sharpens suppression)
                beta = self.boost_beta if self.boost_beta > 0 else 2.0
                self.weights = 1.0 / sigma_f.pow(beta)
                label = f"power β={beta}"
            elif self.pca_source == "whiten_contrastive":
                # Contrastive: w_i = 1/max(σ_f - α·σ_r, ε)
                alpha = self.contrastive_alpha
                excess = (sigma_f - alpha * sigma_r).clamp(min=self.reg_eps)
                self.weights = 1.0 / excess
                label = f"contrastive α={alpha}"
            elif self.pca_source == "whiten_ratio":
                # Ratio: w_i = σ_r / σ_f (suppress where forget >> retain)
                self.weights = sigma_r / sigma_f
                label = "ratio"

            # Normalize weights so mean = 1 (preserve overall activation scale)
            self.weights = self.weights / self.weights.mean()
            dyn = self.weights.max() / self.weights.min()
            logger.info(
                f"SelectiveCollapser[whiten_{label}]: D={D}, "
                f"weight range [{self.weights.min():.3f}, {self.weights.max():.3f}], "
                f"dynamic range {dyn:.0f}x"
            )
            self._reset_vecs()
            return

        elif self.pca_source == "steer":
            # Steering vector removal: d = μ_f - μ_r, normalized
            forget_mean = self.forget_cov.mean.to(pt.float32)
            retain_mean = self.retain_cov.mean.to(pt.float32)
            self.mean = forget_mean
            d = forget_mean - retain_mean
            self.steer_dir = d / (d.norm() + 1e-8)
            logger.info(
                f"SelectiveCollapser[steer]: D={D}, "
                f"steering norm={d.norm():.2f}"
            )
            self._reset_vecs()
            return

        elif self.pca_source == "clip":
            # Variance clipping: store mean ± k·std
            self.mean = self.forget_cov.mean.to(pt.float32)
            std = Sigma_f.diagonal().clamp(min=self.reg_eps).sqrt()
            k = 2.0  # clip at 2 standard deviations
            self.clip_lo = (self.mean - k * std).to(pt.bfloat16)
            self.clip_hi = (self.mean + k * std).to(pt.bfloat16)
            logger.info(
                f"SelectiveCollapser[clip k={k}]: D={D}, "
                f"std range [{std.min():.2f}, {std.max():.2f}]"
            )
            self._reset_vecs()
            return

        elif self.pca_source == "diagonal":
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

        elif self.pca_source == "disco":
            # DISCO: diagonal Σ_r whitening → generalised eigenvectors
            # Fixes: activation-only (no grad collapse), diagonal Σ_r (stable)
            # Uses generalised eigenvalues (not ratio) for Mahalanobis scaling
            diag_r = Sigma_r.diagonal().clamp(min=self.reg_eps)
            inv_sqrt_diag = 1.0 / diag_r.sqrt()
            whitened = Sigma_f * inv_sqrt_diag.unsqueeze(0) * inv_sqrt_diag.unsqueeze(1)
            whitened = (whitened + whitened.T) / 2
            _, gen_eig, U = pt.svd_lowrank(whitened, q=self.max_pcs)
            # Transform back: v = D^{-1/2} u, then normalize
            V = U * inv_sqrt_diag.unsqueeze(1)
            V = V / V.norm(dim=0, keepdim=True)
            self.eig_vec = V
            self.eig_val = gen_eig / gen_eig.min()
            logger.info(
                f"SelectiveCollapser[disco]: {self.max_pcs} dirs, "
                f"gen_eig range [{gen_eig.min():.2f}, {gen_eig.max():.2f}], "
                f"dynamic range {gen_eig.max()/gen_eig.min():.0f}x"
            )

        elif self.pca_source == "contrastive":
            # Contrastive PCA: PCA on (Σ_f - α·Σ_r)
            # Isolates forget-specific variance by subtracting shared retain structure
            alpha = self.contrastive_alpha
            contrastive = Sigma_f - alpha * Sigma_r
            contrastive = (contrastive + contrastive.T) / 2
            _, S, V = pt.svd_lowrank(contrastive, q=self.max_pcs)
            # Keep only positive eigenvalues (forget > α·retain)
            pos_mask = S > self.reg_eps
            n_pos = pos_mask.sum().item()
            if n_pos == 0:
                # Fallback to regular PCA if all eigenvalues negative
                logger.warning(f"SelectiveCollapser[contrastive]: all eigenvalues negative, falling back to PCA")
                _, S, V = pt.svd_lowrank(Sigma_f, q=self.max_pcs)
                self.eig_vec = V
                self.eig_val = S / S.min()
            else:
                V = V[:, pos_mask]
                S = S[pos_mask]
                self.eig_vec = V
                self.eig_val = S / S.min()
            logger.info(
                f"SelectiveCollapser[contrastive α={alpha}]: {n_pos}/{self.max_pcs} pos dirs, "
                f"eig range [{S.min():.2f}, {S.max():.2f}], "
                f"dynamic range {S.max()/S.min():.0f}x"
            )

        elif self.pca_source == "boost":
            # Hybrid PCA + selectivity boost: PCA directions, eigenvalues boosted by ratio^β
            _, S, V = pt.svd_lowrank(Sigma_f, q=self.max_pcs)
            forget_var = S.clamp(min=self.reg_eps)
            retain_var = (V.T @ Sigma_r @ V).diagonal().clamp(min=self.reg_eps)
            ratio = forget_var / retain_var
            # Boost: eigenvalue * ratio^β (β=0 → pure PCA, β>0 → selective dirs suppressed more)
            beta = self.boost_beta
            boosted = S * ratio.pow(beta)
            self.eig_vec = V
            self.eig_val = boosted / boosted.min()
            logger.info(
                f"SelectiveCollapser[boost β={beta}]: {self.max_pcs} dirs, "
                f"boosted range [{boosted.min():.2f}, {boosted.max():.2f}], "
                f"dynamic range {boosted.max()/boosted.min():.0f}x"
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
        if self.pca_source.startswith("whiten"):
            # Per-dimension weighted whitening: (a - μ) * w_i
            return ((vecs - self.mean) * self.weights).to(vecs.dtype)
        elif self.pca_source == "steer":
            # Remove steering vector direction: a - (a·d̂)d̂
            centered = vecs - self.mean
            proj = (centered @ self.steer_dir).unsqueeze(1) * self.steer_dir
            return (centered - proj).to(vecs.dtype)
        elif self.pca_source == "clip":
            # Variance clipping: clamp each dim to μ ± k·σ
            return vecs.clamp(min=self.clip_lo, max=self.clip_hi).to(vecs.dtype)
        else:
            centered = vecs - self.mean
            mahal_dirs = _get_mahal_dirs(centered, self.eig_val, self.eig_vec)
            return _proj_to_mahal_dirs(centered, mahal_dirs).to(vecs.dtype)
