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

        self.mean = self.forget_cov.mean().to(pt.float32)
        Sigma_f = self.forget_cov.cov().to(pt.float32)
        Sigma_r = self.retain_cov.cov().to(pt.float32)
        D = Sigma_f.shape[0]

        Sigma_f = (Sigma_f + Sigma_f.T) / 2
        Sigma_r = (Sigma_r + Sigma_r.T) / 2

        # Store diagonals for diagonal_collapse fallback
        self.forget_diag = Sigma_f.diagonal().clamp(min=self.reg_eps)
        self.retain_diag = Sigma_r.diagonal().clamp(min=self.reg_eps)

        # Generalised eigenvalue: Sigma_f v = λ Sigma_r v
        # Use diagonal Sigma_r for full-rank whitening (all D dimensions).
        # Low-rank SVD of Sigma_r only captures k_r/D of variance,
        # making eigenvalues unreliable in the remaining subspace.
        diag_r = Sigma_r.diagonal().clamp(min=self.reg_eps)
        inv_sqrt_diag = 1.0 / diag_r.sqrt()  # (D,)

        # Whitened forget covariance: D^{-1/2} Sigma_f D^{-1/2}
        # Eigenvectors of this = directions maximising forget/retain variance ratio
        whitened = Sigma_f * inv_sqrt_diag.unsqueeze(0) * inv_sqrt_diag.unsqueeze(1)
        whitened = (whitened + whitened.T) / 2

        _, eigenvalues, V = pt.svd_lowrank(whitened, q=self.n_pcs_select)

        # Transform back to original space: v = D^{-1/2} u
        eigenvectors = V * inv_sqrt_diag.unsqueeze(1)
        eigenvectors = eigenvectors / eigenvectors.norm(dim=0, keepdim=True)

        # Per-direction weight: soft Mahalanobis scaling (like RepCollapse)
        # Normalize by min so all directions contribute; relative ranking preserved
        # w = 1 - λ_min/λ: min dir → 0, max dir → ~1, no hard cutoff
        self.weights = 1.0 - eigenvalues.min() / eigenvalues.clamp(min=1e-6)
        self.eig_vec = eigenvectors  # (D, m)
        self.eigenvalues = eigenvalues

        n_active = (self.weights > 0.01).sum().item()
        self.n_active = n_active
        logger.info(
            f"DISCO: {n_active}/{eigenvectors.shape[1]} active dirs (dim={D}), "
            f"top-5 λ: {[f'{v:.1f}' for v in eigenvalues[:5].tolist()]}"
        )
        self._reset_vecs()

    def collapse(self, vecs):
        """Mahalanobis-style: subtract shared components, keep selective + residual.

        Starts from the full centered vector and removes identified shared
        directions.  Preserves all non-DISCO dimensions for stronger signal.
        """
        centered = vecs - self.mean

        # Project onto DISCO directions
        projected = centered @ self.eig_vec  # (T, m)

        # Remove shared components: (1 - weight) fraction of each direction
        # shared (λ≤1, w=0) → fully removed; selective (high λ, w≈1) → kept
        removal = projected * (1.0 - self.weights)  # (T, m)
        mahal_dirs = centered - removal @ self.eig_vec.T  # (T, D)

        # Double projection: bounded magnitude (like RepCollapse)
        mahal_norm = mahal_dirs / (mahal_dirs.norm(dim=1, keepdim=True) + 1e-8)
        proj_strength = (mahal_norm * centered).sum(dim=1, keepdim=True)
        return (proj_strength * mahal_norm).to(vecs.dtype)

    def token_selectivity(self, vecs):
        """Per-token selectivity score: fraction of activation energy in selective dirs."""
        centered = (vecs - self.mean).to(pt.float32)
        projected = centered @ self.eig_vec  # (T, m)
        selective_energy = (projected ** 2 * self.weights).sum(dim=1)  # (T,)
        total_energy = (centered ** 2).sum(dim=1).clamp(min=1e-8)  # (T,)
        return (selective_energy / total_energy).to(vecs.dtype)  # (T,) in [0, 1]

    def diagonal_collapse(self, vecs):
        """Diagonal DISCO fallback: per-dimension forget/retain variance ratio.

        Used when full DISCO finds 0 active directions (e.g. gradient space
        where retain dominates). Still uses forget/retain ratio (DISCO math),
        just diagonal instead of full covariance.
        """
        # Per-dimension ratio from the stored covariance diagonals
        diag_f = self.forget_diag  # stored during process_saved_vecs
        diag_r = self.retain_diag
        ratio = diag_f / diag_r.clamp(min=self.reg_eps)
        dim_weights = (1.0 - 1.0 / ratio.clamp(min=1e-6)).clamp(min=0)
        return (vecs * dim_weights.to(vecs.dtype)).to(vecs.dtype)
