# SelectiveCollapse / DISCO Experiment Results

Baseline: RepCollapse (PCA + Mahalanobis, n_pcs=400) = **0.032 robustness** on WMDP-Bio, Llama-3.2-3B.

## DISCO iterations (generalised eigenvalue, both acts+grads)

| iter | change | recall_prob | status | notes |
|------|--------|-------------|--------|-------|
| 0 | DISCO baseline (full Σ_r, low-rank) | 0.115 | discard | barely unlearns |
| 1 | + Mahalanobis subtraction (low-rank Σ_r) | 0.114 | discard | same |
| 2 | + diagonal Σ_r (full-rank whitening) | 0.114 | discard | eigenvalues 10x better but same |
| 3 | + gradient collapse (retain backward) | 0.096 | keep | gate/up grads zeroed (λ<1) |
| 7 | + KL token masking | 0.066 | keep | controls disruption |
| 8 | + LoRA adversary | 0.037 | BEST | within 20% of RepCollapse |

## SelectiveCollapse iterations (PCA dirs + ratio cutoff, activation-only)

| iter | change | recall_prob | wikitext_kl | broken | status | notes |
|------|--------|-------------|-------------|--------|--------|-------|
| 0 | threshold=1.5, max_pcs=400, act-only, KL+LoRA | 0.059 (ep3) / 0.094 (ep2) | 0.041 (ep3) / 0.006 (ep2) | Yes (ep3) | discard | Most layers 0/400 dirs above 1.5, fallback to top 10. Too few PCs collapsed → weak unlearning. Broken at ep3. |

### Diagnostic (iter 0)
- **Early layers (L0-L5)**: 0 directions above threshold, ratios 0.2-1.1. Retain variance dominates forget.
- **Middle layers (L8-L14)**: 1-12 directions above threshold, best ratios 3-8x. This is where selectivity exists.
- **Late layers (L20+)**: 0-4 directions above threshold, ratios 0.3-1.5.
- **Key insight**: PCA Mahalanobis works by suppressing ALL 400 directions proportionally to variance. The ratio-based cutoff loses the benefit of broad suppression — the "non-selective" directions still need to be suppressed for effective unlearning.

### Next steps
- [ ] Try lower threshold (1.0) to include more directions
- [ ] Try keeping all 400 PCs but weighting by ratio instead of eigenvalue (no cutoff)
- [ ] Try hybrid: PCA Mahalanobis (all dirs) + ratio-boosted suppression for selective dirs
