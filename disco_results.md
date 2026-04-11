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
| 0 | threshold=1.5, max_pcs=400, act-only, KL+LoRA | 0.059 (ep3) / 0.094 (ep2) | 0.041 (ep3) / 0.006 (ep2) | Yes (ep3) | discard | Most layers 0/400 dirs above 1.5, fallback to top 10. Too few PCs collapsed → weak unlearning. |
| 1 | threshold=0 (all 400 PCs), ratio-weighted, act-only, KL+LoRA | 0.069 (ep3) / 0.106 (ep2) | 0.033 (ep3) / 0.007 (ep2) | Yes (ep3) | discard | All 400 PCs kept, ratio as Mahalanobis weight. Still broken at ep3. Epoch 2 recall_prob (0.106) much worse than RepCollapse (0.032). Ratio-weighting weaker than eigenvalue-weighting. |

### Key diagnostic
- **Iter 0**: Cutoff too aggressive — most layers 0/400 dirs above 1.5. Fallback to 10 PCs too weak.
- **Iter 1**: All 400 PCs kept but ratio-weighted. Ratios have much less dynamic range than eigenvalues (max ratio ~8x vs eigenvalue range ~1000x), so Mahalanobis suppression is weaker. This is why ratio ≈ eigenvalue ranking (Spearman ρ=0.86-0.99) but ratio ≠ eigenvalue **magnitude**.
- **Core finding**: The Mahalanobis formula uses eigenvalue *magnitude* (not just ranking). PCA eigenvalues have huge dynamic range → strong suppression of top PCs. Ratios have small dynamic range → weak suppression. Even identical rankings give different results when magnitudes differ.
- **Activation-only vs dual collapse**: Both iters use activation-only. Without gradient collapse baseline for comparison, can't isolate this effect yet.

### Conclusion
PCA eigenvalue Mahalanobis > ratio Mahalanobis > ratio cutoff.
The eigenvalue magnitude's large dynamic range is essential, not just the direction ranking.
This closes the DISCO/selective approach — PCA is not just near-optimal in ranking, it's optimal in scaling too.
