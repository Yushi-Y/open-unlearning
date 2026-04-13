# Full Leaderboard (all methods, all conditions)

All activation-only with KL+LoRA (MLP+attn) unless noted. recall_prob at last non-broken epoch (lower = better). **Bold** = newly filled cell, ⭐ = best in column.

| Rank | Method | Llama Bio | Qwen Bio | Llama Cyber | Qwen Cyber | Cost |
|------|--------|-----------|----------|-------------|------------|------|
| **1** | **Contrastive PI + complement proj** (NEW, iter 2) | **0.0196** ⭐ | 0.047 | 0.015 | **0.023** ⭐ | 3 mat-muls, 2-line collapse |
| **2** | **Contrastive power iter** | 0.020 | **0.044** | **0.011** ⭐ | 0.036 | 3 mat-muls, Mahalanobis |
| **3** | Power iter (n=3) | 0.020 | 0.082 | 0.012 | 0.040 | 3 mat-muls |
| **3** | PCA eigenvalue | 0.028 | 0.077 | 0.014 | 0.035 ⭐ | full SVD |
| **4** | **Retain-deflated power iter** (NEW) | 0.029 | 0.063 | 0.020 | 0.036 | 6 mat-muls |
| 5 | RepCollapse (act+grad) | 0.032 | — | — | — | full SVD + grad |
| 6 | Contrastive PCA α=0.5 | 0.035 | 0.070 | **0.014** | **0.026** | SVD on Σ_f−αΣ_r |
| 7 | DISCO best (8 iters) | 0.037 | — | — | — | gen. eig + grad |
| 8 | **Soft retain-ortho γ=0.5** (NEW) | **0.037** | **0.065** | **0.024** | **0.039** | 6 mat-muls |
| 9 | KL only (no LoRA) | 0.039 | 0.096 | 0.024 | 0.044 | full SVD |
| 10 | Power whitening 1/σ² | 0.040 | 0.119 | **0.015** | **0.046** | 2 lines |
| 11 | Whitening (a−μ)/σ | 0.042 | 0.127 | **0.035** | **0.039** | 2 lines |
| 12 | No KL no LoRA (MLP-only) | 0.057 | 0.095 | **0.023** | **0.051** | full SVD |
| 13 | Diagonal Mahalanobis | 0.069 | **0.061** | **0.023** | **0.031** | top-k coords |
| — | No KL no LoRA (MLP+attn) | 0.071 | 0.112 | BROKEN | 0.054 | full SVD |
| — | Ratio scaling | 0.094–0.106 | — | — | — | SVD + ratio |
| — | GradDiff baseline | 0.087 | — | — | — | — |
| — | LoRA only (no KL) | 0.104 | — | — | — | destabilizes |
| — | Retain-conf token mask A (τ=0.3/0.5) | BROKEN ep3 | — | — | — | static V_r mask |
| — | Retain-cov reject C (frac=0.5/0.7) | BROKEN ep3 | — | — | — | grad-norm·quad |
| — | Retain-cov reject C (frac=0.9) | stable ep5, recall 0.109 | — | — | — | grad-norm·quad |
| — | Retain-cov reject C (frac=0.95) | stable ep10, recall 0.133 | — | — | — | over-filters |
| — | Hybrid A∪C | BROKEN ep4 | — | — | — | A+C combined |
| — | Soft-C reweighting | BROKEN ep3 | — | — | — | 1/(1+δ/med) |
| — | Retain-ortho power iter (NEW) | BROKEN | BROKEN | BROKEN | BROKEN | 6 mat-muls |
| — | **Contrastive PI w/ matched eigvals** (NEW) | BROKEN | BROKEN | BROKEN | BROKEN | 3 mat-muls |
| — | Closed-form retain-proj (NEW) | BROKEN | BROKEN | BROKEN | BROKEN | 1 solve |
| — | Random projection | BROKEN | BROKEN | — | — | random basis |
| — | Steering removal | BROKEN | — | — | — | 1 direction |
| — | Variance clipping | BROKEN | — | — | — | per-dim clamp |
| — | DISCO (act-only) | BROKEN | — | — | — | gen. eig |
| — | Retain PCA | BROKEN | — | — | — | wrong subspace |

## Key takeaways

1. **Contrastive power iter is the new #1 overall.** It matches power iter on Llama Bio (0.020), wins Llama Cyber (0.011), and crushes Qwen Bio (**0.044 vs 0.082, ~2× better**). Soft retain constraint wins: iterate on $(\Sigma_f - \alpha \Sigma_r)$, 3 mat-muls, no SVD.
2. **Qwen Bio was the bottleneck** — every previous method struggled there (0.07–0.13). Both new retain-aware power-iter variants drop it to 0.044–0.065.
3. **The eigenvalue source controls stability; the direction source controls quality.** All three failures (retain-ortho γ=1, contrastive PI with matched eigvals, closed-form retain-proj) share one trait: they change *where the eigenvalues are read from*. This collapses the Mahalanobis dynamic range and blows up wikitext KL at epoch 2. Every method that *keeps eigvals from $\Sigma_f$* — contrastive PI, retain-deflated, soft retain-ortho γ=0.5 — is stable. Direction finding can be aggressively retain-aware; eigenvalue readout cannot.
4. **Soft retain-ortho γ=0.5 is stable but mediocre** (rank 8) — the γ=1 version was the goal, and softening recovers stability but loses the edge. Contrastive PI remains the best retain-aware formulation.
5. **Contrastive PCA (full SVD) still wins on Qwen Cyber** at 0.026, beating contrastive power iter. For Qwen Cyber specifically the full SVD still has an edge.
6. **Cyber is consistently easier than Bio** — Llama Cyber results now cluster at 0.011–0.035.

## Still TODO

- RepCollapse act+grad (Row 5): crashed on Llama Cyber, Qwen Bio, Qwen Cyber — trainer-side bug, needs debugging.
- DISCO best 8 iters (Row 7): requires the gradient-collapse variant, not yet rerun.
