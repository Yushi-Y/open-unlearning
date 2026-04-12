# Full Leaderboard (all methods, all conditions)

All activation-only with KL+LoRA unless noted. recall_prob at last non-broken epoch (lower = better).

| Rank | Method | Llama Bio | Qwen Bio | Llama Cyber | Qwen Cyber | Cost |
|------|--------|-----------|----------|-------------|------------|------|
| **1** | **Power iter (n=3)** | **0.020** | 0.082 | **0.012** | 0.040 | 3 mat-muls |
| **2** | PCA eigenvalue | 0.028 | 0.077 | 0.014 | **0.035** | full SVD |
| 3 | RepCollapse (act+grad) | 0.032 | — | — | — | full SVD + grad |
| 4 | Contrastive PCA α=0.5 | 0.035 | **0.070** | — | — | SVD on Σ_f−αΣ_r |
| 5 | KL only (no LoRA) | 0.039 | 0.096 | 0.024 | 0.044 | full SVD |
| 6 | Power whitening 1/σ² | 0.040 | 0.119 | — | — | 2 lines |
| 7 | Whitening (a−μ)/σ | 0.042 | 0.127 | — | — | 2 lines |
| 8 | DISCO best (8 iters) | 0.037 | — | — | — | gen. eig + grad |
| 9 | No KL no LoRA (MLP-only) | 0.057 | 0.095 | — | — | full SVD |
| 10 | Diagonal Mahalanobis | 0.069 | — | — | — | top-k coords |
| 11 | No KL no LoRA (MLP+attn) | 0.071 | 0.112 | BROKEN | 0.054 | full SVD |
| 12 | Ratio scaling | 0.094–0.106 | — | — | — | SVD + ratio |
| — | GradDiff baseline | 0.087 | — | — | — | — |
| — | LoRA only (no KL) | 0.104 | — | — | — | destabilizes |
| — | Random projection | BROKEN | BROKEN | — | — | random basis |
| — | Steering removal | BROKEN | — | — | — | 1 direction |
| — | Variance clipping | BROKEN | — | — | — | per-dim clamp |
| — | DISCO (act-only) | BROKEN | — | — | — | gen. eig |
| — | Retain PCA | BROKEN | — | — | — | wrong subspace |
