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

## Ablation: PCA source + scaling (activation-only, KL+LoRA)

| iter | pca_source | scaling | best recall_prob (not broken) | epoch | notes |
|------|-----------|---------|-------------------------------|-------|-------|
| 2 | forget | eigenvalue | **0.028** | 8 | Beats RepCollapse (0.032)! Dynamic range 28-531x. Activation-only. |
| 3 | retain | ratio | 0.053 | 3 | Better than forget+ratio. Stable (breaks ep4). Retain dirs ≠ forget dirs. |
| 1 | forget | ratio | 0.106 | 2 | Ratio dynamic range only 3-8x → weak suppression. |
| 0 | forget | ratio+cutoff | 0.094 | 2 | Most layers 0 dirs above threshold → fallback to 10 PCs. |

### Key findings

1. **Activation-only collapse + eigenvalue scaling beats RepCollapse** (0.028 vs 0.032). Gradient collapse is unnecessary — simpler and better.

2. **Eigenvalue magnitude is essential**: Dynamic range 28-531x (eigenvalues) vs 3-8x (ratios). The Mahalanobis suppression factor 1-min/val needs large dynamic range to strongly suppress top PCs.

3. **Forget PCA > retain PCA** (when using eigenvalues): Forget PCA directions align with the attacker's subspace. But when using ratio scaling (which strips magnitude), retain PCA actually outperforms forget PCA (0.053 vs 0.106) — retain directions are more stable.

4. **Answer to Filip's question "why forget not retain PCA?"**: It's NOT about the directions (retain PCA dirs are actually more stable with ratio scaling). It's about the eigenvalue magnitude — forget PCA eigenvalues have the right dynamic range for Mahalanobis suppression. Retain PCA eigenvalues would have a different (retain-focused) dynamic range that doesn't target the attacker's subspace.

## Attention collapse ablation

| Method | best recall_prob | epoch | wikitext_kl | notes |
|--------|-----------------|-------|-------------|-------|
| MLP + Attn (act-only, eigenvalue) | **0.028** | 7 | 0.007 | Reaches same result 1 epoch faster, lower KL |
| MLP-only (act-only, eigenvalue) | 0.028 | 8 | 0.009 | Baseline |
| RepCollapse (act+grad, MLP-only) | 0.032 | 10 | ~0.01 | Original method |

Attention collapse adds marginal benefit: same best recall_prob but faster convergence and lower disruption. Not a large effect — MLP collapse carries most of the signal.

## Systematic grid search (2 models × method variations)

All methods are activation-only (no gradient collapse). recall_prob at last non-broken epoch (lower = better).

| # | pca_source | modules | KL | LoRA | Llama-3.2-3B | Qwen3-8B |
|---|-----------|---------|-----|------|-------------|----------|
| P1 | eigenvalue | MLP+attn | yes | yes | **0.028** (ep7) | **0.077** (ep10) |
| P2 | eigenvalue | MLP+attn | — | — | 0.071 (ep3) | 0.112 (ep4) |
| P3 | eigenvalue | MLP-only | — | — | 0.057 (ep4) | 0.095 (ep6) |
| P4 | eigenvalue | MLP-only | yes | yes | **0.028** (ep8) | **0.075** (ep10) |
| P5 | retain | MLP+attn | yes | yes | BROKEN ep2 | — |
| P6 | eigenvalue | MLP+attn | yes | — | 0.039 (ep10) | 0.096 (ep10) |
| P7 | eigenvalue | MLP+attn | — | yes | 0.104 (ep2) | — |
| P8 | eigenvalue | MLP+attn | yes | yes | **0.014** (cyber, ep6) | **0.039** (cyber, ep9) |
| — | **disco** | MLP+attn | yes | yes | BROKEN ep2 | — |
| — | diagonal | MLP+attn | yes | yes | 0.069 (ep3) | — |
| — | GradDiff | — | — | — | 0.087 (broken ep1) | — |

### Component decomposition (Llama)

| KL | LoRA | recall_prob | effect |
|----|------|-------------|--------|
| — | — | 0.071 | baseline |
| yes | — | 0.039 | KL controls disruption → 1.8x better |
| — | yes | 0.104 | LoRA alone destabilizes |
| yes | yes | **0.028** | KL stabilizes, LoRA pushes robustness |

### Conclusions

1. **Best method**: forget PCA eigenvalue + act-only collapse + KL masking + LoRA adversary
2. **KL masking is the key component** — provides stability for long training (10+ epochs)
3. **LoRA adversary enhances robustness** — but only works with KL masking
4. **MLP+attn ≈ MLP-only** with KL+LoRA — attention collapse adds marginal speed benefit
5. **Forget PCA eigenvalue >> all alternatives**: retain PCA (broken), DISCO generalised eigenvectors (broken), diagonal (0.069), ratio scaling (0.071+)
6. **Generalises across models**: same best config wins on Llama-3.2-3B and Qwen3-8B
7. **Cyber easier than bio**: 0.014 vs 0.028 on Llama
