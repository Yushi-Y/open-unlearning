# Experiment Results: Selective Unlearning via Activation Collapse

## Baseline
- RepCollapse (act+grad, MLP-only, PCA eigenvalue, KL+LoRA): **0.032** Llama Bio

## 1. Systematic Grid Search (2 models × method variations)

All methods activation-only. recall_prob at last non-broken epoch (lower = better).

| pca_source | modules | KL | LoRA | Llama Bio | Qwen Bio | Llama Cyber | Qwen Cyber |
|-----------|---------|-----|------|-----------|----------|-------------|------------|
| eigenvalue | MLP+attn | yes | yes | **0.028** (ep7) | **0.077** (ep10) | **0.014** (ep6) | **0.039** (ep9) |
| eigenvalue | MLP-only | yes | yes | **0.028** (ep8) | **0.075** (ep10) | 0.019 (ep5) | **0.035** (ep10) |
| eigenvalue | MLP+attn | yes | — | 0.039 (ep10) | 0.096 (ep10) | 0.024 (ep4) | 0.044 (ep10) |
| eigenvalue | MLP-only | — | — | 0.057 (ep4) | 0.095 (ep6) | — | — |
| eigenvalue | MLP+attn | — | — | 0.071 (ep3) | 0.112 (ep4) | BROKEN ep2 | 0.054 (ep4) |
| eigenvalue | MLP+attn | — | yes | 0.104 (ep2) | — | — | — |
| retain PCA | MLP+attn | yes | yes | BROKEN ep2 | — | — | — |
| GradDiff | — | — | — | 0.087 (broken ep1) | — | — | — |

### Component decomposition (Llama Bio)

| KL | LoRA | recall_prob | effect |
|----|------|-------------|--------|
| — | — | 0.071 | baseline |
| yes | — | 0.039 | KL controls disruption → 1.8x better |
| — | yes | 0.104 | LoRA alone destabilizes |
| yes | yes | **0.028** | KL stabilizes, LoRA pushes robustness |

## 2. Non-PCA Direction Alternatives (with Mahalanobis projection)

All with KL+LoRA, MLP+attn, activation-only.

| Method | Llama Bio | Qwen Bio | Dynamic range | Notes |
|--------|-----------|----------|---------------|-------|
| PCA eigenvalue | **0.028** (ep8) | 0.077 (ep10) | 28–531x | baseline |
| Contrastive PCA α=0.5 | 0.035 (ep5) | **0.070** (ep9) | 16–413x | PCA on (Σ_f−αΣ_r), beats PCA on Qwen |
| Diagonal Mahalanobis | 0.069 (ep3) | — | 5–100x | coordinate axes, weak |
| DISCO gen. eigenvectors | BROKEN ep2 | — | 15–324x | whitening distorts directions |
| Retain PCA | BROKEN ep2 | — | — | wrong subspace |

## 3. Simple Methods (no PCA, no Mahalanobis projection)

All with KL+LoRA, MLP+attn, activation-only.

| Method | Llama Bio | Qwen Bio | Code complexity | Notes |
|--------|-----------|----------|-----------------|-------|
| Whitening (a−μ)/σ | 0.042 (ep7) | 0.127 (ep10) | 2 lines | stable but too diffuse |
| Power whitening 1/σ² | 0.040 (ep6) | 0.119 (ep10) | 2 lines | marginal improvement |
| Steering vector removal | BROKEN ep2 | — | 1 line | too blunt |
| Variance clipping k=2 | BROKEN ep2 | — | 1 line | too destructive |

## 4. Diagnostic: Why Whitening is Weak

**Attack subspace analysis** (Llama Layer 14, gate_proj):

| Collapse method | Energy in attacker top-50 PCs | Update concentration |
|----------------|-------------------------------|---------------------|
| Whitening 1/σ | 1.6% | all D=3072 dims (diffuse) |
| Power 1/σ² | 1.1% | all D dims (diffuse) |
| Ratio σ_r/σ_f | 3.8% | all D dims (diffuse) |
| PCA Mahalanobis | ~2% | ~400-dim subspace (concentrated) |

**Key insight**: Whitening avoids the attacker subspace equally well (1.6% vs 2%). The gap comes from **concentration** — PCA focuses the weight update into a low-dimensional subspace where each direction gets a strong update. Whitening spreads across all dimensions, making each update too small to resist fine-tuning.

**Per-dimension structure**:
- Median σ_f/σ_r ratio: 0.893 (retain slightly dominates)
- Only 1.9% of dims are selective (ratio > 1.5)
- 75.4% are shared (0.8–1.5)
- 22.7% are retain-dominated (< 0.8)

## Key Conclusions

1. **Activation-only collapse beats RepCollapse** (0.028 vs 0.032) — gradient collapse is unnecessary
2. **KL masking is essential** — provides training stability (10+ epochs)
3. **LoRA adversary adds robustness** — but only with KL masking
4. **PCA eigenvalue >> all alternatives** tested so far
5. **PCA's advantage = direction selection + update concentration** — whitening matches direction selection but lacks concentration
6. **Contrastive PCA beats standard PCA on Qwen** (0.070 vs 0.077)
7. **Cyber easier than Bio**: Llama 0.014 vs 0.028, Qwen 0.035 vs 0.075
8. **Qwen more stable than Llama**: rarely breaks, converges slower

## Next: Concentration without PCA

The diagnostic shows we need a **cheap basis** that concentrates the update without eigenvector computation. Candidates:
- Random projection + whitening (random basis, k=400)
- Top-k dimension projection (sparse basis)
- Power iteration (approximate PCA, 3-5 iterations)
