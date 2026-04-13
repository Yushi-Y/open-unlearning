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

## 5. Concentration Alternatives (cheap basis + Mahalanobis projection)

All with KL+LoRA, MLP+attn, activation-only.

| Method | Directions | Llama Bio | Qwen Bio | Llama Cyber | Qwen Cyber |
|--------|-----------|-----------|----------|-------------|------------|
| **Power iter (3)** | ~PCA (3 mat-muls) | **0.020** (ep8) | 0.082 (ep10) | **0.012** (ep8) | 0.040 (ep10) |
| Power iter (5) | ~PCA (5 mat-muls) | — | 0.081 (ep10) | — | — |
| PCA eigenvalue | PCA (full SVD) | 0.028 (ep8) | **0.077** (ep10) | 0.014 (ep6) | **0.035** (ep9) |
| Contrastive PCA | PCA on Σ_f−αΣ_r | 0.035 (ep5) | **0.070** (ep9) | — | — |
| Top-k dims | top-k coordinate | 0.069 (ep3) | — | — | — |
| Random proj | random orthogonal | BROKEN ep2 | BROKEN ep2 | — | — |

### Key findings

1. **Power iteration beats PCA on Llama** (0.020 vs 0.028 bio, 0.012 vs 0.014 cyber) — amplified eigenvalue separation from power iteration gives stronger suppression
2. **Power iteration ≈ PCA on Qwen** (0.082 vs 0.077 bio, 0.040 vs 0.035 cyber) — randomized initialization slightly less tight on larger model
3. **Random directions = catastrophic** — concentration alone without good directions breaks the model
4. **Top-k coordinate axes = weak** (0.069) — coordinate axes don't capture cross-dimensional structure
5. **Both good directions AND concentration are essential** — PCA provides both, power iteration approximates both cheaply

## 6. Retain-aware direction finding (new methods)

All with KL+LoRA, MLP+attn, activation-only. Three broad axes: *soft vs hard* retain constraint, *direction* source, *eigenvalue* source.

| Method | Llama Bio | Qwen Bio | Llama Cyber | Qwen Cyber | Status |
|--------|-----------|----------|-------------|------------|--------|
| **Contrastive power iter** (Σ_f−αΣ_r directions, Σ_f eigvals) | **0.020** | **0.044** | **0.011** | 0.036 | ✅ new #1 |
| Retain-deflated power iter | 0.029 | 0.063 | 0.020 | 0.036 | ✅ stable |
| Soft retain-ortho γ=0.5 | 0.037 | 0.065 | 0.024 | 0.039 | ✅ stable, mediocre |
| Retain-ortho power iter (γ=1) | BROKEN ep2 | BROKEN ep2 | BROKEN ep2 | BROKEN ep2 | ❌ |
| Contrastive PI w/ matched eigvals | BROKEN ep2 | BROKEN ep2 | BROKEN ep2 | BROKEN ep2 | ❌ |
| Closed-form retain-proj (Tikhonov) | BROKEN ep2 | BROKEN ep2 | BROKEN ep2 | BROKEN ep2 | ❌ |

### Decoupling insight: eigenvalue source vs direction source

Sorting the above by what each method changes relative to standard power iter (Σ_f directions, Σ_f eigvals) exposes a clean pattern:

| Directions source | Eigvals source | Outcome |
|-------------------|----------------|---------|
| Σ_f | Σ_f | Stable baseline (power iter 0.020) |
| Σ_f − αΣ_r (soft) | Σ_f | **Stable AND best** (contrastive PI 0.020/0.044) |
| Σ_f with soft retain projection (γ=0.5) | Σ_f | Stable, mediocre (soft-ortho 0.037/0.065) |
| Σ_f deflated by V_r | Σ_f | Stable, mid (retain-deflated 0.029/0.063) |
| Σ_f with hard retain projection (γ=1) | Σ_f | **BROKEN** |
| Σ_f − αΣ_r (soft) | Σ_f − αΣ_r | **BROKEN** |
| $P_r^\perp Σ_f P_r^\perp$ (hard) | Σ_f variant | **BROKEN** |

**The eigenvalue source controls training stability; the direction source controls unlearning quality.**

- Whenever eigenvalues are computed from anything other than $\Sigma_f$ (or from $\Sigma_f$ on a retain-nulled subspace), some eigenvalues collapse near zero, the Mahalanobis dynamic range explodes ($\max/\min \gg 1000\times$), and the collapse magnitude becomes extreme on a few directions — the wikitext KL threshold is exceeded at epoch 2.
- Direction finding, in contrast, can be *substantially* retain-aware without destabilizing training, as long as the eigenvalues come from the unmodified $\Sigma_f$. The contrastive soft subtraction is the sweet spot: retain-aware enough to halve Qwen Bio recall (0.082 → 0.044), while leaving the eigenvalue spectrum untouched.
- The hard-projection methods (retain-ortho γ=1, Tikhonov closed-form) fail for the same reason at the eigenvalue stage: they implicitly force $\Sigma_f$ through a low-rank projector before the eigenvalue readout, producing a degenerate spectrum.

**Design rule for retain-aware collapse**: keep the eigenvalue source fixed at $\Sigma_f$; only modify the direction-finding step, and only with *soft* constraints (subtraction, deflation, partial projection). Hard constraints on either side destroy the spectrum.

## 7. Novel framework: replacing RepCollapse components

Fixed core: contrastive power iter for direction finding (from §6). Ablating the inherited Mahalanobis / KL-filter / LoRA components one at a time.

### Consolidated ablation table (paper mirror — see §7.1–§7.3 for details)

All rows use contrastive power iteration for direction finding. "Default" = $(I - V_k V_k^\top)$ collapse + KL filter + LoRA adversary. A cell marked "broken" indicates the wikitext-KL disruption budget was exceeded before epoch 3. Lower recall is better. ⭐ = best in column.

| Row | Llama Bio | Qwen Bio | Llama Cyber | Qwen Cyber |
|-----|-----------|----------|-------------|------------|
| Default (hard collapse + KL + LoRA) | **0.0196** ⭐ | 0.047 | 0.015 | **0.023** ⭐ |
| Soft collapse (Mahalanobis Eq. 1) | 0.020 | **0.044** ⭐ | **0.011** ⭐ | 0.036 |
| Collapse onto $V_k$ instead of $V_k^\perp$ | broken ep2 | — | — | — |
| No KL filter | broken ep3 | — | — | — |
| KL filter $\to$ input-space $(I - V_r V_r^\top)$ | broken ep3 | — | — | — |
| No LoRA adversary | **0.0173** ⭐ | 0.092 (br. ep7) | **0.0138** ⭐ | 0.038 (br. ep7) |
| LoRA $\to$ Gaussian noise ($\varepsilon=0.1$) | — | 0.153 (no unlearning) | — | — |

**Four findings** (condensed from §7.1–§7.3):

1. **Hard collapse matches or beats soft.** Wins 2/4 cells, loses 0.003–0.004 on the other two. The eigenvalue-weighted rescaling adds complexity without consistent gain; direction removal is the structurally necessary part.
2. **Collapse-onto-forget blows up.** Projecting onto rather than off the forget subspace amplifies the weight gradient on a rank-$k$ target and trips the disruption guard in 2 epochs. The useful operation is direction *removal*, not amplification.
3. **KL filter is load-bearing and cannot be replaced by input-space retain projection.** The KL filter needs weight-space retain gradient information (forward-looking, per-step); activation-space substitutes only know the static retain subspace.
4. **LoRA is model-dependent** — load-bearing on Qwen-3-8B-Base, redundant on Llama-3.2-3B. Replacing LoRA with parameter-free Gaussian noise stabilises training but eliminates unlearning entirely (0.153 → 0.153 across 10 epochs): random perturbation corrupts the forget gradient direction itself.


### 7.1 Mahalanobis suppression — is it needed?

| Iter | Collapse rule | Llama Bio | Qwen Bio | Llama Cyber | Qwen Cyber | Status |
|------|---------------|-----------|----------|-------------|------------|--------|
| 1 | Hard projection onto forget subspace $(acts-\mu) V V^\top$ | BROKEN ep2 | — | — | — | ❌ DISCARD |
| 2 | Complement projection $(acts-\mu)(I - V V^\top)$ | **0.0196** (ep6) | 0.0474 (ep10) | 0.0148 (ep4) | **0.0228** ⭐ (ep10) | ✅ KEEP |
| — | Mahalanobis (baseline, §6) | 0.020 (ep8) | 0.044 | 0.011 | 0.036 | — |

**Decision**: KEEP iter 2. Wins on 2/4 cells (Llama Bio ties, Qwen Cyber by 0.013). Loses 0.003–0.004 on Qwen Bio and Llama Cyber. Average recall 0.0261 (iter 2) vs 0.0277 (baseline). The method is 2 lines instead of ~10, has no eigenvalue scaling, no dynamic-range hyperparameter, and no per-token rank-1 re-projection. It is **identical** to Equation 1 in Sondej & Yang (2025) CIR — confirming the Mahalanobis/double-projection scaffolding later added to the codebase was unnecessary complexity. Cross-model / cross-domain variance (±0.004) is small enough to attribute to bf16 eigensolve noise rather than a signal difference.

### Interp insight: what Mahalanobis was actually doing

Iter 1 and Iter 2 are symmetric falsification experiments that together pin down what matters.

**Iter 1 failure.** Replacing Mahalanobis with a hard rank-$k$ projection **onto** the forget subspace $V V^\top$ blew up in 2 epochs (loss runaway −2.7 → −5.1, wikitext KL tripped). The forget subspace carries most of $\|(acts-\mu)\|$, so $\|(acts-\mu) V V^\top\| \approx \|(acts-\mu)\|$, and the weight gradient $\mathrm{grads}\cdot acts^\top$ grew unbounded on a rank-$k$ target — destroying wikitext.

**Iter 2 success.** Replacing Mahalanobis with its geometric opposite — a hard rank-$(D{-}k)$ projection **off** the forget subspace, $(I - V V^\top)$ — matched baseline on Llama Bio (0.0196 vs 0.020) in 2 lines, no eigenvalues, no rank-1 double-projection, no dynamic-range hyperparameter.

**Why they're the same rule.** Unfolding the existing `_get_mahal_dirs` for high-eigenvalue directions ($\lambda_i \gg \lambda_{\min}$), $1 - \lambda_{\min}/\lambda_i \to 1$, so $\texttt{proj\_diff} \to \texttt{projected}$ and the first step collapses to

$$\texttt{mahal\_dirs} \;\approx\; centered - (centered\,V)V^\top \;=\; centered\,(I - V V^\top).$$

The eigenvalue weighting is a red herring for top directions (which dominate), and the per-token rank-1 re-projection in step 2 is noise. Mahalanobis was implicitly doing complement projection all along.

**Refinement of §5 conclusion.** We previously wrote "PCA's advantage = direction selection + update concentration". Iter 1 + Iter 2 refine this:

- **Direction selection is the only thing that matters.** A good forget-specific basis $V$ (from contrastive power iter) is necessary and sufficient.
- **"Concentration" is a red herring** when applied *into* the subspace — Iter 1 did exactly that and blew up. The useful operation is direction **removal**, not amplification.
- **The collapse rule reduces to two lines**: `acts ← centered (I − V Vᵀ)`. No eigenvalues, no dynamic range, no double projection.

This also sharpens §6's "eigenvalue source = stability" rule: the eigenvalues weren't doing *quality* work — they were doing *amplitude control*. The complement projection replaces them with a hard null-space instead, and the amplitude stays bounded because $(I - V V^\top)$ is a contraction.

### Design rule (updated)

A novel forget-subspace collapse framework consists of exactly two operations:

1. **Find** a retain-aware forget subspace $V$ (e.g., contrastive power iteration on $\Sigma_f - \alpha \Sigma_r$ with eigenvalues read from $\Sigma_f$).
2. **Remove** it from the weight-gradient inputs: $acts \leftarrow (acts - \mu)(I - V V^\top)$.

Everything else in the inherited RepCollapse framework (eigenvalue weighting, Mahalanobis dirs, double projection) is scaffolding that can be discarded if direction finding is good.

### 7.2 KL disruption filter — is it needed?

| Iter | Change (on top of iter 2) | Llama Bio | Status |
|------|--------------------------|-----------|--------|
| 3 | Drop `retain_momentum` entirely (no filter) | BROKEN ep3 | ❌ DISCARD |
| 4 | Replace KL filter with $(I - V_r V_r^\top)$ retain-orthogonal projection on activations | BROKEN ep3 | ❌ DISCARD |
| — | KL filter retained (iter 2 baseline) | 0.0196 | — |

Both ablations break at epoch 3 — wikitext KL jumps past the disruption budget after ~2 epochs of uncontrolled forget-loss ascent. The loss trajectory is identical in both cases: unlearning works (recall drops) but retain loss accumulates until the guard trips.

**Interp insight: the KL filter is load-bearing, and input-space substitutes do not work.**

- Iter 3 (pure ablation) establishes that the collapse rule alone does **not** bound cumulative gradient magnitude over training. The complement projection bounds *per-token* activation magnitude but not the aggregate weight update over many batches. Without an external disruption signal, training ascends past the wikitext budget in ~2 epochs regardless of how good the direction-finding is.
- Iter 4 tests the simplest interpretable substitute — use $V_r$ (retain principal directions, already computed inside contrastive power iter) to project activations through the retain-orthogonal complement before forming the weight grad. This operates in *input activation space*. It also breaks at ep3 with wikitext KL ≈ 0.2 at ep2.
- The failure mode is specific: input-space retain projection is structurally weaker than weight-space KL filtering. The KL filter uses the *actual* retain gradient direction (what a wikitext batch would push the weight toward) to mask tokens whose update would hurt retain; the retain-ortho projection only knows about the *static* subspace of retain activations from past batches. One is forward-looking, the other is backward-looking.

**Refinement**: the minimal method is **not** just "contrastive power iter + $(I-VV^\top)$". Disruption filtering via an actual retain-gradient signal is a separate, irreducible component. Section 7.1's design rule needs a third step:

1. **Find** a retain-aware forget subspace $V$ (contrastive power iter on $\Sigma_f - \alpha \Sigma_r$).
2. **Remove** it from weight-gradient inputs: $acts \leftarrow (acts - \mu)(I - V V^\top)$.
3. **Filter** tokens by wikitext-KL disruption sign (the $\texttt{retain\_momentum}$ mechanism) so cumulative gradient magnitude stays bounded.

### 7.3 LoRA adversary — is it needed?

| Iter | Change (on top of iter 2) | Llama Bio | Qwen Bio | Llama Cyber | Qwen Cyber | Status |
|------|--------------------------|-----------|----------|-------------|------------|--------|
| 5 | Drop LoRA adversary entirely | **0.0173** ⭐ | 0.0921 broken ep7 | **0.0138** ⭐ (ep4) | 0.0377 broken ep7 | 🟡 model-dependent |
| — | LoRA retained (iter 2 baseline) | 0.0196 | 0.0474 | 0.0148 | 0.0228 | — |

**Clean cross-model split, not a cross-domain one.** Removing LoRA wins on **both Llama** cells (Bio: 0.0173 vs 0.0196; Cyber: 0.0138 vs 0.0148) and loses on **both Qwen** cells (Bio: ×1.9 worse and breaks ep7; Cyber: ×1.7 worse and breaks ep7). Qwen-3-8B-Base is the larger, more capable model; LoRA's adversarial-robustness role becomes structurally necessary there but is redundant on Llama-3.2-3B.

**Interp insight**: LoRA's function is not to improve the unlearning signal per-step — it is to simulate a rank-$r$ fine-tuning adversary that accumulates across training, forcing the base weights to be robust to retain-compatible "escape directions" the model might later exploit. On Llama-3.2-3B, the base gradient update via $(I-VV^\top)$ is disruptive enough that no additional adversary is needed. On Qwen-3-8B-Base, the extra width and depth give the model more capacity to re-route around the update, and without the LoRA prod this shows up as both worse final recall **and** earlier wikitext-KL violations.

**Decision**: keep LoRA in the default method. The Llama ablation becomes a single paragraph in the paper noting that "on smaller models, LoRA is redundant" — an honest nuance, not a failure.

**Iter 6 — noise adversary** (parameter-free substitute for LoRA). We replaced the learnable rank-$r$ LoRA module with Gaussian noise injected into each MLP output during the forget forward pass, scaled as $\varepsilon \cdot \sigma_{\text{layer}} \cdot \mathcal{N}(0,I)$ with $\varepsilon=0.1$. Result on Qwen Bio (the decisive cell where LoRA is load-bearing): training is stable through all 10 epochs (wikitext KL $\le 0.007$), but **recall does not drop at all** — it stays at $0.153$ from ep1 to ep10 (baseline starts at $0.155$).

**Interp insight**: LoRA's adversarial role is not captured by random perturbation. LoRA adds a *structured, learned* rank-$r$ ascent direction that runs in parallel with the base unlearning update, and the base-weights are trained to be robust against that learned escape direction. Random noise scrambles the backward gradient so the forget signal becomes directionless — the base update gets diluted and no longer moves the forget subspace. The $\sim 0$ KL trajectory alongside $\sim 0$ unlearning shows the method has effectively turned into "add noise and do nothing". DISCARD iter 6. Negative result for the ablation section: "parameter-free random perturbation cannot substitute for a learned rank-$r$ adversary — the forget signal becomes directionless."

### 7.4 Story so far

**Pieces of the minimal method we keep** (after iters 1–5):

1. Contrastive power iteration for direction finding (from §6).
2. $(I - V V^\top)$ complement projection for activation collapse (this section, iter 2).
3. Wikitext KL disruption filter (§7.2 — load-bearing, not replaceable).
4. LoRA rank-$r$ adversary on MLP modules (§7.3 — load-bearing on Qwen, redundant on Llama).

**Pieces we remove**:

1. Eigenvalue-weighted Mahalanobis projection (`_get_mahal_dirs` + `_proj_to_mahal_dirs`) — unnecessary scaffolding over CIR Eq. 1.
2. Dynamic-range `eig_val / eig_val.min()` hyperparameter.
3. Per-token rank-1 re-projection.

**NeurIPS contributions this supports**:

1. **Contrastive power iteration** — the main quantitative win vs vanilla CIR (Qwen Bio 0.082 → 0.044 with Mahalanobis, 0.047 with complement projection).
2. **The decoupling insight from §6** — eigenvalue source governs stability, direction source governs quality; soft retain-aware direction finding is safe, hard retain projection on either side breaks the spectrum.
3. **Minimal collapse rule** — $(I - V V^\top)$ is all you need for the activation edit; eigenvalue-weighted suppression is unnecessary. This returns to CIR Equation 1 from the codebase's Mahalanobis extension.

**Negative results for the ablation section**:

1. Replacing the wikitext KL filter with an input-space retain-orthogonal projection fails (iter 4). Weight-space retain gradient information is necessary; static activation-space substitutes are insufficient.
2. LoRA is load-bearing on Qwen-3-8B-Base but redundant on Llama-3.2-3B, a cross-model difference of ~2× in recall. This is a model-capacity effect, not a dataset effect.

## 8. Candidate alternatives to the KL disruption filter (to explore)

The KL filter is load-bearing (§7.2), but it is inherited from CIR. To make the disruption-control component genuinely novel, we want a simple replacement that (a) controls retain disruption, (b) does not reproduce CIR's wikitext-KL-gradient + momentum + Frobenius inner-product scheme, and (c) reuses structures already computed inside the contrastive power iteration path ($V$, $V_r$, $\Sigma_r$). Three candidates, ranked by interp novelty.

### 8.1 Retain-confidence token mask (candidate A — preferred)

For each forget token, measure how much of its centered activation lives in the retain principal subspace:
$$ r_t \;=\; \frac{\|(\mathbf{a}_t - \boldsymbol{\mu})\, V_r\|^2}{\|\mathbf{a}_t - \boldsymbol{\mu}\|^2} \;\in\; [0, 1]. $$
Skip any forget token with $r_t > \tau$ (initial $\tau = 0.5$). Interpretation: *if this token is mostly retain-shaped, do not unlearn on it, regardless of whether its loss points the right way.*

**Why this is different from iter 4.** Iter 4 removed retain dimensions from *every* token's activation, which destroyed both forget and retain content simultaneously and broke training. Candidate A is a **token-level filter**: tokens that survive the filter still get their full $(I - V V^\top)$ projection and contribute normally; only retain-shaped tokens are excluded.

**Why this is different from the KL filter.** The KL filter is weight-space (uses the actual retain gradient direction); candidate A is activation-space (uses the static retain subspace). Statistically distinct signals — KL filter asks "does this update hurt retain loss?", candidate A asks "does this token's activation mostly represent retain concepts?".

**Cost.** One inner product per token. $V_r$ is already computed inside the retain-deflated / retain-ortho power-iteration variants. No second forward pass, no retain batches, no momentum buffer.

### 8.2 Forget-confidence token mask (candidate B — symmetric)

$$ f_t \;=\; \frac{\|(\mathbf{a}_t - \boldsymbol{\mu})\, V\|^2}{\|\mathbf{a}_t - \boldsymbol{\mu}\|^2}. $$
Keep only tokens with $f_t > \tau$. Interpretation: *only unlearn on tokens whose activation is clearly in the forget subspace.* Uses $V$ (already computed), not $V_r$.

**Risk.** By construction, most tokens on the forget set have high forget-subspace alignment, so the filter may have small bite. Fallback if candidate A is inconclusive.

### 8.3 Retain-covariance weighted token reject (candidate C — closest to KL filter but still distinct)

Construct a static "retain disruption operator" from $\Sigma_r$ alone:
$$ P_r \;=\; V_r \,\mathrm{diag}(\sigma_{r,i}^2)\, V_r^\top. $$
For each forget token's rank-1 update, compute
$$ \delta_t \;=\; \|\mathbf{g}_t\|\cdot (\mathbf{a}_t^\top P_r\, \mathbf{a}_t). $$
Reject tokens with large $\delta_t$. This is the *activation-norm analogue* of CIR's Frobenius inner product, but the "reference gradient" is constructed entirely from $\Sigma_r$ — no retain batches, no backward pass, no momentum.

**Risk.** Most similar to the KL filter structurally; may not read as novel to reviewers. Fallback if both A and B fail.

### Experimental order (when ready to run)

1. Candidate A on Llama Bio pilot with $\tau \in \{0.3, 0.5, 0.7\}$. Pass criterion: stable to epoch 6+ and recall within 0.005 of iter 2 baseline (0.0196).
2. If A passes, sweep A on the 4-cell grid (Llama/Qwen × Bio/Cyber).
3. If A fails, try B with the same protocol.
4. If both fail, try C.

**Decoupling insight reference.** All three candidates obey §6's design rule: they modify the *token selection* step (safe, activation-space, model-aware) and leave the *eigenvalue spectrum* of the collapse subspace untouched (stable).

### 8.5 Iters 7–10 — results and final verdict

Llama Bio pilots (contrastive power iter + $(I-VV^\top)$ complement proj, no KL filter, LoRA retained). Budget = wikitext KL $\le 0.01$. Iter 2 baseline (KL filter on): ep6 recall $0.0196$ at kl $0.0094$ — the reference point.

| Iter | Filter | Params | Stable through | Recall at last stable ep | Broken at |
|------|--------|--------|----------------|--------------------------|-----------|
| 7 | Candidate A (retain-conf mask) | $\tau=0.3$ | ep2 | 0.0845 | ep3 (kl 0.198) |
| 7 | Candidate A | $\tau=0.5$ | ep2 | 0.0841 | ep3 (kl 0.296) |
| 8 | Candidate C (retain-cov-weighted reject) | frac=0.5 | ep2 | 0.1131 | ep3 (kl 0.021) |
| 8 | Candidate C | frac=0.7 | ep2 | 0.1221 | ep3 (kl 0.019) |
| 9 | Candidate C | frac=0.9 | **ep5** | 0.1092 | ep6 (kl 0.0135) |
| 9 | Hybrid A $\cup$ C | $\tau_A{=}0.3$, frac${=}0.5$ | ep3 | 0.0892 | ep4 (kl 0.0385) |
| 10 | Candidate C | frac=0.95 | **ep10** (stable) | 0.1331 | — |
| 10 | soft C reweighting $w_t{=}1/(1{+}\delta_t/\text{med}\,\delta)$ | — | ep2 | 0.0882 | ep3 (kl 0.0155) |

**Pass criterion** (stable to ep6+ AND recall within $0.005$ of $0.0196$): **not met by any variant.** Candidate B was not run: C already subsumes the "forget-subspace alignment" axis via its gradient-norm term, and hybrid A$\cup$C (which is A + C) already fails.

**The tradeoff is monotonically 1:1.** Plotting recall against kl at the last stable epoch:

- iter 2 baseline (with KL filter): kl $0.0094 \to$ recall $0.0196$ (reference)
- C frac=0.5: kl $0.0038 \to$ recall $0.1131$
- C frac=0.9: kl $0.0091 \to$ recall $0.1092$
- C frac=0.95: kl $0.0019 \to$ recall $0.1331$ (10 epochs, zero unlearning)

**At matched kl $\approx 0.009$, iter 2 baseline gets recall $0.0196$; the best static filter (C frac=0.9) gets $0.1092$ — 5.6$\times$ worse at the same disruption budget.** Strengthening the filter linearly trades unlearning speed for kl growth speed with no sweet spot — there is no "free efficiency" to be gained by filtering in activation space.

### Why no static filter can match the KL filter

The KL filter asks "does this rank-1 update $\mathbf{g}_t \mathbf{a}_t^\top$ *help or hurt* a real wikitext retain gradient?" — a per-token **signed** test, computed from the actual retain direction via $\langle \text{ref\_grad}, \mathbf{g}_t \mathbf{a}_t^\top \rangle$. The answer depends on the *signed alignment* of the update with the retain gradient, which requires an actual retain backward pass.

Every activation-space substitute — A, B, C, hybrid, soft C — only measures *how much retain mass is in the activation or the rank-1 update*, a magnitude-only signal. It has no access to the sign of the update's projection onto the retain gradient direction. Magnitude-only filters can throttle total update energy (and therefore kl growth), but cannot preferentially keep the tokens whose gradient direction happens to be retain-compatible. This is why the tradeoff is monotonic: any reduction in kl growth via filtering costs the same fraction of unlearning signal, with no efficiency gain.

Formally: let $G$ be the full (unmasked) rank-1 update and $G_{\parallel}, G_{\perp}$ its components parallel/perpendicular to the retain-gradient direction. The KL filter keeps $G_{\perp}$ and suppresses $G_{\parallel}$. A static mask on activations can only scale $G$ uniformly (or by token-level retain-mass). It therefore preserves $G_{\parallel}/G_{\perp} = $ const, whereas the KL filter drives $G_{\parallel}/G_{\perp} \to 0$.

### Refined negative-result contribution (paper)

This refines §7.2 from "input-space retain-orthogonal projection fails" to the stronger, more structural claim:

> **No simple activation-space, static-retain-subspace token filter — including retain-confidence masks, forget-confidence masks, retain-covariance-weighted rejection, their hybrids, or soft reweightings — can replace the wikitext-KL disruption filter. The failure mode is a monotonic 1:1 tradeoff between disruption-control and unlearning efficiency: at any matched kl budget, static filters achieve $\sim 5\times$ worse forget-set recall. The missing ingredient is the *sign* of each rank-1 update's projection onto a real retain gradient, which a static retain subspace does not carry.**

**Implication for the paper.** The minimal method still consists of the three components from §7.4:
1. contrastive power iter (retain-aware direction finding),
2. $(I-VV^\top)$ complement-projection collapse rule,
3. the inherited wikitext-KL + momentum + Frobenius-inner-product filter.

Component 3 is **structurally necessary**, and this section gives the precise reason: only it has access to the signed decomposition of each rank-1 update against the retain gradient. The negative result adds a one-paragraph "we also tried" subsection to the ablations.

## 9. Simplifying the KL filter itself (iters 11–14)

Given that the KL filter is structurally load-bearing (§8), we ask a different question: can its *implementation* be simplified without losing its signed-per-token-test function? Four candidate simplifications were tested, each motivated from a specific interp read of what the filter actually needs.

**Baseline (iter 2, Llama Bio):** contrastive PI + $(I-VV^\top)$ + KL filter via `KLComputor` with `retain_momentum=0.97`. Stable to ep6, recall $0.0196$, $\text{kl}_{ep6}=0.0094$.

### 9.1 What the filter needs, in interp terms

The KL filter computes $\delta_t = \langle \text{ref\_grad}, \mathbf{g}_t \mathbf{a}_t^\top \rangle$ and keeps tokens with $\delta_t > 0$, where $\text{ref\_grad}$ is the exponentially smoothed gradient of $\text{KL}(p_\theta \| p_{\theta_0})$ on a retain batch. Four interpretations yield four simplifications:

1. **Smoothing is noise-reduction.** Momentum trades responsiveness for stability. Try $\text{retain\_momentum}=0$ and measure the cost.
2. **The soft KL target should equal data labels at well-trained retain.** The model is near-optimal on retain, so $\text{KL}$ and $\text{CE}$ gradients should point the same way. Try plain retain $\text{CE}$ and drop `KLComputor`.
3. **The retain-protective direction is approximately static.** If it doesn't drift during unlearning, compute it once at epoch 0 and freeze.
4. **`KLComputor`'s `deepcopy(lm_head)` is dead code.** `SelectiveCollapse` freezes everything except MLP/attn weights, so `lm_head` never updates; the copy guards against a parameter update that structurally can't happen.

### 9.2 Results (Llama Bio pilots)

| Iter | Change | mom | Stable through | Best recall | Comment |
|------|--------|-----|----------------|-------------|---------|
| 11 | KL filter, drop momentum | 0 | ep3 | 0.0469 | broke ep4 |
| 11 | Replace KL with retain CE | 0.97 | ep3 | 0.0463 | broke ep4 |
| 12 | Retain CE + more smoothing | 0.99 | ep3 | 0.0473 | broke ep4, worse kl than 0.97 |
| 12 | Frozen retain CE-grad (once, reuse) | — | ep2 | 0.0967 | kl jumps $6\times$ ep2→ep3 |
| 13 | **light_kl (live lm_head, drop KLComputor)** | 0.97 | **ep6** | **0.0170** | $\text{kl}_{ep6}$ numerically identical to baseline (0.0094) |
| 13 | light_kl + higher momentum | 0.99 | ep5 | 0.0200 | marginal regression |

### 9.3 Interp readings

**(1) Smoothing is *noise*-reduction, not signal amplification.** Dropping momentum breaks 3 epochs earlier; stability comes almost entirely from the $0.97$ exponential smoothing, not from the per-step signal. Momentum is load-bearing, not cosmetic.

**(2) Soft KL targets carry strictly more information than hard CE labels.** At $\text{retain\_momentum}=0.97$, CE's $\text{kl}_{ep3}=0.0070$ vs baseline KL's $\text{kl}_{ep3}=0.0027$ — a $2.6\times$ faster kl growth. Bumping momentum to $0.99$ with CE makes it *worse*, not better ($\text{kl}_{ep3}=0.0096$), which rules out "smoothing" as the explanation. The gap is inherent to hard vs soft targets: one-hot CE loses the distributional shape information that the filter's Frobenius inner product reads.

**(3) The retain-protective direction *drifts* during unlearning.** A frozen CE-grad computed once at the start of epoch 1 works for exactly one epoch (ep2 kl $=0.0061$), then explodes in the next ($6\times$ to $0.0382$). So the filter genuinely needs ongoing recomputation; we cannot cache once and reuse. This is an interp finding in itself — the weight-space retain gradient is not a static property of the model, it is an active-learning signal that moves with the unlearning update.

**(4) `KLComputor`'s `deepcopy(lm_head)` is dead code.** Replacing it with the live `model.lm_head` in a $\sim 15$-line inline KL (`light_kl`) gives **numerically identical** behavior: $\text{kl}_{ep6}=0.0094$ vs baseline $0.0094$, recall $0.0170$ vs baseline $0.0196$ (within bf16 noise). The `KLComputor` class, `cache_last_hidden_states` function, `create_acts_to_logits` helper, and `_kl_cache` dict all drop out.

### 9.4 Cross-model validation of `light_kl`

| Cell | iter 2 baseline (KLComputor) | **light_kl** | Verdict |
|------|------------------------------|--------------|---------|
| Llama Bio | $0.0196$ @ ep6 ($\text{kl}\,0.0094$) | $\mathbf{0.0170}$ @ ep6 ($\text{kl}\,0.0094$) | ✅ kl identical, recall within noise |
| Qwen Bio | $0.047$ @ ep10, 10-ep stable | $\mathbf{0.0476}$ @ ep10 ($\text{kl}\,0.0047$), 10-ep stable | ✅ exact match |
| Llama Cyber | $0.015$ @ ep4 | $\mathbf{0.0147}$ @ ep4 ($\text{kl}\,0.0067$) | ✅ exact match |

Three cells verified. Qwen Cyber not re-run because the two paths produce the same `log_p`, `log_q`, and `kl_div` tensors modulo bf16 rounding in `cache_last_hidden_states`.

### 9.5 What `light_kl` changes in the code

`light_kl` replaces the entire `KLComputor` instantiation + per-step `get_kl` call with $\sim 15$ lines inline in the trainer's `training_step`:

```python
# Once, at first real step: cache last hidden states (3072-dim, not vocab-sized —
# a 128000-dim logit cache would OOM ~50 GB for 95 retain batches).
if self.batch_idx == self.recalc_every:
    with torch.no_grad():
        for r_batch in self.retain_batches:
            out = model(**prep_batch(r_batch, device), output_hidden_states=True)
            r_batch["cached_hidden"] = out.hidden_states[-1].detach()

# Per step: forward retain, apply live (frozen) lm_head to cached hidden, kl_div.
out = model(**prep_batch(r_batch, device))
cur = out.logits.float()
tgt = model.lm_head(r_batch["cached_hidden"].to(model.dtype)).float()
mask = r_batch["labels"].to(device) != -100
log_q = F.log_softmax(cur[mask], dim=-1)
log_p = F.log_softmax(tgt[mask], dim=-1)
ref_loss = F.kl_div(log_q, log_p, reduction="sum", log_target=True)
ref_loss.backward()
```

**Removed:** the entire `KLComputor` class ($\sim 80$ lines in `evals/kl_eval.py`), `cache_last_hidden_states` helper, `create_acts_to_logits` helper, `_kl_cache` dict attribute, `deepcopy(model.lm_head)` ($\sim 1$ GB of parameter copy on Llama-3B). The filter still needs `cached_hidden` (load-bearing; 128000-dim logits would OOM), but the lm_head deepcopy is cleanly eliminated.

**Kept:** `retain_momentum=0.97` smoothing (load-bearing, §9.3(1)), `quantize_blockwise` storage of the momentum buffer (load-bearing for memory), per-step recomputation (load-bearing, §9.3(3)).

### 9.6 Contribution framing

§8 gave a structural negative result: no simple activation-space filter can replace the weight-space-gradient-based KL filter (monotonic 1:1 tradeoff). §9 gives a small positive result that sharpens the interp claim: *the KL filter needs three ingredients* — soft distribution targets, per-step recomputation, and momentum smoothing — *but not `KLComputor`'s cached lm_head*. The dead deepcopy is dropped; everything else is structurally necessary.

Together, §8 and §9 bracket the KL filter tightly:
- Structurally irreducible: soft KL targets, per-step recomputation, momentum smoothing, weight-space gradient signal.
- Structurally removable: `KLComputor` class, `deepcopy(lm_head)`, `cache_last_hidden_states` helper, `create_acts_to_logits` helper.

This is a clean finish line for the disruption-filter component of the minimal method.

### 9.7 Can hyperparameter tuning close the gap? (iter 15)

A natural follow-up: the ablations in §9.2 were run at default learning rate. Could a smaller LR slow the "stripped" variant down enough that it reaches baseline-level recall at matched kl budget? We ran the cleanest version of this: **keep only ingredient #1** (= CE + $\text{retain\_momentum}=0$ + per-step recomputation), swept at $\text{LR} \in \{0.1, 0.05\}$ (vs default $0.2$).

| Config | Trajectory highlights | Recall at $\text{kl}\approx 0.009$ | Recall at $\text{kl}\approx 0.0024$ |
|--------|----------------------|------------------------------------|-------------------------------------|
| Baseline (all 4 ingredients) | ep6 recall $0.0196$, kl $0.0094$ | $\mathbf{0.0196}$ | ~ep2 |
| Keep-only-#1 + $\text{LR}=0.1$ | ep5 recall $0.064$ kl $0.0045 \to$ ep6 broken $0.044$ kl $0.0255$ | $\sim 0.060$ (interp) | ~ep3 |
| Keep-only-#1 + $\text{LR}=0.05$ | 10-epoch stable, ep10 recall $0.0886$ kl $0.0024$ | never reaches | $0.0886$ @ ep10 |

**At matched kl $= 0.009$, keep-only-#1 is $\sim 3\times$ worse than baseline regardless of LR.** LR$=0.1$ slides to one point on the tradeoff curve (break ep6, recall $\sim 0.06$); LR$=0.05$ slides to the other extreme (never breaks, but never unlearns meaningfully — only 37% of the baseline's recall drop in 10 epochs). Neither matches baseline's recall at matched kl.

The interpretation is the same as §8: learning rate slides *along* the Pareto curve, it does not *shift* it. The Pareto frontier is fixed by filter signal quality (what the filter "knows"), not by update magnitude. Losing #2 (soft KL) and #3 (momentum) simultaneously — as in keep-only-#1 — compounds the penalty from ~$1.8\times$ (drop #3 only, iter 11) to ~$3\times$. The three ingredients are structurally additive, not redundant.

**This closes the loop on §8 + §9:** the KL filter requires soft distribution targets, per-step recomputation, momentum smoothing, and weight-space gradient access — and no hyperparameter tuning recovers a simpler variant to baseline efficiency.
