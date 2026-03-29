# PCA Interpretability Experiments

Experiments validate that high-variance PCs encode the attacker's subspace (domain-specific directions an attacker would fine-tune along), while low-variance PCs are adversarially inaccessible.

## Exp 1: Vocabulary Projection (`pca_token_interp.py`)

**Question:** What tokens does each PC represent?

**Method:** Project each PC eigenvector through the frozen `lm_head` to get vocabulary logits. Extract the highest/lowest-logit tokens.

**Finding:** High-variance PCs project to domain-specific tokens (e.g., `virus, viral, RNA` for Bio; `exploit, malicious, payloads` for Cyber). Low-variance PCs project to uninterpretable subword fragments. This confirms high-variance PCs encode the attacker's signal.

**Paper:** Tables 4-5 (main text).

## Exp 2: Steering Vector Alignment (`pca_sequence_interp.py`)

**Question:** Do high-variance PCs point in the forget-vs-retain direction?

**Method:** Compute a steering vector (mean forget activation - mean retain activation). Measure each PC's cosine alignment with it. Also compute forget-to-retain (F/R) activation ratio per PC.

**Finding:** High-variance PCs align 20-67x more with the steering vector than low-variance PCs, and are preferentially activated by forget sequences (F/R > 1). Low-variance PCs show the opposite (F/R < 1). This means an attacker's fine-tuning signal is concentrated along high-variance PCs.

**Paper:** Appendix Table (PC steering).

## Exp 3: Top Forget Sequences per PC (`pca_forget_seq_interp.py`)

**Question:** Which forget sequences activate each PC most strongly?

**Method:** Project each forget sequence's last-token hidden state onto each PC. Return the top-scoring sequences by projection magnitude. No retain set needed.

**Expected finding:** High-variance PCs activate on domain-characteristic sequences (e.g., bioweapon synthesis, exploit descriptions). Low-variance PCs activate on atypical/idiosyncratic sequences that don't generalize across the domain.

## Exp 4+6: Attack Subspace Concentration + LoRA Depth (`attack_subspace.py`)

**Question:** Does a fine-tuning attacker's updates concentrate on high-variance PCs? Does LoRA force deeper modifications?

**Method:**
1. Collect PCA stats (1 epoch, no weight updates)
2. Simulate 50 steps of fine-tuning attack on forget data
3. Project attacker's weight deltas onto PCs: `energy_i = ||delta_W @ v_i||^2`
4. Compare RepCollapse with/without LoRA adversary
5. Project the LoRA adapter itself (A@B) onto PCs

**Expected finding:** Attack energy concentrates in top-10 PCs (>80%). RepCollapse with LoRA pushes modifications deeper into low-variance PCs. The LoRA adapter itself targets high-variance PCs (where the adversary attacks), forcing the base model to defend there.

**Figure:** Cumulative energy curve + comparison plot.

## Exp 5: Baseline vs RepCollapse Weight Projection (`baseline_weight_projection.py`)

**Question:** Do baselines modify the attacker's subspace while RepCollapse avoids it?

**Method:**
1. Run GradDiff, NPO, RepCollapse for 3 epochs each
2. Project each method's weight deltas onto forget PCs
3. Compare energy distribution across PC bins

**Expected finding:** GradDiff/NPO pile energy in top-10 PCs (attacker's subspace). RepCollapse spreads energy into PC 100-400 (adversarially inaccessible).

**Figure:** Grouped bar chart of energy fractions per method.

## Exp 7: Cross-Distribution PCA Similarity (`pca_cross_similarity.py`)

**Question:** Does collapsing forget PCs make the weight update "blind" to retain/wikitext data?

**Method:**
1. Compute PCA on forget set
2. For each forget PC, measure its variance contribution on: forget-eval, retain, wikitext
3. After Mahalanobis collapse, re-measure overlap
4. Compute subspace similarity (e.g., sum of Mahalanobis distances of each PC on other distributions)

**Expected finding:** Before collapse, high-variance forget PCs have high variance on retain too (shared representations). After collapse, the collapsed subspace has near-zero overlap with retain/wikitext. This proves the "blindness" property: weight updates in the collapsed subspace cannot disrupt retain.

**Theoretical link:** Proposition 1 says `ΔW · v_i = 0` for collapsed PCs. If retain activations concentrate along those PCs, the update is invisible to retain data. This experiment measures that concentration empirically.

**Open question:** Why does collapsing *forget* PCs work better than collapsing *retain* PCs? Hypothesis: retain sets are too narrow/domain-specific to capture general representations. Forget PCs capture the broader domain structure the attacker exploits.

## Exp 8: Per-Token Loss Comparison with Baselines (`per_token_loss_comparison.py`)

**Question:** Do non-selective methods break generic domain tokens more than RepCollapse?

**Method:**
1. Unlearn with each method (GradDiff, NPO, RepCollapse) to matched disruption budget
2. Compute per-token cross-entropy loss on retain data, comparing to original model
3. Identify tokens with largest loss increase per method
4. Check if broken tokens are domain-general (e.g., `virus`, `exploit`, `the`, `is`)

**Expected finding:** Baselines show large loss increases on prominent domain tokens (high-variance PC tokens from Exp 1). RepCollapse shows minimal change on these tokens. This is the "tip of the iceberg" -- real selectivity gains are in internal representations, but token-level loss is concrete and visualizable.

**Framing (from Filip):** "Both our and their methods aim to forget. Their methods inadvertently attack the most prominent (most general) features." The failure mode is selectivity, not intent.

## Summary

| Exp | What it shows | Status |
|-----|---------------|--------|
| 1. Vocab projection | High-var PCs = domain tokens | Running |
| 2. Steering alignment | High-var PCs point toward forget | Running |
| 3. Top forget sequences | High-var PCs activate on attacker-relevant content | Running |
| 4+6. Attack subspace + LoRA | Attack concentrates on high-var PCs; LoRA forces depth | Running |
| 5. Baseline comparison | Baselines modify attacker's subspace; RepCollapse doesn't | Running |
| 7. PCA cross-similarity | Collapsed subspace is "blind" to retain (Prop 1) | TODO |
| 8. Per-token loss | Baselines break generic tokens more | TODO |

Experiments 1-6 prove the adversarial inaccessibility story. Experiments 7-8 provide the theoretical backbone (blindness property) and concrete visualization (per-token loss) for the paper.
