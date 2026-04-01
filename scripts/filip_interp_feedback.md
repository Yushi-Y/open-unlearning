# Filip's Interpretability Feedback (Mar 30, 2026)

## On Section 3 (high/low-variance PCs)
- The claim "low-variance PCs encode benign noise, therefore suppress high-variance PCs" doesn't follow. If top PCs have harmful knowledge, that's a reason to unlearn *on* them, not collapse them.
- "Looks like noise under logit lens" doesn't mean benign — just uninterpretable.
- Inferential gap in current framing.

## On steering vector (A.2)
- LDA-based separation would be more principled than difference-of-means as the steering vector baseline.

## On LoRA sentence
- Claim that LoRA adversary "forces updates into deeper low-variance directions" not empirically verified — Table 10 shows +LoRA vs -LoRA numbers nearly identical. Don't claim what you can't show.

## On A.7 (per-token retain loss)
- Comparison confounded by different unlearning levels. GradDiff may simply be more aggressive, not inherently more disruptive per unit of forgetting. Need to control for amount of unlearning.

## On A.3 (cross-distribution variance) — MAIN POSITIVE
- Wants tiered analysis: variance ratios (forget/retain/wikitext) for PC tiers 0-3, 3-10, 10-30, 100-300.
- Key insight: the **ratio** forget-variance/retain-variance matters, not just absolute forget-variance.
- This would directly motivate which PCs to collapse vs. unlearn on.
- "I think showing this would be an amazing motivation for top PC collapse."
- Also wants same done for retain-set PCs — may help answer why forget PCA works better than retain PCA.

## On A.4 (top sequences per PC)
- Agrees high-PCs activate on common themes, low-PCs on rare/niche content.
- Pushes back on calling low-PC content "benign" — more accurately "niche, rare, specific."
- Wants: more examples per domain, consistent PC indexing (not cherry-picking), possibly ask Opus to characterise differences blind.
- Suggests: sort by |cosine similarity| rather than projection magnitude.

## On A.5/A.6 (weight projection)
- Asks: how do you project 2D weight update ΔW onto 1D PC? Need to clarify mechanics (row-wise norm or Frobenius).

## Summary
- A.3 and A.4 most promising, especially A.3 with tiered analysis.
- Some claims in text overreach the evidence.

## Additional ideas 

### Collapse interpretation via nearest neighbors
- For a forget activation, collapse it, then find most similar activations from forget+retain. Three variants:
  1. Most similar to raw activation before collapse
  2. Most similar to activation after collapse
  3. Most similar to remainder (raw minus collapsed)
- Interesting if post-collapse similarities are more meaningful than raw.
- More powerful than individual PC analysis — uses all top PCs at once.

### Reverse similarity (advanced)
- Find examples known to be meaningfully related (e.g., paraphrases of same fact).
- Construct transformation to make activations more similar to related texts, relative to unrelated.

### Why forget PCA > retain PCA?
- Counterintuitive that collapsing based on forget PCA works better than retain PCA.
- Hypothesis: retain set not adequate — maybe more similar retain set would perform better.
