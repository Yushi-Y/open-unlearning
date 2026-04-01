# New Unlearning Method: Generalised Eigenvalue Approach

## Motivation 
The current RepCollapse method collapses top-k forget PCs (attacker subspace) and unlearns on the residual. This works, but implicitly assumes retain covariance is isotropic (Σ_r ≈ I) within the complement — i.e., it doesn't account for the structure of retain activations when choosing where to unlearn.

What actually matters is not absolute forget variance per PC, but the **ratio** of forget variance to retain variance. A direction with high forget/retain ratio is selective — modifying it forgets without disrupting retain. A direction with low ratio is shared — modifying it causes collateral damage.

## The Three-Step Ideal

1. **Exclude attacker subspace**: Compute top-k eigenvectors of Σ_f, call this V_k. Project onto complement: P_⊥ = I - V_k V_k⊤
2. **Rank by forget/retain ratio within complement**: Solve the generalised eigenvalue problem restricted to complement:
   - P_⊥ Σ_f P_⊥ v = λ P_⊥ Σ_r P_⊥ v
   - Eigenvectors ranked by λ = forget/retain variance ratio
   - Top eigenvectors: most selective for forget, already outside attacker subspace
3. **Unlearn on top-ratio directions**: Project activations onto top-m generalised eigenvectors, use in weight update

## Math: Generalised Eigenvalue Problem

**What we want**: For each direction v, define selectivity score:
```
score(v) = v⊤ Σ_f v / v⊤ Σ_r v
```

**Maximising this** gives the generalised eigenvalue problem:
```
Σ_f v = λ Σ_r v
```
where λ = score(v). Eigenvectors are optimal selectivity directions ranked by eigenvalue.

**Contrast with regular PCA**: PCA solves Σ_f v = λ v, which maximises absolute forget variance with no denominator. It's the special case where Σ_r = I. That assumption is wrong — retain activations are structured.

## How Current RepCollapse Approximates This

- **Step 1** (exclude attacker subspace): ✅ Correct — collapses V_k
- **Step 2** (rank by ratio): ❌ Assumes Σ_r ≈ I within complement (PCA assumption)
- **Step 3** (unlearn on top ratio directions): ❌ Takes single residual direction per token (Eq. 2) rather than top-m selective directions

Gap is entirely in steps 2 and 3.

## Connection to LDA

Step 2 is exactly LDA restricted to the complement subspace. LDA finds directions maximising between-class separation relative to within-class spread.

## Efficiency

Current approach: One SVD of Σ_f per epoch per module → O(d² k)

New approach:
- Same SVD of Σ_f for step 1
- Plus computing Σ_r → same cost as Σ_f
- Plus generalised SVD of (P_⊥ Σ_f P_⊥, P_⊥ Σ_r P_⊥) → O(d² m)

Roughly 2-3× more expensive, same asymptotic complexity.

## Practical Considerations

- Σ_r estimation needs enough retain samples. With d=4096, want ~20k-40k samples for stability. Current retain set is 1000 from FineFineWeb (expandable).
- **Cheaper alternative**: Diagonal approximation of Σ_r — per-dimension retain variance. O(d) memory vs O(d²), far fewer samples needed.
- Run tiered A.3 first to see if the gap is large enough to justify added complexity.

## Key Experiment: Tiered A.3 Analysis

Show variance ratios (forget/retain/wikitext) for PC tiers: 0-3, 3-10, 10-30, 30-100, 100-300.

- If PCs 100-300 have 10× more forget than retain variance, and PCs 0-3 have only 2×, this directly motivates collapsing top PCs and unlearning on bottom PCs.
- If retain variance is roughly uniform across tiers, PCA ≈ generalised approach and current method is near-optimal.
- Also do same analysis with retain-set PCs to understand why forget PCA works better than retain PCA.

## Why This Could Explain Figure 4

Top PCs are shared between forget and retain (Table 8: top-10 forget PCs explain 11-15% of retain variance). Baselines modifying those directions simultaneously disrupt retain, hitting KL budget fast. RepCollapse operates in low-variance subspace where retain activations concentrate less → less collateral damage per step → more unlearning steps within budget.

Same geometric fact explains both:
- **Robustness**: attacker can't follow into low-variance directions
- **Efficient forgetting**: low-variance directions overlap less with retain
