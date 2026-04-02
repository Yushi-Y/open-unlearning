# DISCO Autoresearch — Llama-3.2-3B, WMDP-Bio

**Target:** RepCollapse robustness = 0.031

## Iteration Log

| iter | robustness | epochs | status | change |
|------|-----------|--------|--------|--------|
| 0 | 0.115 | 2/10 | base | Original DISCO (reconstruction, low-rank Σ_r) |
| 1 | 0.114 | 2/10 | discard | Mahalanobis subtraction |
| 2 | 0.114 | 2/10 | discard | Diagonal Σ_r (full-rank whitening, λ: 8→65) |
| 3 | 0.096 | 7/10 | **KEEP** | + DISCO gradient collapse |
| 4 | 0.113 | 2/10 | discard | Skip zero-active → raw grads |
| 5 | 0.111 | 9/10 | discard | No double projection |
| 6 | 0.113 | 2/10 | discard | Act all + grad selective |
| 7 | 0.066 | 6/10 | **KEEP** | + KL token masking |
| 8 | 0.037 | 7/10 | **BEST** | + LoRA adversary |
| 9 | 0.071 | 2/10 | discard | Post-hoc weight projection |
| 10a | 0.115 | 2/10 | discard | Adaptive threshold (median) |
| 10b | 0.115 | 2/10 | discard | Diagonal grad DISCO fallback |
| 11a | 0.082 | 4/10 | discard | Joint Kronecker weighting |
| 11b | 0.082 | 4/10 | discard | Token selectivity weighting |
| 12 | 0.115 | 2/10 | discard | Soft Mahalanobis (normalize by min) |

Note: epochs = training epochs before wikitext_kl exceeded 0.01 budget (epoch 0-1 are data collection, training starts at epoch 2). RepCollapse ran all 10/10 epochs without breaking.

## 3 changes that stuck (cumulative)

1. **Diagonal Σ_r + DISCO grad collapse** (iter 2-3): 0.115 → 0.096
2. **KL token masking** (iter 7): 0.096 → 0.066
3. **LoRA adversary** (iter 8): 0.066 → 0.037

## Key finding

The hard cutoff `w = max(0, 1 - 1/λ)` is correct — when gradient DISCO finds λ < 1 (retain dominates), zeroing is the principled answer. Every attempt to activate those directions (iters 4, 6, 10a, 10b, 11a, 12) broke the model.

## Hyperparameters used (single config, no sweep)

### Llama-3.2-3B

**DISCO (iter 8 — best):**
```yaml
trainer.args.learning_rate: 0.2
trainer.args.optim: sgd
trainer.args.num_train_epochs: 10
trainer.args.per_device_train_batch_size: 8
trainer.method_args.cfg.n_pcs_select: 200
trainer.method_args.cfg.reg_eps: 1e-4
trainer.method_args.cfg.retain_momentum: 0.97
trainer.method_args.cfg.lora_lr: 0.1
trainer.method_args.cfg.lora_rank: 32
```

**RepCollapse (baseline):**
```yaml
trainer.args.learning_rate: 0.2
trainer.args.optim: sgd
trainer.args.num_train_epochs: 10
trainer.args.per_device_train_batch_size: 8
trainer.method_args.cfg.n_pcs: 400
trainer.method_args.cfg.retain_momentum: 0.97
trainer.method_args.cfg.lora_lr: 0.1
trainer.method_args.cfg.lora_rank: 32
```

## Comparison: DISCO vs RepCollapse (Llama-3.2-3B, default params)

| Metric | RepCollapse | DISCO (iter 8) |
|--------|-----------|----------------|
| Robustness (recall_prob) | 0.031 | 0.037 |
| forget_acc_t0 (after relearn) | 0.317 | 0.366 |
| forget_acc_t1 (after relearn) | 0.304 | 0.342 |
| recall_prob (after relearn) | 0.023 | 0.027 |

## What differs between DISCO and RepCollapse

| Component | RepCollapse | DISCO (iter 8) |
|-----------|------------|----------------|
| Act collapse | PCA Mahalanobis (forget-only covariance) | DISCO Mahalanobis (forget/retain ratio, diagonal Σ_r) |
| Grad collapse | PCA Mahalanobis (forget-only covariance) | DISCO (forget/retain ratio) — gate/up zeroed (λ<1) |
| KL masking | ✓ | ✓ (same) |
| LoRA adversary | ✓ | ✓ (same) |
| Weighting | `1 - λ_min/λ` (soft, all dirs active) | `max(0, 1 - 1/λ)` (hard cutoff at λ=1) |
| Retain backward (epoch 0) | None | Extra pass for gradient DISCO stats |

## Qwen3-8B-Base comparison

Running: DISCO at lr=0.05, lr=0.1 + RepCollapse at default lr=0.2. Results pending.

| Config | Robustness | Notes |
|--------|-----------|-------|
| DISCO lr=0.2 | 0.137 | Broke at epoch 2, too aggressive |
| DISCO lr=0.1 | pending | |
| DISCO lr=0.05 | pending | |
| RepCollapse lr=0.2 | pending | |
