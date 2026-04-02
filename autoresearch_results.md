# DISCO Autoresearch Results

## Target to beat (RepCollapse baseline, default params, WMDP-Bio, Llama-3.2-3B)
| Metric | After Unlearn | After Relearn |
|--------|--------------|---------------|
| wikitext_kl | 0.0026 | 0.31 |
| recall_prob | 0.0247 | 0.0226 |
| forget_acc_t0 | 0.366 | 0.317 |
| forget_acc_t1 | 0.309 | 0.304 |
| Robustness | — | 0.031 |

## Base model (no unlearning)
| Metric | Value |
|--------|-------|
| forget_acc_t0 | 0.4146 |
| forget_acc_t1 | 0.3697 |
| recall_prob | 0.1092 |

## DISCO Experiments

### Exp 0: DISCO baseline (lr=0.2, n_pcs=200)
- Status: BROKEN at epoch 2 (wikitext_kl=3.67)
- After relearn: recall_prob=0.0863, forget_acc_t0=0.439, Robustness=0.115
- Problem: LR way too high, model destroyed immediately

### Exp 1-3: LR grid search (Phase 1)
