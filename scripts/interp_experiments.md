# PCA Interpretability Experiments

Three experiments validate that high-variance PCs encode the attacker's subspace (domain-specific directions an attacker would fine-tune along), while low-variance PCs are adversarially inaccessible.

## Exp 1: Vocabulary Projection (`pca_token_interp.py`)

**Question:** What tokens does each PC represent?

**Method:** Project each PC eigenvector through the frozen `lm_head` to get vocabulary logits. Extract the highest/lowest-logit tokens.

**Finding:** High-variance PCs project to domain-specific tokens (e.g., `virus, viral, RNA` for Bio; `exploit, malicious, payloads` for Cyber). Low-variance PCs project to uninterpretable subword fragments. This confirms high-variance PCs encode the attacker's signal.

**Paper:** Tables 4-5 (main text).

```bash
python scripts/pca_token_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    ~trainer.method_args.cfg.lora_rank \
    task_name=PCA_TOKEN_INTERP_Llama
```

## Exp 2: Steering Vector Alignment (`pca_sequence_interp.py`)

**Question:** Do high-variance PCs point in the forget-vs-retain direction?

**Method:** Compute a steering vector (mean forget activation - mean retain activation). Measure each PC's cosine alignment with it. Also compute forget-to-retain (F/R) activation ratio per PC.

**Finding:** High-variance PCs align 20-67x more with the steering vector than low-variance PCs, and are preferentially activated by forget sequences (F/R > 1). Low-variance PCs show the opposite (F/R < 1). This means an attacker's fine-tuning signal is concentrated along high-variance PCs.

**Paper:** Appendix Table (PC steering).

```bash
python scripts/pca_sequence_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    ~trainer.method_args.cfg.lora_rank \
    task_name=PCA_SEQ_INTERP_LLAMA_BIO
```

## Exp 3: Top Forget Sequences per PC (`pca_forget_seq_interp.py`)

**Question:** Which forget sequences activate each PC most strongly?

**Method:** Project each forget sequence's last-token hidden state onto each PC. Return the top-scoring sequences by projection magnitude. No retain set needed.

**Expected finding:** High-variance PCs activate on domain-characteristic sequences (e.g., bioweapon synthesis, exploit descriptions) — exactly the content an attacker would fine-tune on. Low-variance PCs activate on atypical/idiosyncratic sequences that don't generalize across the domain.

```bash
python scripts/pca_forget_seq_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    ~trainer.method_args.cfg.lora_rank \
    task_name=PCA_FORGET_SEQ_LLAMA_BIO
```

## Summary

| Exp | Granularity | Data needed | What it shows |
|-----|------------|-------------|---------------|
| 1. Vocab projection | Token-level | None (lm_head only) | High-var PCs = domain tokens |
| 2. Steering alignment | Sequence-level | Forget + Retain | High-var PCs point toward forget |
| 3. Top forget sequences | Sequence-level | Forget only | High-var PCs activate on attacker-relevant content |

All three converge: high-variance PCs encode the attacker's subspace. Collapsing them makes unlearning adversarially inaccessible.
