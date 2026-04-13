#!/bin/bash
# Iter 9: Candidate C extreme — frac=0.9 (reject top 90% of gnorm*quad_r).
set +e
cd /VData/kebl6672/open-unlearning
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=disabled
source /VData/kebl6672/open-unlearning/.env
export HF_HOME HF_TOKEN
eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

BASE="python3 src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=Llama-3.2-3B trainer.args.per_device_train_batch_size=4"
SC="trainer=SelectiveCollapse trainer.method_args.cfg.collapse_attn=true ~eval.fewshot_attack ~eval.fewshot_attack_10"

echo "[$(date)] Iter9 pilot: C frac=0.9 extreme — Llama Bio"
$BASE $SC \
    trainer.method_args.cfg.pca_source=contrastive_power_iter \
    trainer.method_args.cfg.boost_beta=3 \
    +trainer.method_args.cfg.collapse_rule=complement_proj \
    ~trainer.method_args.cfg.retain_momentum \
    +trainer.method_args.cfg.token_filter=retain_cov_reject \
    +trainer.method_args.cfg.token_filter_tau=0.9 \
    task_name=iter9_candC_frac09_llama_bio

echo "[$(date)] Iter9 pilot DONE"
