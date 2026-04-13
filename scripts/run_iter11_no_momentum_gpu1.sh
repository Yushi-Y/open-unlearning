#!/bin/bash
# Iter 11: KL filter simplification #1 — drop momentum (retain_momentum=0).
# Same KL forward signal but without smoothing or quantize machinery overhead.
set +e
cd /VData/kebl6672/open-unlearning
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled
source /VData/kebl6672/open-unlearning/.env
export HF_HOME HF_TOKEN
eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

BASE="python3 src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=Llama-3.2-3B trainer.args.per_device_train_batch_size=4"
SC="trainer=SelectiveCollapse trainer.method_args.cfg.collapse_attn=true ~eval.fewshot_attack ~eval.fewshot_attack_10"

echo "[$(date)] Iter11 pilot: KL filter, retain_momentum=0 — Llama Bio"
$BASE $SC \
    trainer.method_args.cfg.pca_source=contrastive_power_iter \
    trainer.method_args.cfg.boost_beta=3 \
    +trainer.method_args.cfg.collapse_rule=complement_proj \
    trainer.method_args.cfg.retain_momentum=0 \
    task_name=iter11_no_momentum_llama_bio

echo "[$(date)] Iter11 pilot DONE"
