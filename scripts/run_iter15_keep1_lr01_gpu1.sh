#!/bin/bash
# Iter 15: "keep only #1" LR sweep — CE + mom=0 + per-step, LR=0.1 (half default).
# Tests whether LR tuning can shift the Pareto curve or just slides along it.
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

echo "[$(date)] Iter15: keep only #1 (CE + mom=0 + per-step) LR=0.1"
$BASE $SC \
    trainer.args.learning_rate=0.1 \
    trainer.method_args.cfg.pca_source=contrastive_power_iter \
    trainer.method_args.cfg.boost_beta=3 \
    +trainer.method_args.cfg.collapse_rule=complement_proj \
    trainer.method_args.cfg.retain_momentum=0 \
    +trainer.method_args.cfg.retain_grad_source=ce \
    task_name=iter15_keep1_lr01_llama_bio
echo "[$(date)] Iter15 LR=0.1 DONE"
