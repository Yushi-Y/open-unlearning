#!/bin/bash
# Iter 13: Light KL — cache retain logits with live (frozen) lm_head.
# No KLComputor, no lm_head deepcopy, no cached hidden states, no acts_to_logits.
# Should match baseline EXACTLY since math is identical (KL with same cached targets).
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

echo "[$(date)] Iter13 pilot: light_kl + mom=0.97 — Llama Bio"
$BASE $SC \
    trainer.method_args.cfg.pca_source=contrastive_power_iter \
    trainer.method_args.cfg.boost_beta=3 \
    +trainer.method_args.cfg.collapse_rule=complement_proj \
    +trainer.method_args.cfg.retain_grad_source=light_kl \
    task_name=iter13_lightkl_llama_bio

echo "[$(date)] Iter13 pilot DONE"
