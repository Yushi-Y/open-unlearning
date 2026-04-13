#!/bin/bash
# Iter 14: light_kl validation on Llama Cyber (baseline 0.015).
set +e
cd /VData/kebl6672/open-unlearning
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=disabled
source /VData/kebl6672/open-unlearning/.env
export HF_HOME HF_TOKEN
eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

SC="trainer=SelectiveCollapse trainer.method_args.cfg.collapse_attn=true ~eval.fewshot_attack ~eval.fewshot_attack_10"
COMMON="trainer.method_args.cfg.pca_source=contrastive_power_iter trainer.method_args.cfg.boost_beta=3 +trainer.method_args.cfg.collapse_rule=complement_proj +trainer.method_args.cfg.retain_grad_source=light_kl"

echo "[$(date)] Iter14: light_kl — Llama Cyber"
python3 src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=Llama-3.2-3B trainer.args.per_device_train_batch_size=4 wmdp_domain=cyber \
    $SC $COMMON task_name=iter14_lightkl_llama_cyber
echo "[$(date)] Iter14 Llama Cyber DONE"
