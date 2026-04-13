#!/bin/bash
# Iter 12: NOVEL — frozen retain gradient. Compute CE-grad ONCE at end of epoch 0
# (averaged over all retain batches), reuse forever. No momentum, no KLComputor,
# no per-step recomputation. Tests interp claim: "retain protective direction is
# approximately stable during unlearning".
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

echo "[$(date)] Iter12 pilot: frozen retain CE-grad (compute once, reuse) — Llama Bio"
$BASE $SC \
    trainer.method_args.cfg.pca_source=contrastive_power_iter \
    trainer.method_args.cfg.boost_beta=3 \
    +trainer.method_args.cfg.collapse_rule=complement_proj \
    +trainer.method_args.cfg.frozen_retain_grad=true \
    task_name=iter12_frozen_grad_llama_bio

echo "[$(date)] Iter12 pilot DONE"
