#!/bin/bash
# Iter 16: NOVEL two-sided retain null-space weight projection.
# Replaces KL filter with: grads ← (I - V_r^out V_r^out^T) grads,
# alongside the existing acts ← (acts - μ)(I - V V^T) collapse rule.
# Interp claim: a weight update can only disrupt retain if it couples a retain
# input direction to a retain output direction; killing either side suffices.
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

echo "[$(date)] Iter16 pilot: two-sided retain proj, no KL filter, no LoRA — Llama Bio"
$BASE $SC \
    trainer.method_args.cfg.pca_source=contrastive_power_iter \
    trainer.method_args.cfg.boost_beta=3 \
    +trainer.method_args.cfg.collapse_rule=complement_proj \
    ~trainer.method_args.cfg.retain_momentum \
    +trainer.method_args.cfg.two_sided_retain_proj=true \
    task_name=iter16_two_sided_llama_bio
echo "[$(date)] Iter16 DONE"
