#!/bin/bash
set +e
cd /VData/kebl6672/open-unlearning
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=disabled
source /VData/kebl6672/open-unlearning/.env
export HF_HOME HF_TOKEN
eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

echo "=========================================="
echo "GPU 3: Exp 7+8 for Qwen3-8B-Base"
echo "=========================================="

# Exp 7: Cross-distribution PCA similarity
echo "[$(date)] Exp 7: Cross PCA similarity Qwen Bio..."
python3 scripts/pca_cross_similarity.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_CROSS_SIM_QWEN_BIO 2>&1

echo "[$(date)] Exp 7: Cross PCA similarity Qwen Cyber..."
python3 scripts/pca_cross_similarity.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_CROSS_SIM_QWEN_CYBER 2>&1

# Exp 8: Per-token loss comparison
echo "[$(date)] Exp 8: Per-token loss Qwen Bio..."
python3 scripts/per_token_loss_comparison.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    task_name=TOKEN_LOSS_QWEN_BIO 2>&1

echo "[$(date)] GPU 3 Exp 7+8 ALL DONE"
