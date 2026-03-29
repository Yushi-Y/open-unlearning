#!/bin/bash
set +e
cd /VData/kebl6672/open-unlearning
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
source /VData/kebl6672/open-unlearning/.env
export HF_HOME HF_TOKEN
eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

echo "=========================================="
echo "GPU 0: Exp 7+8 for Llama-3.2-3B"
echo "=========================================="

# Exp 7: Cross-distribution PCA similarity
echo "[$(date)] Exp 7: Cross PCA similarity Llama Bio..."
python3 scripts/pca_cross_similarity.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_CROSS_SIM_LLAMA_BIO 2>&1

echo "[$(date)] Exp 7: Cross PCA similarity Llama Cyber..."
python3 scripts/pca_cross_similarity.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_CROSS_SIM_LLAMA_CYBER 2>&1

# Exp 8: Per-token loss comparison
echo "[$(date)] Exp 8: Per-token loss Llama Bio..."
python3 scripts/per_token_loss_comparison.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    task_name=TOKEN_LOSS_LLAMA_BIO 2>&1

echo "[$(date)] GPU 0 Exp 7+8 ALL DONE"
