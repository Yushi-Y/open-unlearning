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
echo "GPU 0: Exp 7 (fixed) + Exp 5 (5 baselines)"
echo "=========================================="

# Exp 7: Cross PCA similarity (fixed labels issue)
echo "[$(date)] Exp 7: Cross PCA similarity Llama Bio..."
python3 scripts/pca_cross_similarity.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepCollapse \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_CROSS_SIM_LLAMA_BIO 2>&1

echo "[$(date)] Exp 7: Cross PCA similarity Llama Cyber..."
python3 scripts/pca_cross_similarity.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepCollapse \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_CROSS_SIM_LLAMA_CYBER 2>&1

# Exp 5: Baseline comparison with 5 methods
echo "[$(date)] Exp 5: Baseline comparison Llama Bio (5 methods)..."
python3 scripts/baseline_weight_projection.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B \
    trainer=RepCollapse \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    task_name=BASELINE_5M_LLAMA_BIO 2>&1

echo "[$(date)] GPU 0 ALL DONE"
