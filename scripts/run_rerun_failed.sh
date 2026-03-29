#!/bin/bash
# Rerun Exp 4+6 and Exp 5 for Llama on GPU 0
set +e
cd /VData/kebl6672/open-unlearning
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
source /VData/kebl6672/open-unlearning/.env
export HF_HOME HF_TOKEN
eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

echo "=========================================="
echo "Rerunning Exp 4+6 and 5 on GPU 0 (Llama)"
echo "=========================================="

# Exp 4+6: Attack subspace Llama Bio (no LoRA override — script handles internally)
echo "[$(date)] Exp 4+6: Attack subspace Llama Bio..."
python3 scripts/attack_subspace.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    task_name=ATTACK_SUB_LLAMA_BIO 2>&1 | tee saves/pca_interp/log_attack_llama_bio.txt

echo "[$(date)] Exp 4+6: Attack subspace Llama Cyber..."
python3 scripts/attack_subspace.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    task_name=ATTACK_SUB_LLAMA_CYBER 2>&1 | tee saves/pca_interp/log_attack_llama_cyber.txt

# Exp 5: Baseline comparison (needs PCA stats from above)
echo "[$(date)] Exp 5: Baseline comparison Llama Bio..."
python3 scripts/baseline_weight_projection.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B \
    trainer=RepSelect \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    task_name=BASELINE_CMP_LLAMA_BIO 2>&1 | tee saves/pca_interp/log_baseline_llama_bio.txt

echo "[$(date)] GPU 0 rerun ALL DONE"
