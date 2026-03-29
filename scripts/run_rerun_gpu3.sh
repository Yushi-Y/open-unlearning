#!/bin/bash
# Rerun Exp 4+6 and Exp 5 for Qwen + Gemma Exp 1,3 on GPU 3
set +e
cd /VData/kebl6672/open-unlearning
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=disabled
source /VData/kebl6672/open-unlearning/.env
export HF_HOME HF_TOKEN
eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

echo "=========================================="
echo "Rerunning failed experiments on GPU 3"
echo "=========================================="

# --- Exp 4+6: Attack subspace Qwen Bio ---
echo "[$(date)] Exp 4+6: Attack subspace Qwen Bio..."
python3 scripts/attack_subspace.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    task_name=ATTACK_SUB_QWEN_BIO 2>&1 | tee saves/pca_interp/log_attack_qwen_bio.txt

# --- Exp 4+6: Attack subspace Qwen Cyber ---
echo "[$(date)] Exp 4+6: Attack subspace Qwen Cyber..."
python3 scripts/attack_subspace.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepSelect \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    task_name=ATTACK_SUB_QWEN_CYBER 2>&1 | tee saves/pca_interp/log_attack_qwen_cyber.txt

# --- Exp 5: Baseline comparison Qwen Bio ---
echo "[$(date)] Exp 5: Baseline comparison Qwen Bio..."
python3 scripts/baseline_weight_projection.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base \
    trainer=RepSelect \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    task_name=BASELINE_CMP_QWEN_BIO 2>&1 | tee saves/pca_interp/log_baseline_qwen_bio.txt

# --- Gemma Exp 1+3 (also failed) ---
echo "[$(date)] Exp 1: Token interp Gemma Bio..."
python3 scripts/pca_token_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Gemma-3-1B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_TOKEN_INTERP_GEMMA_BIO 2>&1 | tee saves/pca_interp/log_token_gemma_bio.txt

echo "[$(date)] Exp 1: Token interp Gemma Cyber..."
python3 scripts/pca_token_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Gemma-3-1B trainer=RepSelect \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_TOKEN_INTERP_GEMMA_CYBER 2>&1 | tee saves/pca_interp/log_token_gemma_cyber.txt

echo "[$(date)] Exp 3: Forget seq interp Gemma Bio..."
python3 scripts/pca_forget_seq_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Gemma-3-1B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_FORGET_SEQ_GEMMA_BIO 2>&1 | tee saves/pca_interp/log_forget_seq_gemma_bio.txt

echo "[$(date)] Exp 3: Forget seq interp Gemma Cyber..."
python3 scripts/pca_forget_seq_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Gemma-3-1B trainer=RepSelect \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_FORGET_SEQ_GEMMA_CYBER 2>&1 | tee saves/pca_interp/log_forget_seq_gemma_cyber.txt

echo "[$(date)] GPU 3 rerun ALL DONE"
