#!/bin/bash
# GPU 0: Llama-3.2-3B experiments (Exp 1, 2, 3, 4+6, 5)
set +e  # don't exit on error, keep running subsequent experiments
cd /VData/kebl6672/open-unlearning
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
source /VData/kebl6672/open-unlearning/.env
export HF_HOME HF_TOKEN
eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

echo "=========================================="
echo "GPU 0: Llama-3.2-3B experiments"
echo "=========================================="

# --- Exp 1: Token interp (Bio + Cyber) ---
echo "[$(date)] Exp 1: Token interp Llama Bio..."
python3 scripts/pca_token_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_TOKEN_INTERP_LLAMA_BIO 2>&1 | tee saves/pca_interp/log_token_llama_bio.txt

echo "[$(date)] Exp 1: Token interp Llama Cyber..."
python3 scripts/pca_token_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_TOKEN_INTERP_LLAMA_CYBER 2>&1 | tee saves/pca_interp/log_token_llama_cyber.txt

# --- Exp 2: Sequence interp (Bio + Cyber) ---
echo "[$(date)] Exp 2: Sequence interp Llama Bio..."
python3 scripts/pca_sequence_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_SEQ_INTERP_LLAMA_BIO 2>&1 | tee saves/pca_interp/log_seq_llama_bio.txt

echo "[$(date)] Exp 2: Sequence interp Llama Cyber..."
python3 scripts/pca_sequence_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_SEQ_INTERP_LLAMA_CYBER 2>&1 | tee saves/pca_interp/log_seq_llama_cyber.txt

# --- Exp 3: Forget sequence interp (Bio + Cyber) ---
echo "[$(date)] Exp 3: Forget seq interp Llama Bio..."
python3 scripts/pca_forget_seq_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_FORGET_SEQ_LLAMA_BIO 2>&1 | tee saves/pca_interp/log_forget_seq_llama_bio.txt

echo "[$(date)] Exp 3: Forget seq interp Llama Cyber..."
python3 scripts/pca_forget_seq_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B trainer=RepSelect \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
    task_name=PCA_FORGET_SEQ_LLAMA_CYBER 2>&1 | tee saves/pca_interp/log_forget_seq_llama_cyber.txt

# --- Exp 4+6: Attack subspace + LoRA depth (Bio + Cyber) ---
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

# --- Exp 5: Baseline vs RepCollapse (Llama Bio) ---
echo "[$(date)] Exp 5: Baseline comparison Llama Bio..."
python3 scripts/baseline_weight_projection.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Llama-3.2-3B \
    trainer=RepSelect \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    task_name=BASELINE_CMP_LLAMA_BIO 2>&1 | tee saves/pca_interp/log_baseline_llama_bio.txt

echo "[$(date)] GPU 0 (Llama) ALL DONE"
