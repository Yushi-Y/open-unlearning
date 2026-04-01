#!/bin/bash
# GPU 2: Qwen3-8B-Base experiments (Exp 1, 2, 3, 4+6)
set -e
cd /VData/kebl6672/open-unlearning
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=disabled

echo "=========================================="
echo "GPU 2: Qwen3-8B-Base experiments"
echo "=========================================="

# --- Exp 1: Token interp (Bio + Cyber) ---
echo "[$(date)] Exp 1: Token interp Qwen Bio..."
python scripts/pca_token_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    ~trainer.method_args.cfg.lora_rank \
    task_name=PCA_TOKEN_INTERP_QWEN_BIO 2>&1 | tee saves/pca_interp/log_token_qwen_bio.txt

echo "[$(date)] Exp 1: Token interp Qwen Cyber..."
python scripts/pca_token_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    ~trainer.method_args.cfg.lora_rank \
    task_name=PCA_TOKEN_INTERP_QWEN_CYBER 2>&1 | tee saves/pca_interp/log_token_qwen_cyber.txt

# --- Exp 2: Sequence interp (Bio + Cyber) ---
echo "[$(date)] Exp 2: Sequence interp Qwen Bio..."
python scripts/pca_sequence_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    ~trainer.method_args.cfg.lora_rank \
    task_name=PCA_SEQ_INTERP_QWEN_BIO 2>&1 | tee saves/pca_interp/log_seq_qwen_bio.txt

echo "[$(date)] Exp 2: Sequence interp Qwen Cyber..."
python scripts/pca_sequence_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    ~trainer.method_args.cfg.lora_rank \
    task_name=PCA_SEQ_INTERP_QWEN_CYBER 2>&1 | tee saves/pca_interp/log_seq_qwen_cyber.txt

# --- Exp 3: Forget sequence interp (Bio + Cyber) ---
echo "[$(date)] Exp 3: Forget seq interp Qwen Bio..."
python scripts/pca_forget_seq_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    ~trainer.method_args.cfg.lora_rank \
    task_name=PCA_FORGET_SEQ_QWEN_BIO 2>&1 | tee saves/pca_interp/log_forget_seq_qwen_bio.txt

echo "[$(date)] Exp 3: Forget seq interp Qwen Cyber..."
python scripts/pca_forget_seq_interp.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    ~trainer.method_args.cfg.lora_rank \
    task_name=PCA_FORGET_SEQ_QWEN_CYBER 2>&1 | tee saves/pca_interp/log_forget_seq_qwen_cyber.txt

# --- Exp 4+6: Attack subspace + LoRA depth (Bio + Cyber) ---
echo "[$(date)] Exp 4+6: Attack subspace Qwen Bio..."
python scripts/attack_subspace.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    task_name=ATTACK_SUB_QWEN_BIO 2>&1 | tee saves/pca_interp/log_attack_qwen_bio.txt

echo "[$(date)] Exp 4+6: Attack subspace Qwen Cyber..."
python scripts/attack_subspace.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default \
    model=Qwen3-8B-Base trainer=RepCollapse \
    wmdp_domain=cyber \
    trainer.args.num_train_epochs=1 \
    trainer.args.eval_strategy=no \
    trainer.args.report_to=none \
    trainer.args.per_device_train_batch_size=4 \
    task_name=ATTACK_SUB_QWEN_CYBER 2>&1 | tee saves/pca_interp/log_attack_qwen_cyber.txt

echo "[$(date)] GPU 2 (Qwen) ALL DONE"
