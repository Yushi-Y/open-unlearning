#!/bin/bash
# Post-training evaluation of MMLU + HellaSwag on saved models
# Run after sweep finishes and best configs are identified
# Usage: bash scripts/run_posthoc_eval.sh <model_path> <task_name> [gpu_id]
#
# Example:
#   bash scripts/run_posthoc_eval.sh saves/unlearn/best_RepSelect_bio best_RepSelect_bio 0

set -e
cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export WANDB_MODE=disabled

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

MODEL_PATH=${1:?Usage: $0 <model_path> <task_name> [gpu_id]}
TASK_NAME=${2:?Usage: $0 <model_path> <task_name> [gpu_id]}
GPU_ID=${3:-0}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python src/eval.py \
    experiment=eval/lm_eval_general \
    model=Llama-3.2-3B \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    task_name=${TASK_NAME} \
    paths.output_dir=saves/eval/${TASK_NAME}
