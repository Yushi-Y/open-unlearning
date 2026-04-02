#!/bin/bash
# Usage: ./run_disco_exp.sh <gpu_id> <task_name> [extra hydra overrides...]
# Example: ./run_disco_exp.sh 0 disco_lr01 trainer.args.learning_rate=0.01

GPU=$1
TASK=$2
shift 2

CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n unlearning \
  python3 src/unlearn_relearn.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/wmdp_low_mi/default \
  model=Llama-3.2-3B \
  trainer=DISCO \
  task_name=$TASK \
  trainer.args.report_to=none \
  "$@" \
  2>&1 | tee /tmp/${TASK}.log
