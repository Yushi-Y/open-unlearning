#!/bin/bash

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml

common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default"
reference="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer.args.num_train_epochs=0"

# Auto-detect if we're on SLURM
if command -v sbatch &> /dev/null; then
    echo "Running on SLURM"
    common="sbatch runners/slurm_runner.sh $common"
    reference="sbatch runners/slurm_runner.sh $reference"
fi

# model=Llama-3.2-3B
model=Qwen2.5-3B

wmdp_domain='bio'
# wmdp_domain='cyber'

version=v1

${reference} model=${model} wmdp_domain=${wmdp_domain} trainer=GradDiff task_name=${version}_${model}_${wmdp_domain}_reference
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=RepSelect task_name=${version}_${model}_${wmdp_domain}_RepSelect
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=GradDiff task_name=${version}_${model}_${wmdp_domain}_GradDiff
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=NPO task_name=${version}_${model}_${wmdp_domain}_NPO
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=RMU task_name=${version}_${model}_${wmdp_domain}_RMU
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=SimNPO task_name=${version}_${model}_${wmdp_domain}_SimNPO
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=UNDIAL task_name=${version}_${model}_${wmdp_domain}_UNDIAL

# RepSelect ablations (all use wide LR range for fair comparison)
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=RepSelect hydra/sweeper=RepSelect_wide task_name=${version}_${model}_${wmdp_domain}_RepSelect_wide
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=RepSelect hydra/sweeper=RepSelect_no_lora '~trainer.method_args.cfg.lora_lr' task_name=${version}_${model}_${wmdp_domain}_RepSelect_no_lora
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=RepSelect hydra/sweeper=RepSelect_no_retain '~trainer.method_args.cfg.retain_momentum' task_name=${version}_${model}_${wmdp_domain}_RepSelect_no_retain
${common} model=${model} wmdp_domain=${wmdp_domain} trainer=RepSelect hydra/sweeper=RepSelect_no_pcs '~trainer.method_args.cfg.n_pcs' task_name=${version}_${model}_${wmdp_domain}_RepSelect_no_pcs
