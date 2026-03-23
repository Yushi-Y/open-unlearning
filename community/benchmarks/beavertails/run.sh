#!/bin/bash

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml

common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated"
reference="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated trainer.args.num_train_epochs=0"

# Auto-detect if we're on SLURM
if command -v sbatch &> /dev/null; then
    echo "Running on SLURM"
    common="sbatch runners/slurm_runner.sh $common"
    reference="sbatch runners/slurm_runner.sh $reference"
fi

# model=gemma-3-270m
# model=gemma-3-4b-pt
model=gemma-2-2b

category='animal_abuse'
# category='terrorism,organized_crime'

version=v5
# "no version" used the original beavertails dataset, where there is data duplication and mislabeling; also, it wasn't finished, got terminated before 50 trials
# v2 uses our curated high-quality subset
# v3 uses a 2x smaller LR during relearning, because it was too severe
# v4 tunes probability, not loss; also npo saturation was removed from repselect
# v5 evel smaller relearning LR

${reference} model=${model} category=${category} trainer=GradDiff task_name=${version}_${model}_${category}_reference
${common} model=${model} category=${category} trainer=RepSelect task_name=${version}_${model}_${category}_RepSelect
${common} model=${model} category=${category} trainer=GradDiff task_name=${version}_${model}_${category}_GradDiff2
${common} model=${model} category=${category} trainer=NPO task_name=${version}_${model}_${category}_NPO
${common} model=${model} category=${category} trainer=RMU task_name=${version}_${model}_${category}_RMU2
${common} model=${model} category=${category} trainer=SimNPO task_name=${version}_${model}_${category}_SimNPO
${common} model=${model} category=${category} trainer=UNDIAL task_name=${version}_${model}_${category}_UNDIAL2

# RepSelect ablations (all use wide LR range for fair comparison)
${common} model=${model} category=${category} trainer=RepSelect hydra/sweeper=RepSelect_wide task_name=${version}_${model}_${category}_RepSelect_wide
${common} model=${model} category=${category} trainer=RepSelect hydra/sweeper=RepSelect_no_lora '~trainer.method_args.cfg.lora_lr' task_name=${version}_${model}_${category}_RepSelect_no_lora
${common} model=${model} category=${category} trainer=RepSelect hydra/sweeper=RepSelect_no_retain '~trainer.method_args.cfg.retain_momentum' task_name=${version}_${model}_${category}_RepSelect_no_retain
${common} model=${model} category=${category} trainer=RepSelect hydra/sweeper=RepSelect_no_pcs '~trainer.method_args.cfg.n_pcs' task_name=${version}_${model}_${category}_RepSelect_no_pcs
