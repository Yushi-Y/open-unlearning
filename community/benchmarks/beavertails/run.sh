#!/bin/bash

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml

common="sbatch runners/slurm_runner.sh python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/default"
reference="sbatch runners/slurm_runner.sh python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/default trainer.args.num_train_epochs=0"

# model=gemma-3-270m
model=gemma-3-4b-pt
# model=gemma-2-2b

${reference} model=${model} trainer=GradDiff task_name=${model}_reference
${common} model=${model} trainer=RepSelect task_name=${model}_RepSelect
${common} model=${model} trainer=GradDiff task_name=${model}_GradDiff
${common} model=${model} trainer=NPO task_name=${model}_NPO
${common} model=${model} trainer=RMU task_name=${model}_RMU
${common} model=${model} trainer=SimNPO task_name=${model}_SimNPO
${common} model=${model} trainer=UNDIAL task_name=${model}_UNDIAL

# later add category='terrorism,organized_crime'


# common="python3 src/unlearn_relearn.py \
# --multirun \
# --config-name=unlearn.yaml \
# experiment=unlearn/wmdp_low_mi/default"

# ver=v4


# # Auto-detect if we're on SLURM
# if command -v sbatch &> /dev/null; then
#     echo "Running on SLURM"
#     common="sbatch $HOME/open-unlearning/runners/slurm_runner.sh $common"
# fi


# $common model=Llama-3.2-3B trainer=GradDiff task_name=${ver}_3B_GradDiff_bio
# $common model=Llama-3.2-3B trainer=NPO task_name=${ver}_3B_NPO_bio
# $common model=Llama-3.2-3B trainer=RMU task_name=${ver}_3B_RMU_bio
# $common model=Llama-3.2-3B trainer=SimNPO task_name=${ver}_3B_SimNPO_bio
# $common model=Llama-3.2-3B trainer=UNDIAL task_name=${ver}_3B_UNDIAL_bio
# $common model=Llama-3.2-3B trainer=RepSelect task_name=${ver}_3B_RepSelect_bio
# $common model=Llama-3.2-3B trainer=RepSelect task_name=${ver}_3B_RepSelectstrict_bio eval.wikitext_kl.disr_budget=0.005
# $common model=Llama-3.2-3B trainer=NPO task_name=${ver}_3B_NPOstrict_bio eval.wikitext_kl.disr_budget=0.005

# $common model=Llama-3.2-3B trainer=GradDiff task_name=${ver}_3B_GradDiff_cyber wmdp_domain=cyber
# $common model=Llama-3.2-3B trainer=NPO task_name=${ver}_3B_NPO_cyber wmdp_domain=cyber
# $common model=Llama-3.2-3B trainer=RMU task_name=${ver}_3B_RMU_cyber wmdp_domain=cyber
# $common model=Llama-3.2-3B trainer=SimNPO task_name=${ver}_3B_SimNPO_cyber wmdp_domain=cyber
# $common model=Llama-3.2-3B trainer=UNDIAL task_name=${ver}_3B_UNDIAL_cyber wmdp_domain=cyber
# $common model=Llama-3.2-3B trainer=RepSelect task_name=${ver}_3B_RepSelect_cyber wmdp_domain=cyber
# $common model=Llama-3.2-3B trainer=RepSelect task_name=${ver}_3B_RepSelectstrict_cyber eval.wikitext_kl.disr_budget=0.005 wmdp_domain=cyber
# $common model=Llama-3.2-3B trainer=NPO task_name=${ver}_3B_NPOstrict_cyber eval.wikitext_kl.disr_budget=0.005 wmdp_domain=cyber
