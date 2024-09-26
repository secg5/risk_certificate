#!/bin/bash 

project_dir=~/projects/risk_certificate
conda_env=feedback

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ${project_dir}/scripts/notebooks" ENTER

    for start_seed in 42 45 48 51 54
    do 
        seed=$((${session}+${start_seed}))
        echo ${seed}

        trials=100
        arm_distribution=uniform
        out_folder=vary_gaps
        delta=0.1
        max_pulls_per_arm=50
        n_arms=10
        arm_distribution=uniform
        first_stage_pulls_per_arm=25

        uniform_low=0
        uniform_high=1
        tmux send-keys -t match_${session} "conda activate ${conda_env}; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --uniform_low ${uniform_low} --uniform_high ${uniform_high} --run_all_k" ENTER

        uniform_low=0.5
        uniform_high=1
        tmux send-keys -t match_${session} "conda activate ${conda_env}; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER

        uniform_low=0.75
        uniform_high=1
        tmux send-keys -t match_${session} "conda activate ${conda_env}; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER

        uniform_low=0.9
        uniform_high=1
        tmux send-keys -t match_${session} "conda activate ${conda_env}; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER

        uniform_low=0
        uniform_high=0.1
        tmux send-keys -t match_${session} "conda activate ${conda_env}; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER

        uniform_low=0
        uniform_high=0.25
        tmux send-keys -t match_${session} "conda activate ${conda_env}; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER

        uniform_low=0
        uniform_high=0.5
        tmux send-keys -t match_${session} "conda activate ${conda_env}; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
    done 
done 