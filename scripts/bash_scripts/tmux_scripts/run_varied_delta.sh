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
        out_folder=vary_delta
        n_arms=10
        max_pulls_per_arm=50
        first_stage_pulls_per_arm=25

        for delta in 0.1 0.05 0.01 0.005 0.0001
        do 
            tmux send-keys -t match_${session} "conda activate ${conda_env}; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
        done 

    done 
done 