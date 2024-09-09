#!/bin/bash 

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/risk_certificate/scripts/notebooks" ENTER

    for start_seed in 42 45 48 51 54
    do 
        seed=$((${session}+${start_seed}))
        echo ${seed}

        trials=25
        arm_distribution=uniform
        out_folder=vary_n_m
        delta=0.1

        for n_arms in 5 10 25
        do 
            max_pulls_per_arm=10
            for first_stage_pulls_per_arm in 1 2 4 8
            do 
                tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
            done 

            max_pulls_per_arm=50
            for first_stage_pulls_per_arm in 5 10 25 40 45
            do 
                tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
            done 

            max_pulls_per_arm=100
            for first_stage_pulls_per_arm in 5 10 25 50 90
            do 
                tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
            done 

            max_pulls_per_arm=500
            for first_stage_pulls_per_arm in 5 10 25 50 100 250 450
            do 
                tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
            done 
        done 
    done 
done 


# 1. Compare amongst k
# 2. Compare to UCB
# 3. Vary N, M
# 4. Vary Gaps 
# 5. Use Prior 