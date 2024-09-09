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
        arm_distribution=beta
        out_folder=prior_data
        n_arms=10
        delta=0.1

        for alpha in 1 2 5
        do 
            for beta in 1 2 5 
            do 
                max_pulls_per_arm=20
                first_stage_pulls_per_arm=10
                tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta} --run_all_k" ENTER

                max_pulls_per_arm=50
                first_stage_pulls_per_arm=25
                tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta}  --run_all_k" ENTER

                max_pulls_per_arm=100
                first_stage_pulls_per_arm=50
                tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta}  --run_all_k" ENTER
            done 
        done 

    done 
done 


# 1. Compare amongst k
# 2. Compare to UCB
# 3. Vary N, M
# 4. Vary Gaps 
# 5. Use Prior 