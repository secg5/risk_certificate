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
        out_folder=vary_gaps
        delta=0.1

        for n_arms in 5 10
        do 
            for max_pulls_per_arm in 20 50 100
            do 
                frac=0.5 
                first_stage_pulls_per_arm=$(echo "${max_pulls_per_arm}*${frac}" | bc)
                first_stage_pulls_per_arm=$(printf "%.0f" $first_stage_pulls_per_arm)
            
                arm_distribution=uniform
                tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
            
                arm_distribution=unimodal_diff
                diff_std_1=0.01
                for diff_mean_1 in 0.01 0.02 0.05
                do 
                    tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --diff_mean_1 ${diff_mean_1} --diff_std_1 ${diff_std_1} --run_all_k" ENTER
                done 

                arm_distribution=bimodal_diff
                diff_std_1=0.01
                diff_std_2=0.01
                for diff_mean_1 in 0.005 0.01 0.02 0.05 0.1
                do 
                    for diff_mean_2 in 0.005 0.01 0.02 0.05 0.1
                    do 
                        if awk -v low="$diff_mean_1" -v high="$diff_mean_2" 'BEGIN { exit !(high > low) }'; then
                            tmux send-keys -t match_${session} "conda activate feedback; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --diff_mean_1 ${diff_mean_1} --diff_std_1 ${diff_std_1} --diff_mean_2 ${diff_mean_2}  --diff_std_2 ${diff_std_2} --run_all_k" ENTER
                        fi
                    done 
                done 


            done 
        done 

    done 
done 


# 1. Compare amongst k
# 2. Compare to UCB
# 3. Vary N, M
# 4. Vary Gaps 
# 5. Use Prior 