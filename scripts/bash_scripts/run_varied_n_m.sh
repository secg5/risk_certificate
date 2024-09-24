#!/bin/bash 

for session in 1
do
    # Simulating the 'tmux new-session' behavior without tmux
    echo "Session: match_${session}"
    cd ~/Documents/risk_certificate/scripts/notebooks

    for start_seed in 42
    do 
        seed=$((${session}+${start_seed}))
        echo "Seed: ${seed}"

        # trials=25
        # ["bimodal_best", "bimodal_better", "bimodal_normal", "bimodal_worse"]
        trials=10
        arm_distribution=bimodal_zero
        out_folder=vary_n_m
        
        for delta in 0.1
        do
            for n_arms in 200
            do 
                max_pulls_per_arm=10
                for first_stage_pulls_per_arm in 1 2 4 5 9
                do 
                    echo "Running: conda activate risk_certificates; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k"
                    # conda init
                    # conda activate risk_certificates
                    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k
                done 

                # Uncomment to run additional configurations
                max_pulls_per_arm=50
                for first_stage_pulls_per_arm in 5 10 20 25 45
                do 
                    echo "Running: python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k"
                    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k
                done 

                max_pulls_per_arm=100
                for first_stage_pulls_per_arm in 10 20 40 50 90
                do 
                    echo "Running: python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k"
                    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k
                done 
                
                max_pulls_per_arm=500
                for first_stage_pulls_per_arm in 50 100 200 250 450
                do 
                    echo "Running: python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k"
                    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k
                done 
            done 
        done
    done 
done

echo "All sessions completed."

# Debugging steps:
# 1. Compare amongst k
# 2. Compare to UCB
# 3. Vary N, M
# 4. Vary Gaps 
# 5. Use Prior

# #!/bin/bash 

# for session in 1 2 3
# do
#     tmux new-session -d -s match_${session}
#     tmux send-keys -t match_${session} ENTER 
#     tmux send-keys -t match_${session} "cd ~/Documents/risk_certificate/scripts/notebooks" ENTER

#     for start_seed in 42 45 48 51 54
#     do 
#         seed=$((${session}+${start_seed}))
#         echo ${seed}

#         # trials=25
#         trials=10
#         arm_distribution=bimodal
#         out_folder=vary_n_m
#         delta=0.1

#         for n_arms in 200
#         do 
#             max_pulls_per_arm=10
#             for first_stage_pulls_per_arm in 1 2 4 8
#             do 
#                 tmux send-keys -t match_${session} "conda activate risk_certificates; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
#             done 

#             # max_pulls_per_arm=50
#             # for first_stage_pulls_per_arm in 5 10 25 40 45
#             # do 
#             #     tmux send-keys -t match_${session} "conda activate risk_certificates; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
#             # done 

#             # max_pulls_per_arm=100
#             # for first_stage_pulls_per_arm in 5 10 25 50 90
#             # do 
#             #     tmux send-keys -t match_${session} "conda activate risk_certificates; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
#             # done 

#             # max_pulls_per_arm=500
#             # for first_stage_pulls_per_arm in 5 10 25 50 100 250 450
#             # do 
#             #     tmux send-keys -t match_${session} "conda activate risk_certificates; python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k" ENTER
#             # done 
#         done 
#     done 
# done 


# # 1. Compare amongst k
# # 2. Compare to UCB
# # 3. Vary N, M
# # 4. Vary Gaps 
# # 5. Use Prior 