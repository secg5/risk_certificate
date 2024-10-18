#!/bin/bash 

cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    trials=100
    arm_distribution=beta
    out_folder=prior_data
    n_arms=10
    delta=0.1
    max_pulls_per_arm=50
    first_stage_pulls_per_arm=25

    # Regular Beta Distribution
    for alpha in 1 2 4 8 16 32
    do 
        beta=${alpha}
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta}  --run_all_k
    done 

    beta=1
    for alpha in 1 2 4 8
    do 
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta}  --run_all_k
    done 

    alpha=1
    for beta in 1 2 4 8
    do 
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta}  --run_all_k
    done 

    for alpha in 1 2 4 8 16 32
    do 
        frac=0.5 
        beta=$(echo "${alpha}*${frac}" | bc)
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta}  --run_all_k
    done 

    for alpha in 1 2 4 8 16 32
    do 
        frac=0.25
        beta=$(echo "${alpha}*${frac}" | bc)
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta}  --run_all_k
    done 

    # Beta Misspecified 

    arm_distribution=beta_misspecified
    for diff_mean_1 in -0.2 -0.1 -0.05 0.05 0.1 0.2
    do 
        alpha=2
        beta=2
        diff_std_1=0.1
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta} --diff_mean_1 ${diff_mean_1} --diff_std_1 ${diff_std_1}  --run_all_k
    done 

    for diff_std_1 in 0.01 0.05 0.1 0.25
    do 
        alpha=2
        beta=2
        diff_mean_1=0
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --alpha ${alpha} --beta ${beta} --diff_mean_1 ${diff_mean_1} --diff_std_1 ${diff_std_1}  --run_all_k
    done 

    # Real world data
    arm_distribution=effect_size
    for delta in 0.1 0.01 0.001 0.0001
    do 
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm}  --run_all_k
    done 
done 