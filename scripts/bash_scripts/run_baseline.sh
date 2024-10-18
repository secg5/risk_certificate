#!/bin/bash 

cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    trials=100
    arm_distribution=uniform
    out_folder=baseline
    delta=0.1
    max_pulls_per_arm=50

    for first_stage_pulls_per_arm in 5 10 15 20 25 30 35 40 45 
    do 
        for n_arms in 5 10
        do
            python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k
        done 

        for n_arms in 20 50
        do
            python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k
        done 
    done 

    n_arms=10
    for max_pulls_per_arm in 10 20 30 50 100 200 400
    do 
        frac=0.5 
        first_stage_pulls_per_arm=$(echo "${max_pulls_per_arm}*${frac}" | bc)
        first_stage_pulls_per_arm=$(printf "%.0f" $first_stage_pulls_per_arm)
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k
    done 

    for max_pulls_per_arm in 10 20 50 100 200 400
    do 
        for frac in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
        do 
            first_stage_pulls_per_arm=$(echo "${max_pulls_per_arm}*${frac}" | bc)
            first_stage_pulls_per_arm=$(printf "%.0f" $first_stage_pulls_per_arm)
            python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k
        done 
    done 
done 