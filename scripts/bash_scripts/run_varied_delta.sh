#!/bin/bash 

for seed in $(seq 43 57); 
do 
    echo ${seed}

    trials=100
    arm_distribution=uniform
    out_folder=vary_delta
    n_arms=10
    max_pulls_per_arm=50
    first_stage_pulls_per_arm=25

    for delta in 0.1 0.05 0.01 0.005 0.0001
    do 
        python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --run_all_k
    done 
done 