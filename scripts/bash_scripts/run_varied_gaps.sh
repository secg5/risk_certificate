#!/bin/bash 

for seed in $(seq 43 57); 
do 
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
    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --uniform_low ${uniform_low} --uniform_high ${uniform_high} --run_all_k

    uniform_low=0.5
    uniform_high=1
    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --uniform_low ${uniform_low} --uniform_high ${uniform_high} --run_all_k

    uniform_low=0.75
    uniform_high=1
    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --uniform_low ${uniform_low} --uniform_high ${uniform_high} --run_all_k

    uniform_low=0.9
    uniform_high=1
    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --uniform_low ${uniform_low} --uniform_high ${uniform_high} --run_all_k

    uniform_low=0
    uniform_high=0.1
    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --uniform_low ${uniform_low} --uniform_high ${uniform_high} --run_all_k

    uniform_low=0
    uniform_high=0.25
    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --uniform_low ${uniform_low} --uniform_high ${uniform_high} --run_all_k

    uniform_low=0
    uniform_high=0.5
    python all_policies.py --seed ${seed} --out_folder ${out_folder} --trials ${trials} --arm_distribution ${arm_distribution} --delta ${delta} --n_arms ${n_arms} --max_pulls_per_arm ${max_pulls_per_arm} --first_stage_pulls_per_arm ${first_stage_pulls_per_arm} --uniform_low ${uniform_low} --uniform_high ${uniform_high} --run_all_k
done 