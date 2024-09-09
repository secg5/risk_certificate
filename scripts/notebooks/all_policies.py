# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: risk_certificates
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('/usr0/home/naveenr/projects/risk_certificate')

import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import argparse
import secrets
from certificate.run_simulations import run_experiments, delete_duplicate_results
import json 

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    trials = 25
    n_arms = 10
    max_pulls_per_arm = 500
    first_stage_pulls_per_arm = 5
    arm_distribution = 'bimodal_diff'
    out_folder = "baseline_comparison"
    arm_parameters=  {'alpha': 2, 'beta': 2, 'diff_mean_1': 0.05, 'diff_std_1': 0.05,'diff_mean_2': 0.01, 'diff_std_2': 0.001}
    delta = 0.1
    run_all_k = False
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--trials', help='Trials', type=int, default=25)
    parser.add_argument('--n_arms',         '-N', help='Number of arms', type=int, default=10)
    parser.add_argument('--max_pulls_per_arm',        help='Maximum pulls per arm', type=int, default=10)
    parser.add_argument('--first_stage_pulls_per_arm',          help='Number of first stage pulls ', type=int, default=4)
    parser.add_argument('--arm_distribution',          help='Distribution of arms', type=str, default='uniform')
    parser.add_argument('--run_all_k',        help='Maximum pulls per arm', action='store_true')
    parser.add_argument('--delta',        help='Maximum pulls per arm', type=float, default=0.1)
    parser.add_argument('--alpha',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--beta',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--diff_mean_1',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--diff_std_1',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--diff_mean_2',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--diff_std_2',        help='Maximum pulls per arm', type=float, default=2)
    parser.add_argument('--out_folder', help='Which folder to write results to', type=str, default='policy_comparison')

    args = parser.parse_args()

    seed = args.seed
    n_arms = args.n_arms
    max_pulls_per_arm = args.max_pulls_per_arm 
    first_stage_pulls_per_arm = args.first_stage_pulls_per_arm
    arm_distribution = args.arm_distribution
    out_folder = args.out_folder
    delta = args.delta 
    alpha = args.alpha 
    beta = args.beta 
    trials = args.trials 
    diff_mean_1 = args.diff_mean_1 
    diff_std_1 = args.diff_std_1 
    diff_mean_2 = args.diff_mean_2 
    diff_std_2 = args.diff_std_2
    arm_parameters = {'alpha': alpha, 'beta': beta, 'diff_mean_1': diff_mean_1, 'diff_mean_2': diff_mean_2, 'diff_std_1': diff_std_1, 'diff_std_2': diff_std_2}
    run_all_k = args.run_all_k

save_name = secrets.token_hex(4)  
# -

random.seed(seed)
np.random.seed(seed)

arm_means = []
for i in range(n_arms):
    if arm_distribution == 'uniform':
        arm_means.append(random.random())
    elif arm_distribution == 'beta':
        arm_means.append(np.random.beta(arm_parameters['alpha'],arm_parameters['beta']))
if arm_distribution == 'unimodal_diff':
    arm_means.append(np.random.random())    
    for i in range(1,n_arms):
        diff = np.random.normal(arm_parameters['diff_mean_1'],arm_parameters['diff_std_1']) 
        arm_means.append(min(max(arm_means[-1]-diff,0.0001),1))
if arm_distribution == 'bimodal_diff':
    arm_means.append(np.random.random())    
    for i in range(1,n_arms):
        if np.random.random() < 0.5:
            diff = np.random.normal(arm_parameters['diff_mean_1'],arm_parameters['diff_std_1']) 
        else:
            diff = np.random.normal(arm_parameters['diff_mean_2'],arm_parameters['diff_std_2']) 
        arm_means.append(min(max(arm_means[-1]-diff,0.0001),1))

experiment_config = {
    'number_arms': n_arms, 
    'sample_size': max_pulls_per_arm*n_arms, 
    'first_stage_size': first_stage_pulls_per_arm*n_arms, 
    'distribution': arm_means, 
    'arm_distribution': arm_distribution, 
    'random_seed': seed+1, 
    'delta': delta,
    'run_all_k': run_all_k, 
    'reward_parameters': arm_parameters
}

# +
all_results = []

for i in range(trials):
    experiment_config['random_seed'] = seed+i
    results = run_experiments(experiment_config)
    all_results.append(results)

# +
aggregate_results = {}
aggregate_results['parameters'] = experiment_config
aggregate_results['parameters']['seed'] = seed 

for method in all_results[0]:
    aggregate_results[method] = {}
    aggregate_results[method]['certificate'] = [max(i[method]['certificate']) for i in all_results]
    aggregate_results[method]['true_value'] = all_results[0][method]['true_value']
# -

# ## Write Data

save_path = "{}/{}.json".format(out_folder,save_name)

delete_duplicate_results(out_folder,"",aggregate_results)

json.dump(aggregate_results,open('../../results/'+save_path,'w'))


