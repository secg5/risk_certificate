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

# +
import sys
sys.path.append('/Users/scortesg/Documents/risk_certificate')
# sys.path.append('/usr0/home/naveenr/projects/risk_certificate')
# -

import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import argparse
import secrets
from certificate.run_simulations import run_experiments, delete_duplicate_results
import json 
import sys
import os

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    trials = 100
    n_arms = 10
    max_pulls_per_arm = 50
    first_stage_pulls_per_arm = 25
    arm_distribution = 'beta_misspecified'
    out_folder = "prior_data"
    arm_parameters=  {'alpha': 50, 'beta': 50, 'diff_mean_1': 0.05, 'diff_std_1': 0.01,'diff_mean_2': 0.01, 'diff_std_2': 0.001}
    delta = 0.1
    run_all_k = True
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
    elif arm_distribution == 'beta_misspecified':
        arm_means.append(np.clip(np.random.beta(arm_parameters['alpha'],arm_parameters['beta']) + np.random.normal(arm_parameters['diff_mean_1'],arm_parameters['diff_std_1']),0,1))
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
    # import pdb;pdb.set_trace()


def bimodal_arms(n_arms,num_high_means):
    # Generate means for one-fourth of the arms (0.8 to 0.9)
    
    high_means = [random.uniform(0.9, 0.99) for _ in range(num_high_means)]
    # Generate means for three-fourths of the arms (0.1 to 1.5)
    num_low_means = n_arms - num_high_means
    low_means = [random.uniform(0.0001, 0.1) for _ in range(num_low_means)]
    # Combine the two lists to form arm_means
    arm_means = high_means + low_means
    # Shuffle the arm_means to mix the values
    random.shuffle(arm_means)
    return arm_means

if arm_distribution == "bimodal_best":
    num_high_means = n_arms // 20
    arm_means = bimodal_arms(n_arms, num_high_means)

if arm_distribution == "bimodal_better":
    num_high_means = n_arms // 10
    arm_means = bimodal_arms(n_arms, num_high_means)
if arm_distribution == "bimodal_normal":
    num_high_means = n_arms // 2
    arm_means = bimodal_arms(n_arms, num_high_means)
if arm_distribution == "bimodal_worse":
    num_high_means = n_arms
    arm_means = bimodal_arms(n_arms, num_high_means)
if arm_distribution == "bimodal_perfect":
    num_high_means = n_arms // 20
    arm_means = bimodal_arms(n_arms, num_high_means)
    arm_means[42] = 1
    arm_means[43] = 1
    arm_means[44] = 1



# if arm_distribution =="good_mu":


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
    aggregate_results[method]['delta'] = [i[method]['delta'].tolist() for i in all_results]
    aggregate_results[method]['true_value'] = all_results[0][method]['true_value']

# -

np.mean(aggregate_results['sample_split_total']['certificate'])

np.mean(aggregate_results['sample_split']['certificate'])

if 'prior' in aggregate_results:
    print(np.mean(aggregate_results['prior']['certificate'])/np.mean(aggregate_results['sample_split_total']['certificate']))

np.mean(aggregate_results['random']['certificate'])

np.mean(aggregate_results['k_{}'.format(n_arms)]['delta'])

np.mean(aggregate_results['one_stage']['certificate'])

np.mean(aggregate_results['k_{}'.format(1)]['certificate'])

np.mean(aggregate_results['omniscient']['certificate'])

np.mean(aggregate_results['k_{}'.format(n_arms)]['true_value'])-np.mean(aggregate_results['omniscient']['delta'])

# ## Write Data

save_path = "{}/{}.json".format(out_folder,save_name)

delete_duplicate_results(out_folder,"",aggregate_results)

if is_jupyter:
    dirname = os.path.abspath('')
    filename = os.path.join(dirname, '../../results/'+save_path)
else:
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirname, '../../results/'+save_path)
json.dump(aggregate_results,open(filename,'w'))
