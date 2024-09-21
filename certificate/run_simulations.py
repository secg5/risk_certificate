import json
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import glob 
import os
from copy import deepcopy 

from certificate.config import Config


def sample_bernoulli(N, K, mu, indices=None):
    """
    Generate a sample of N from K different Bernoulli distributions with given probabilities.
    
    Parameters:
    N (int): Number of samples to draw from each distribution.
    K (int): Number of different Bernoulli distributions.
    p (list or np.array): Probabilities for each of the K distributions.
    
    Returns:
    np.array: A K x N array where each row contains N samples from the corresponding Bernoulli distribution.
    """
    # if len(mu) != K:
    #     raise ValueError("Length of p must be equal to K")
    if indices is None:
        indices = range(K)
    # uniform allocation
    # import pdb; pdb.set_trace()
    samples = np.random.binomial(1, mu[indices, np.newaxis], (K, N//K))
    
    return samples

def run_algorithm(first_stage, K, n, m, mu):
    """

    Args:
        first_stage (_type_): 
        K (_type_): 
        n (_type_): 
        m (_type_): 

    Returns:
        _type_: 
    """
    # Is this propperly implemented? Should i use the second stage instead?

    sample_number = n // K
    train_data = first_stage[:,:(sample_number // 2)]
    test_data = first_stage[:,(sample_number // 2):sample_number]
    train_means = train_data.mean(axis=1)
    train_means_indices = train_means.argsort()[::-1]

    test_means = test_data.mean(axis=1)

    srt_test_means = np.take_along_axis(test_means, train_means_indices, axis=0)
    delta = np.sqrt((np.arange(K)+1)/(2*m))

    values = []
    for i in range(K):
        value = np.max(srt_test_means[:(i+1)]) - delta[i]
        values.append(value)
    values = np.array(values)
    
    best_estimate = np.argmax(values)
    # import pdb; pdb.set_trace()
    
    top_k_split = best_estimate + 1

    # First not dominant arm, why here? as an alternative to top-k...
    # Is not clear how to use this in the original two satge approach
    # The band used in the two stage approach is dependant on the size of the set picked, 
    # thus adding making it dependant on the size itself?
    # code top-, top-2 top-3

    # import pdb; pdb.set_trace()
    aux = values - np.roll(srt_test_means, -1, axis=0)
    non_negative_mask = aux >= 0
    non_negative_indices = np.where(non_negative_mask)[0]
    dominant_k = non_negative_indices[0] + 1 if non_negative_indices.size > 0 else 1
    
    # import pdb; pdb.set_trace()
    str_mu = np.take_along_axis(mu, train_means_indices, axis=0)
    true_values = str_mu - delta
    # TODO : Think about what the real objective is
    omniscient_k = np.argmax(true_values) + 1
    # import pdb; pdb.set_trace()
    # TODO pick top 1,2 and 3
    # TODO Do the method that is omnicient on the sample splitting
    
    return top_k_split, omniscient_k, dominant_k, train_means_indices

def delete_duplicate_results(folder_name,result_name,data):
    """Delete all results with the same parameters, so it's updated
    
    Arguments:
        folder_name: Name of the results folder to look in
        results_name: What experiment are we running (hyperparameter for e.g.)
        data: Dictionary, with the key parameters
        
    Returns: Nothing
    
    Side Effects: Deletes .json files from folder_name/result_name..."""

    all_results = glob.glob("../../results/{}/{}*.json".format(folder_name,result_name))

    for file_name in all_results:
        try:
            load_file = json.load(open(file_name,"r"))
        except:
            continue 

        if 'parameters' in load_file and load_file['parameters'] == data['parameters']:
            try:
                os.remove(file_name)
            except OSError as e:
                print(f"Error deleting {file_name}: {e}")

def UCB(arm_means, num_arms, total_steps, delta=1e-4):
    ### Choosing the optimal arm
    optimal_arm = np.argmax(arm_means)

    num_iterations = 10 # number of times we perform the same experiment to reduce randomness

    regret = np.zeros([total_steps, num_iterations])
    DELTA = 0.1 
    for iter in range(num_iterations):
        ucb = 100 * np.ones(num_arms)
        emp_means = np.zeros(num_arms)
        num_pulls = np.zeros(num_arms)
        for step_count in range(total_steps):
            greedy_arm = np.argmax(ucb)
            # generate bernoulli reward from the picked greedy arm
            reward = np.random.binomial(1, arm_means[greedy_arm])
            num_pulls[greedy_arm] += 1
            regret[step_count, iter] = arm_means[optimal_arm] - arm_means[greedy_arm]
            emp_means[greedy_arm] += (reward - emp_means[greedy_arm])/num_pulls[greedy_arm]
            # We are setting the exploration to be constant as oposse to decreasing (log(t))
            # ucb[greedy_arm] = emp_means[greedy_arm] + np.sqrt(np.log(1/delta) / (2*num_pulls[greedy_arm]))
            ucb[greedy_arm] = emp_means[greedy_arm] + np.sqrt(np.log(iter + 1) / (num_pulls[greedy_arm]))
    delta = np.sqrt((1/(2*num_pulls))*np.log(2/DELTA))
    # np.sqrt(2*(k/m)*np.log(2/DELTA))
    emp_means -= delta 

    return emp_means , delta 

def successive_elimination(arm_means, num_arms, total_steps, delta=1e-4):
    """
    Successive Elimination algorithm for best arm identification.

    Parameters:
        arm_means (list or np.array): True means of the arms (for simulation).
        num_arms (int): Number of arms.
        total_steps (int): Total number of time steps for pulling arms.
        delta (float): Confidence parameter.

    Returns:
        best_arm (int): The index of the identified best arm.
        arm_pulls (list): Number of pulls for each arm.
    """
    # Initialize counts and empirical means for each arm
    arm_pulls = np.zeros(num_arms)
    empirical_means = np.zeros(num_arms)
    remaining_arms = list(range(num_arms))
    
    step = 0
    while len(remaining_arms) > 1 and step < total_steps:
        for arm in remaining_arms:
            # Pull each remaining arm once
            arm_pulls[arm] += 1
            reward = np.random.binomial(1, arm_means[arm])
            # np.random.normal(arm_means[arm], 1)  # Simulated reward, assuming unit variance
            empirical_means[arm] = ((empirical_means[arm] * (arm_pulls[arm] - 1)) + reward) / arm_pulls[arm]
            step += 1
            if step >= total_steps:
                break
        
        # Update confidence bounds
        confidence_bound = np.sqrt(np.log(2 / delta) / 2* arm_pulls[remaining_arms])
        
        # Calculate upper and lower bounds
        # Here we do need the 2/delta
        upper_bounds = empirical_means[remaining_arms] + confidence_bound
        lower_bounds = empirical_means[remaining_arms] - confidence_bound
        
        # Find the best arm based on current estimates
        best_arm_index = np.argmax(empirical_means[remaining_arms])
        best_arm = remaining_arms[best_arm_index]
        
        # Eliminate arms whose upper bound is worse than the best arm's lower bound
        remaining_arms = [
            arm for i, arm in enumerate(remaining_arms)
            if upper_bounds[i] >= lower_bounds[best_arm_index]
        ]

    return lower_bounds, confidence_bound


def compute_hoeffding_bound(n_data, delta=1e-4):
    return np.sqrt((1/(2*n_data))*np.log(1/delta))


# if __name__ == "__main__":
def run_experiments(config_dict):
    # parser = ArgumentParser(description="Run experiments with JSON configuration.")
    # parser.add_argument("config", help="Path to the JSON configuration file")
    # args = parser.parse_args()
    # with open(args.config, "r") as f:
    #     config_dict = json.load(f)
    # guardar el config
    config = Config(**config_dict)
    np.random.seed(config.random_seed)

    K = config.number_arms  # Number of Bernoulli distributions

    # N = Budget
    N = config.sample_size  # Number of samples to draw from each distribution
    
    mu = config.distribution
    n = config.first_stage_size
    m = N - n
    assert n % K == 0, "The number of samples in the first stage must be a multiple of the number of arms."
    assert m % K == 0, "The number of samples in the second stage must be a multiple of the number of arms."
    assert len(mu) == K , "Number of arms is not consisted with the distribtuion provided."
    

    mu = np.array(mu)
    first_stage = sample_bernoulli(n, K, mu)
    top_k_split, omniscient_k, dominant_k, train_means_indices = run_algorithm(first_stage, K, n, m, mu)
    DELTA = config.delta 

    def second_stage(k, m, mu, train_means_indices):
        B = train_means_indices[:k]
        second_stage = sample_bernoulli(m, k, mu, B)
        delta = compute_hoeffding_bound(m/k, DELTA)
        certificate = second_stage.mean(axis=1) - delta
        return certificate, B, delta

    certificate_split, B_split, delta_split = second_stage(top_k_split, m, mu, train_means_indices)
    # delta_split_total = np.sqrt((1/(2*(m/top_k_split + n/K)))*np.log(2/DELTA))
    delta_split_total = compute_hoeffding_bound(m/top_k_split + n/K, DELTA)
    certificate_split_total = certificate_split + delta_split - delta_split_total
    
    # This is omnicient with respect with the data split
    certificate_omniscient, B_omniscient, delta_omniscient = second_stage(omniscient_k, m, mu, train_means_indices)
    certificate_dominant, B_dominant, delta_dominant = second_stage(dominant_k, m, mu, train_means_indices)
    true_value = float(np.max(mu))

    certificate_ucb, delta_ucb = UCB(mu, K, N, delta=DELTA)
    certificate_se, delta_se = successive_elimination(mu, K, N, delta=DELTA)

    artifacts = {"sample_split": {"certificate":certificate_split, 
                        "delta": delta_split, 
                        "true_value": true_value}}
    artifacts["sample_split_total"] = {"certificate":certificate_split_total, 
                        "delta": delta_split_total, 
                        "true_value": true_value}

    artifacts["omniscient"] = {"certificate":certificate_omniscient, "delta": delta_omniscient,"true_value": true_value}
    artifacts["dominant"] = {"certificate":certificate_dominant, "delta": delta_dominant,"true_value": true_value}
    artifacts["ucb"] = {'certificate': certificate_ucb, "true_value": true_value, 'delta': delta_ucb}
    artifacts["successive_elimination"] = {'certificate': certificate_se, "true_value": true_value, 'delta': delta_se}

    if config.run_all_k:
        for new_k in range(1,K+1):
            certificate_k, _, delta_k = second_stage(new_k, m, mu, train_means_indices)
            artifacts["k_{}".format(new_k)] = {'certificate': certificate_k, "true_value": true_value, 'delta': delta_k}

        artifacts["random"] = artifacts["k_{}".format(np.random.randint(1,K))]
        artifacts["one_stage"] = deepcopy(artifacts["k_{}".format(K)])
        artifacts["one_stage"]["delta"] = compute_hoeffding_bound(N//K, DELTA)
        artifacts["one_stage"]["certificate"] += (artifacts["k_{}".format(K)]["delta"]-artifacts["one_stage"]["delta"])

    if config.arm_distribution == 'beta' or config.arm_distribution == "beta_misspecified":
        new_alphas = []
        new_betas = []

        for i in range(len(first_stage)):
            success = np.sum(first_stage[i])
            total = len(first_stage[i])
            new_alphas.append(config.reward_parameters['alpha']+success)
            new_betas.append(config.reward_parameters['beta']-success+total)

        current_set = []
        curr_value = 0

        for k in range(K):
            next_best_value = (0,0)
            for next_elem in range(K):
                if next_elem in current_set:
                    continue 
            
                new_set = deepcopy(current_set)
                new_set.append(next_elem)

                num_trials = 100
                min_certificate = []

                for i in range(num_trials):
                    means = np.array([np.random.beta(new_alphas[new_set[j]],new_betas[new_set[j]]) for j in range(len(new_set))])
                    sampled_values = sample_bernoulli((N-n)//len(new_set), len(new_set), means)
                    predicted_means = np.mean(sampled_values,axis=1)
                    min_certificate_delta = compute_hoeffding_bound((N-n)//len(new_set), DELTA)
                    min_certificate.append(np.max(predicted_means) - min_certificate_delta)
                min_certificate = np.mean(min_certificate)

                if next_best_value[0] < min_certificate:
                    next_best_value = (min_certificate,next_elem)
            if next_best_value[0] == 0:
                break 
            elif next_best_value[0] <= curr_value:
                break 
            else:
                curr_value = next_best_value[0]
                current_set.append(next_best_value[1])

        artifacts["prior"] = {'certificate': [curr_value], "true_value": true_value, 'delta': compute_hoeffding_bound((N-n)//len(current_set), DELTA)}

    # print(certificate_split, B_split, delta_split)
    return artifacts


def get_results_matching_parameters(folder_name,result_name,parameters):
    """Get a list of dictionaries, with data, which match some set of parameters
    
    Arguments:
        folder_name: String, which folder the data is located in
        result_name: String, what the suffix is for the dataset
        parameters: Dictionary with key,values representing some known parameters
        
    Returns: List of Dictionaries"""

    all_results = glob.glob("../../results/{}/{}*.json".format(folder_name,result_name))
    ret_results = []

    for file_name in all_results:
        load_file = json.load(open(file_name,"r"))

        for p in parameters:
            if p not in load_file['parameters'] or load_file['parameters'][p] != parameters[p]:
                if p in load_file['parameters']['reward_parameters']:
                    if load_file['parameters']['reward_parameters'][p] != parameters[p]:
                        break 
                else:
                    break 
        else:
            ret_results.append(load_file)
    return ret_results

def aggregate_data(results):
    """Get the average and standard deviation for each key across 
        multiple trials
        
    Arguments: 
        results: List of dictionaries, one for each seed
    
    Returns: Dictionary, with each key mapping to a 
        tuple with the mean and standard deviation"""

    r = [] 
    for row in results:
        temp = {}
        for key in row:
            if key != 'parameters':
                temp[key] = row[key]['certificate']
        r.append(temp)
    results = r

    ret_dict = {}
    for l in results:
        for k in l:
            if type(l[k]) == int or type(l[k]) == float:
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k].append(l[k])
            elif type(l[k]) == list and (type(l[k][0]) == int or type(l[k][0]) == float):
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k].append(l[k][0])
            elif type(l[k]) == type(np.array([1,2])):
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k] += list(l[k])
            elif type(l[k]) == list:
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k] += list(l[k][0])

    for i in ret_dict:
        ret_dict[i] = (np.mean(ret_dict[i]),np.std(ret_dict[i]))
    
    return ret_dict 
