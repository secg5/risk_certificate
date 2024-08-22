import json
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

import wandb
import numpy as np
import pandas as pd
import pickle

from config import Config


def sample_bernoulli(N, K, mu, indices=None, rng=None):
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
    samples = rng.binomial(1, mu[indices, np.newaxis], (K, N//K))
    
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
    delta = np.sqrt((np.arange(K)+1)/m)

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
    dominant_k = non_negative_indices[0] + 1 if non_negative_indices.size > 0 else None
    
    # import pdb; pdb.set_trace()
    str_mu = np.take_along_axis(mu, train_means_indices, axis=0)
    true_values = str_mu - delta
    # TODO : Think about what the real objective is
    omniscient_k = np.argmax(true_values) + 1
    # import pdb; pdb.set_trace()
    # TODO pick top 1,2 and 3
    # TODO Do the method that is omnicient on the sample splitting
    
    return top_k_split, omniscient_k, dominant_k, train_means_indices

# if __name__ == "__main__":
def run_experiments(config_dict):
    # parser = ArgumentParser(description="Run experiments with JSON configuration.")
    # parser.add_argument("config", help="Path to the JSON configuration file")
    # args = parser.parse_args()
    # with open(args.config, "r") as f:
    #     config_dict = json.load(f)
    # guardar el config
    config = Config(**config_dict)

    rng = np.random.default_rng(config.random_seed)

    K = config.number_arms  # Number of Bernoulli distributions
    N = config.sample_size  # Number of samples to draw from each distribution
    
    # TODO: Decide if the code should do a single run for a particular p
    # mu = np.random.rand(K) # Random probabilities for each distribution
    mu = config.distribution
    n = config.first_stage_size
    m = N - n
    assert n % K == 0, "The number of samples in the first stage must be a multiple of the number of arms."
    assert m % K == 0, "The number of samples in the second stage must be a multiple of the number of arms."
    assert len(mu) == K , "Number of arms is not consisted with the distribtuion provided."
    

    mu = np.array(mu)
    # print(mu, K, "n", n, "m", m)
    first_stage = sample_bernoulli(n, K, mu, rng=rng)
    top_k_split, omniscient_k, dominant_k, train_means_indices = run_algorithm(first_stage, K, n, m, mu)
    DELTA = 0.1

    # print("K", top_k_split, omniscient_k, dominant_k)
    def second_stage(k, m, mu, train_means_indices, rng):
        # import pdb; pdb.set_trace()
        B = train_means_indices[:k]
        second_stage = sample_bernoulli(m, k, mu, B, rng=rng)
        delta = np.sqrt(2*(k/m)*np.log(2/DELTA))
        certificate = second_stage.mean(axis=1) - delta
        
        return certificate, B, delta
    

    certificate_split, B_split, delta_split = second_stage(top_k_split, m, mu, train_means_indices, rng)
    # This is omnicient with respect with the data split
    certificate_omniscient, B_omniscient, delta_omniscient = second_stage(omniscient_k, m, mu, train_means_indices, rng)
    certificate_dominant, B_dominant, delta_dominant = second_stage(dominant_k, m, mu, train_means_indices, rng)
    true_value = np.max(mu) - np.sqrt(2*(1/m)*np.log(2/DELTA))
    # print ("B", "splt", B_split, "Omnicient",B_omniscient,"Dominant", B_dominant)

    # TODO: is it worth using dta classes?
    artifacts = {"sample_split": {"certificate":certificate_split, "B": B_split, "delta": delta_split, "true_value": true_value}}
    artifacts["omniscient"] = {"certificate":certificate_omniscient, "B": B_omniscient, "delta": delta_omniscient}
    artifacts["dominant"] = {"certificate":certificate_dominant, "B": B_dominant, "delta": delta_dominant}

    with open(f"artifacts_{config.experiment_name}.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    # print(certificate_split, B_split, delta_split)
    return artifacts

