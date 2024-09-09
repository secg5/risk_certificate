import json
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

import wandb
import numpy as np
import pandas as pd
import pickle

from certificate.config import Config
from certificate.run_simulations import sample_bernoulli

DELTA = 0.1

def sample_from_posterior(successes, failures, alpha=1, beta=1, num_samples=1000, rng=None):
    """
    Sample from the posterior distribution of a Bernoulli with a Beta prior.

    Parameters:
    - successes: Number of observed successes (sum of 1s in Bernoulli trials).
    - failures: Number of observed failures (sum of 0s in Bernoulli trials).
    - alpha: Prior alpha parameter of the Beta distribution.
    - beta: Prior beta parameter of the Beta distribution.
    - num_samples: Number of samples to draw from the posterior.

    Returns:
    - samples: A NumPy array of samples from the posterior Beta distribution.
    """
    alpha = np.full(len(successes), alpha) if np.isscalar(alpha) else np.array(alpha)
    beta = np.full(len(failures), beta) if np.isscalar(beta) else np.array(beta)

    # Posterior parameters
    alpha_post = alpha + successes
    beta_post = beta + failures

    # Sample from the posterior Beta distribution
    samples = np.random.beta(alpha_post[:, None], beta_post[:, None], (len(successes), num_samples))
    
    return samples, alpha_post, beta_post

def sample_posterior_brute_force(successes, failures, sample_mu, prior ,num_samples=1000, rng=None):
    """
    Samples from the posterior distribution using a brute-force approach on adiscrete sample.

        Args:
            successes (numpy.ndarray): A 1D array representing the number of successes for each element.
            failures (numpy.ndarray): A 1D array representing the number of failures for each element.
            sample_mu (numpy.ndarray): A 2D array where each row corresponds to a different set of probabilities (parameter values) over which the posterior is being sampled.
            prior (numpy.ndarray): A 1D array representing the prior distribution over the parameter values.
            num_samples (int, optional): The number of samples to draw from the posterior. Defaults to 1000.
            rng (numpy.random.Generator, optional): A NumPy random generator instance for reproducibility. Defaults to None.

        Returns:
            numpy.ndarray: A 1D array of samples drawn from the posterior distribution, of length `num_samples`.
    """
    likelihood = np.prod(sample_mu**successes * (1-sample_mu)**failures, axis=1)
    posterior = likelihood*prior/np.sum(likelihood*prior)
    posterior_sample = rng.choice(sample_mu, size=num_samples, p=posterior)

    return posterior_sample, posterior

def single_prior(config_dict):
    parser = ArgumentParser(description="Run experiments with JSON configuration.")
    parser.add_argument("config", help="Path to the JSON configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    # guardar el config
    
    config = Config(**config_dict)

    rng = np.random.default_rng(config.random_seed)

    K = config.number_arms  # Number of Bernoulli distributions
    N = config.sample_size  # Number of samples to draw from each distribution

    n = config.first_stage_size
    m = N - n

    # TODO: Decide if the code should do a single run for a particular p
    mu = config.distribution
    assert len(mu) == K , "Number of arms is not consisted with the distribtuion provided."
    
    # Although we sample knowing mu in reality the sample is the only thing that is
    # import pdb; pdb.set_trace()
    mu = np.array(mu)
    # TODO: Do i need a second stage at all?
    sample = sample_bernoulli(n, K, mu)
    
    # Example usage:
    # Assume you observed 10 successes and 5 failures
    successes = sample.sum(axis=-1)
    failures = (n//K) - successes
    print(n//K)

    # Assume a prior Beta(1, 1) which is equivalent to a uniform prior
    alpha_prior = 1
    beta_prior = 1
   
    # Draw 1000 samples from the posterior


    # Brute force approach with a grid of possible values for mu
    # mu_saples = ???
    # n = mu_samples.shape(0)
    # mu = config.distribution
    # n = config.first_stage_size
    # m = N - n
    
    # p_0 = np.array([1/n for _ in range(len(n))])
    # import pdb; pdb.set_trace()
    # p_0 = np.array([1/K for _ in range(K)])

    # import pdb; pdb.set_trace()
    B = []
    previous_objective = 0
    # The trivila solution is just outputting the number of arms
    while len(B) < K:
        max_objective = -np.inf
        max_index = 0
        
        # How to code this??
        # posterior_samples = sample_posterior_brute_force(successes, failures, mu_samples, p_0, num_samples=1000, rng=rng)
        posterior_samples, alpha_post, beta_post  = sample_from_posterior(successes, failures, alpha_prior, beta_prior, num_samples=1000, rng=rng)
        
        for i in range(K):
            if i in B:
                continue
            # Geedy step?
            candidate_set = B + [i]
            delta = np.sqrt(2*(len(candidate_set)/m)*np.log(2/DELTA))
            objective = posterior_samples[:,candidate_set].max(axis=1).mean() - delta
            if objective > max_objective:
                max_objective = objective
                max_index = i

        # Greedy step
        B.append(max_index)

        # Is this correct? Isnt sumodular optimization only good  in the monotone for unconstrained optimization?
        if max_objective < previous_objective:
            break
        previous_objective = max_objective  
        alpha_prior = alpha_post
        beta_prior = beta_post 

    print(B)



# if __name__ == "__main__":

def uniform_prior(config_dict):
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

    n = config.first_stage_size
    m = N - n

    # TODO: Decide if the code should do a single run for a particular p
    mu = config.distribution
    assert len(mu) == K , "Number of arms is not consisted with the distribtuion provided."
    
    # Although we sample knowing mu in reality the sample is the only thing that is
    # import pdb; pdb.set_trace()
    mu = np.array(mu)
    # TODO: Do i need a second stage at all?
    sample = sample_bernoulli(n, K, mu)


    n_samples_mu = 100
    # beware, fixing the alpha will fix the curtosis
    alpha = 2
    beta_params = [(mean * alpha, (1 - mean) * alpha) for mean in mu]
    samples_mu = np.array([np.random.beta(a, b, n_samples_mu) for a, b in beta_params]).T
  
    # Example usage:
    # Assume you observed 10 successes and 5 failures
    successes = sample.sum(axis=-1)
    failures = (n//K) - successes

    alpha = 0.1


    Bs = []
    values = []
    # The trivila solution is just outputting the number of arms
    for j in range(1, K + 1):
        p_0 = np.array([1/n_samples_mu for _ in range(n_samples_mu)])
        B = []
        prior = p_0

        while len(B) < j:
            max_objective = -np.inf
            max_index = 0
            
            # posterior_samples = sample_posterior_brute_force(successes, failures, mu_samples, p_0, num_samples=1000, rng=rng)
            posterior_samples, posterior  = sample_posterior_brute_force(successes, failures, samples_mu, prior, num_samples=1000, rng=rng)
            
            # bad indexing
            for i in range(K):
                if i in B:
                    continue
                # Geedy step
                candidate_set = B + [i]
                objective = posterior_samples[:,candidate_set].max(axis=1).mean()
                if objective > max_objective:
                    max_objective = objective
                    max_index = i

            # Greedy step
            B.append(max_index)  
            prior = posterior
        value = posterior_samples[:,B]
        values.append(value)
        Bs.append(B)
    # import pdb; pdb.set_trace()


    
    deltas = np.sqrt(2*(np.array(range(1, K+1))/m)*np.log(2/DELTA))
    max_values = [value.max(axis=1).mean() for value in values]
    objective = np.array(max_values) - deltas
    size = np.argmax(objective)
    certificate = values[size].mean(axis=0) - deltas[size]
    delta = deltas[size]
    B = Bs[size]
    # print(B)

    artifacts = {"uniform_prior": {"certificate":certificate, "B": B, "delta": delta}}

    with open(f"artifacts_{config.experiment_name}.pkl", "wb") as f:
        pickle.dump(artifacts, f)









     





    # Using some sample to estimate effect sizes
