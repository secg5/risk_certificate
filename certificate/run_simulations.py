import numpy as np
from copy import deepcopy 
from certificate.utils import compute_hoeffding_bound, sample_bernoulli
from certificate.adaptive_policies import UCB, successive_elimination
from certificate.non_adaptive_policies import compute_top_k, fixed_k_policies, two_stage_thompson_sampling, two_stage_successive_elimination
from certificate.prior_policies import beta_prior_policy

def generate_arm_means(arm_distribution,arm_parameters,n_arms):
    """Generate the underlying parameters for all arms 
    
    Arguments:
        arm_distribution: String, either uniform, beta, 
            or beta_misspecified
        arm_parameters: Dictionary, with information on the following: 
            alpha, beta for beta distributions
            diff_mean_1, diff_std_1, for misspecified
        n_arms: Integer, number of arms we're generating 
    
    Returns: List of floats, the means for all arms"""

    arm_means = []

    for _ in range(n_arms):
        if arm_distribution == 'uniform':
            arm_means.append(np.random.random())
        elif arm_distribution == 'beta':
            arm_means.append(np.random.beta(arm_parameters['alpha'],arm_parameters['beta']))
        elif arm_distribution == 'beta_misspecified':
            arm_means.append(np.clip(np.random.beta(arm_parameters['alpha'],arm_parameters['beta']) + np.random.normal(arm_parameters['diff_mean_1'],arm_parameters['diff_std_1']),0,1))
    if arm_distribution == 'unimodal_diff':
        arm_means.append(np.random.random())    
        for _ in range(1,n_arms):
            diff = np.random.normal(arm_parameters['diff_mean_1'],arm_parameters['diff_std_1']) 
            arm_means.append(min(max(arm_means[-1]-diff,0.0001),1))
    if arm_distribution == 'bimodal_diff':
        arm_means.append(np.random.random())    
        for _ in range(1,n_arms):
            if np.random.random() < 0.5:
                diff = np.random.normal(arm_parameters['diff_mean_1'],arm_parameters['diff_std_1']) 
            else:
                diff = np.random.normal(arm_parameters['diff_mean_2'],arm_parameters['diff_std_2']) 
            arm_means.append(min(max(arm_means[-1]-diff,0.0001),1))
    return arm_means 


def run_experiments(config):
    """Given a set of parameters, run experiments comparing various
        policies for selecting arms
        
    Arguments:
        config: Dictionary with information on number of arms, 
            sample_size, delta, arm distributions, and s_1, T
    
    Returns: Dictionary of results; each key is a type of algorithm
        which maps to a dictionary with a (list of) certificates and a 
        certificate width"""

    n = config['number_arms']
    T = config['sample_size']
    mu = config['distribution']
    s_1 = config['first_stage_size']
    delta = config['delta']
    reward_parameters = config['reward_parameters']
    seed = config['random_seed']
    s_2 = T - s_1    
    mu = np.array(mu)

    np.random.seed(seed)
    first_stage = sample_bernoulli(s_1, n, mu)

    # Two Stage Algorithms
    top_k_split, omniscient_k, dominant_k, train_means_indices = compute_top_k(first_stage, n, s_1, s_2, mu)
    certificate_omniscient, width_omniscient = fixed_k_policies(omniscient_k, s_2, mu, train_means_indices,delta,seed)
    certificate_dominant, width_dominant = fixed_k_policies(dominant_k, s_2, mu, train_means_indices,delta,seed)
    certificate_split, width_split = fixed_k_policies(top_k_split, s_2, mu, train_means_indices,delta,seed)
    width_split_total = compute_hoeffding_bound(s_2/top_k_split + s_1/n, delta)
    certificate_split_total = certificate_split + width_split - width_split_total 

    # Other Two Stage Algorithms
    certificate_thompson, width_thompson = two_stage_thompson_sampling(first_stage,n,delta,s_1,T,mu,seed)
    certificate_two_stage_se, width_two_stage_se = two_stage_successive_elimination(first_stage,n,mu, s_1, T, delta,seed)

    # Adaptive Algorithms
    certificate_ucb, width_ucb = UCB(mu, n, T, delta,seed)
    certificate_se, width_se = successive_elimination(mu, n, T, delta,seed)

    results = {} 
    results["omniscient"] = {"certificate":certificate_omniscient, 
        "certificate_width": width_omniscient}
    results["dominant"] = {"certificate":certificate_dominant, 
        "certificate_width": width_dominant}

    results["sample_split"]  = {"certificate":certificate_split, 
                        "certificate_width": width_split}
    results["sample_split_total"] = {"certificate":certificate_split_total, 
                        "certificate_width": width_split_total}

    results['two_stage_thompson'] = {"certificate": certificate_thompson, 
                        "certificate_width": width_thompson}
    results['two_stage_successive_elimination'] = {'certificate': certificate_two_stage_se, 
                        "certificate_width": width_two_stage_se}

    results["ucb"] = {'certificate': certificate_ucb,
        'certificate_width': width_ucb}
    results["successive_elimination"] = {'certificate': certificate_se, 
        'certificate_width': width_se}
    
    if config['run_all_k']:
        for new_k in range(1,n+1):
            certificate_k, delta_k = fixed_k_policies(new_k, s_2, mu, train_means_indices,delta,seed)
            results["k_{}".format(new_k)] = {'certificate': certificate_k, 'certificate_width': delta_k}

        results["random"] = results["k_{}".format(np.random.randint(1,n))]
        results["one_stage"] = deepcopy(results["k_{}".format(n)])
        results["one_stage"]["certificate_width"] = compute_hoeffding_bound(T//n, delta)
        results["one_stage"]["certificate"] += (results["k_{}".format(n)]["certificate_width"]-results["one_stage"]["certificate_width"])

    if 'beta' in config['arm_distribution']:
        certificate_prior, width_prior = beta_prior_policy(first_stage,n,reward_parameters,delta,s_1,T,seed)
        results["prior"] = {'certificate': certificate_prior, 'certificate_width': width_prior}

    return results

