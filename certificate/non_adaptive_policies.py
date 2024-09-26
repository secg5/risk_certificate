from certificate.utils import compute_hoeffding_bound, sample_bernoulli
import numpy as np

def compute_top_k(first_stage, n, s_1, s_2, mu):
    """
    Arguments:
        first_stage: Numpy matrix (n x s_{1}//n) of rewards for each
            arm for each timestep in the first stage
        n: Number of total arms 
        s_1: Total arm pulls in the first stage 
        s_2: Total arm pulls in the second stage
        mu: True arm means

    Returns: 4 things
        top_k_split: Value of k from sample splitting
        omniscient_k: Value of k with knowledge of mu
        dominant_k: Value of k where ??
        train_means_indices: Sorted list of the top arms 
    """

    sample_number = s_1 // n
    train_data = first_stage[:,:(sample_number // 2)]
    test_data = first_stage[:,(sample_number // 2):sample_number]
    train_means = train_data.mean(axis=1)
    train_means_indices = train_means.argsort()[::-1]

    test_means = test_data.mean(axis=1)

    srt_test_means = np.take_along_axis(test_means, train_means_indices, axis=0)
    delta = np.sqrt((np.arange(n)+1)/(2*s_2))

    values = []
    for i in range(n):
        value = np.max(srt_test_means[:(i+1)]) - delta[i]
        values.append(value)
    values = np.array(values)
    
    best_estimate = np.argmax(values)    
    top_k_split = best_estimate + 1

    aux = values - np.roll(srt_test_means, -1, axis=0)
    non_negative_mask = aux >= 0
    non_negative_indices = np.where(non_negative_mask)[0]
    dominant_k = non_negative_indices[0] + 1 if non_negative_indices.size > 0 else 1
    str_mu = np.take_along_axis(mu, train_means_indices, axis=0)
    true_values = str_mu - delta
    omniscient_k = np.argmax(true_values) + 1
    
    return top_k_split, omniscient_k, dominant_k, train_means_indices


def fixed_k_policies(k, s_2, arm_means, selected_arms,delta,seed):
    np.random.seed(seed)
    B = selected_arms[:k]
    second_stage = sample_bernoulli(s_2, k, arm_means, B)
    lower_bound = compute_hoeffding_bound(s_2/k, delta)
    certificate = second_stage.mean(axis=1) - lower_bound
    return certificate, lower_bound
