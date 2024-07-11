import json
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt

import wandb
import numpy as np
from config import Config


def sample_bernoulli(N, K, mu):
    """
    Generate a sample of N from K different Bernoulli distributions with given probabilities.
    
    Parameters:
    N (int): Number of samples to draw from each distribution.
    K (int): Number of different Bernoulli distributions.
    p (list or np.array): Probabilities for each of the K distributions.
    
    Returns:
    np.array: A K x N array where each row contains N samples from the corresponding Bernoulli distribution.
    """
    if len(mu) != K:
        raise ValueError("Length of p must be equal to K")
    
    samples = np.random.binomial(1, mu[:, np.newaxis], (K, N//K))
    
    return samples

def run_algorithm(first_stage, K, n, m):

    # Is this propperly implemented? Should i use the second stage instead?
    
    train_data = first_stage[:,:(n // 2)]
    test_data = first_stage[:,(n // 2):n]
    train_means = train_data.mean(axis=1)
    train_means_indices = train_means.argsort()[::-1]

    srt_means = np.take_along_axis(train_means, train_means_indices, axis=0)
    test_means = test_data.mean(axis=1)

    srt_test_means = np.take_along_axis(test_means, train_means_indices, axis=0)
    delta = np.sqrt((np.arange(K)+1)/m)
    values = srt_test_means - delta

    best_estimate = np.argmax(values)
    top_k_split = train_means_indices[best_estimate]

    # First not dominant arm, why here? as an alternative to top-k...
    # Is not clear how to use this in the original two satge approach
    # The band used in the two stage approach is dependant on the size of the set picked, 
    # thus adding making it dependant on the size itself?
    # code top-, top-2 top-3
    # train menas?
    aux = values - np.roll(srt_test_means, -1, axis=0)
    non_negative_mask = aux >= 0
    non_negative_indices = np.where(non_negative_mask)[0]
    dominant_k = non_negative_indices[0] if non_negative_indices.size > 0 else None
    
    # import pdb; pdb.set_trace()
    str_mu = np.take_along_axis(mu, train_means_indices, axis=0)
    true_values = str_mu - delta
    # TODO : Think about what the real objective is
    omniscient_k = train_means_indices[np.argmax(true_values)]
    # TODO pick top 1,2 and 3
    
    return top_k_split, dominant_k, omniscient_k, train_means_indices

if __name__ == "__main__":
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
    
    # TODO: Decide if the code should do a single run for a particular p
    # mu = np.random.rand(K) # Random probabilities for each distribution
    mu = config.distribution
    n = config.first_stage_size
    m = N - n
    assert n % K == 0, "The number of samples in the first stage must be a multiple of the number of arms."
    assert m % K == 0, "The number of samples in the second stage must be a multiple of the number of arms."
    

    mu = np.array(mu)
    
    first_stage = sample_bernoulli(n, K, mu)
    top_k_split, dominant_k, omniscient_k, train_means_indices = run_algorithm(first_stage, K, n, m)

    DELTA = 0.1
    print(top_k_split)
    second_stage = sample_bernoulli(m, K, mu)
    certificate = second_stage[train_means_indices].mean(axis=1) - np.sqrt(2*(top_k_split/m)*np.log(2/DELTA))

    print(np.sqrt(2*(top_k_split/m)*np.log(2/DELTA)))
    print(certificate)


    # Example data
    x = np.arange(5)
    y = mu  # Array to be plotted as dots
    lower_bounds = certificate    # Example lower bounds

    # Set ggplot style
    plt.style.use('ggplot')

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 6))

    # Define the color for the elements
    color = 'dodgerblue'

    # Plot the main array as dots
    ax.plot(x, y, 'o', color=color, label='Data points')

    # Plot the lower bounds as solid lines and horizontal bars
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [lower_bounds[i], y[i]], '-', color=color)  # Vertical line
        ax.plot([x[i] - 0.05, x[i] + 0.05], [lower_bounds[i], lower_bounds[i]], '-', color=color)  # Horizontal bar

    # Add labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Plot with Data Points and Lower Bounds')
    ax.legend()

    # Customize the background grid lines to match ggplot
    ax.grid(True, color='white', linestyle='-', linewidth=0.7)
    ax.set_facecolor('#E5E5E5')  # ggplot-like grey background

    # Save the plot to a file
    plt.savefig('plot_with_bounds.png')
    # param_grid = {
    #     "loss_functions": config.loss_functions,
    #     "alphas": config.alphas,
    #     "lambdas": config.lambdas,
    # }

    # param_combinations = list(itertools.product(*param_grid.values()))
    # processed_loss_types = set()
    # dfs = []
    # for combination_idx, (loss_function, alpha, lambda_) in enumerate(
    #     tqdm(param_combinations)
    # ):
    #     loss = getattr(losses, loss_function)(dataset.calibration_size)
    #     if (
    #         loss.is_lambda_independent
    #         and (loss_function, alpha) in processed_loss_types
    #     ):
    #         continue
    #     wandb.init(
    #         project="uq_decision_making",
    #         entity="cmpatino",
    #         name=f"{config.dataset}_{loss_function}_{alpha}_{lambda_}",
    #         config={
    #             "dataset": config.dataset,
    #             "probabilities_id": config.probabilities_id,
    #             "loss_function": loss_function,
    #         },
    #     )

    #     # Get penalties
    #     penalties, pairwise_distances = dataset.get_penalties(all_labels)

    #     scores = loss.get_scores(
    #         penalties=penalties,
    #         pairwise_distances=pairwise_distances,
    #         probabilities=all_softmax,
    #         lambda_=lambda_,
    #         labels=all_labels,
    #     )

    #     cal_scores = scores[calibration_indicator]
    #     val_labels = all_labels[~calibration_indicator]
    #     val_softmax = all_softmax[~calibration_indicator]
    #     val_penalties = penalties[~calibration_indicator]

    #     q_hat = loss.get_quantile(cal_scores, alpha)

    #     prediction_sets = loss.get_prediction_sets(
    #         penalties=val_penalties,
    #         pairwise_distances=pairwise_distances,
    #         probabilities=val_softmax,
    #         lambda_=lambda_,
    #     )

    #     n_val = val_softmax.shape[0]
    #     empirical_coverage = prediction_sets[np.arange(n_val), val_labels].mean()
        
    #     set_size, set_loss = loss.eval_sets(prediction_sets, val_penalties)
    #     additional_log = {}

    #     if len(set_loss.shape) > 1:
    #         set_loss, set_loss_1 = set_loss[0], set_loss[1]
            
    #         additional_log = {
    #             "loss_1_mean": set_loss_1.mean(),
    #             "loss_1_std": set_loss_1.std(),
    #             "loss_1_median": np.median(set_loss_1),
    #             "loss_1_max": set_loss_1.max(),
    #             "loss_1_min": set_loss_1.min(),
    #             "loss_1_mode": stats.mode(set_loss_1).mode,
    #             "size_loss_1_hist": wandb.Image(plot_loss_size_hists(set_size, set_loss_1))
    #         }

    #     log = {
    #             "empirical_coverage": empirical_coverage,
    #             "quantile": q_hat,
    #             "lambda": lambda_,
    #             "1 - alpha": 1 - alpha,
    #             "size_mean": set_size.mean(),
    #             "size_std": set_size.std(),
    #             "size_median": np.median(set_size),
    #             "size_max": set_size.max(),
    #             "size_min": set_size.min(),
    #             "size_mode": stats.mode(set_size).mode,
    #             "loss_mean": set_loss.mean(),
    #             "loss_std": set_loss.std(),
    #             "loss_median": np.median(set_loss),
    #             "loss_max": set_loss.max(),
    #             "loss_min": set_loss.min(),
    #             "loss_mode": stats.mode(set_loss).mode,
    #             "size_loss_hist": wandb.Image(plot_loss_size_hists(set_size, set_loss))
    #         }
    #     processed_loss_types.add((loss_function, alpha))

    #     log.update(additional_log)
    #     wandb.log(log)

    #     wandb.finish()

    #     size_loss_df = pd.DataFrame(
    #         {
    #             "size": set_size,
    #             "loss": set_loss,
    #             "loss_1": set_loss_1,
    #             "lambda": lambda_,
    #             "loss_function": loss_function,
    #             "alpha": alpha,
    #         }
    #     )
    #     dfs.append(size_loss_df)
    
    # size_loss_df = pd.concat(dfs)
    # size_loss_df.to_csv(f"size_loss_{config.dataset}_baseline.csv", index=False)