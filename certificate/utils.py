
import glob 
import json 
import numpy as np
import os

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

def compute_hoeffding_bound(n_data, delta):
    """Given n pulls of an arm, compute the lower bound
        on the \delta^th percentile for the mean
        Using a Hoeffding Bound
    
    Arguments:
        n_data: Integer, number of times algorithm has pulled
        delta: Float, percent chance of success of Hoeffding
    
    Returns: Float, lower bound (when subtracted from \mu)"""

    return np.sqrt((1/(2*n_data))*np.log(1/delta))

def compute_hoeffding_bound_one_way(n_data, delta):
    """Given n pulls of an arm, compute the lower bound
        on the \delta^th percentile for the mean
        Using a Hoeffding Bound
        Here, we do 2/delta, because we just want a 
        bound for whether it is larger
    
    Arguments:
        n_data: Integer, number of times algorithm has pulled
        delta: Float, percent chance of success of Hoeffding
    
    Returns: Float, lower bound (when subtracted from \mu)"""

    return np.sqrt(np.log(2 / delta) / (2* n_data))


def sample_bernoulli(N, K, mu, indices=None):
    """
    Generate a sample of N from K different Bernoulli distributions with given probabilities.
    
    Arguments:
        N (int): Number of samples to draw from each distribution.
        K (int): Number of different Bernoulli distributions.
        p (list or np.array): Probabilities for each of the K distributions.
    
    Returns:
        np.array: A K x N array where each row contains N samples from the corresponding Bernoulli distribution.
    """
    if indices is None:
        indices = range(K)
    samples = np.random.binomial(1, mu[indices, np.newaxis], (K, N//K))
    return samples
