"""Utility functions used in deep symbolic regression."""

import os
import functools
import numpy as np
import pandas as pd
import time
import pickle

def is_float(s):
    """Determine whether str can be cast to float."""

    try:
        float(s)
        return True
    except ValueError:
        return False

def load_pickle(path):
    # function loads saved charge points
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

    # Adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points given an array of costs.

    Parameters
    ----------

    costs : np.ndarray
        Array of shape (n_points, n_costs).

    Returns
    -------

    is_efficient_maek : np.ndarray (dtype:bool)
        Array of which elements in costs are pareto-efficient.
    """

    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


def setup_output_files(logdir, output_file, token_names):
    """
    Writes the main output file header and returns the reward, hall of fame, and Pareto front config filenames.

    Parameters:
    -----------

    logdir : string
        Directory to log to.

    output_file : string
        Name of output file.

    token_names : list
        Names of tokens in library, used to add to end of output file to correctly name tokens in output file.

    Returns:
    --------

    all_r_output_file : string
        all_r output filename

    hof_output_file : string
        hof output filename

    pf_output_file : string
        pf output filename
    """
    os.makedirs(logdir, exist_ok=True)
    output_file = os.path.join(logdir, output_file)
    prefix, _ = os.path.splitext(output_file)
    all_r_output_file = "{}_all_r.npy".format(prefix)
    hof_output_file = "{}_hof.csv".format(prefix)
    pf_output_file = "{}_pf.csv".format(prefix)
    with open(output_file, 'w') as f:
        # r_best : Maximum across all iterations so far
        # r_max : Maximum across this iteration's batch
        # r_avg_full : Average across this iteration's full batch (before taking epsilon subset)
        # r_avg_sub : Average across this iteration's epsilon-subset batch
        # n_unique_* : Number of unique Programs in batch
        # n_novel_* : Number of never-before-seen Programs per batch
        # a_ent_* : Empirical positional entropy across sequences averaged over positions
        # invalid_avg_* : Fraction of invalid Programs per batch
        headers = ["base_r_best",
                   "base_r_max",
                   "base_r_avg_full",
                   "base_r_avg_sub",
                   "r_best",
                   "r_max",
                   "batch_r_max_expression",
                   "r_avg_full",
                   "r_avg_sub",
                   "l_avg_full",
                   "l_avg_sub",
                   "ewma",
                   "pg_loss",
                   "inv_loss",
                   "ent_loss",
                   "n_unique_full",
                   "n_unique_sub",
                   "n_novel_full",
                   "n_novel_sub",
                   "a_ent_full",
                   "a_ent_sub",
                   "invalid_avg_full",
                   "invalid_avg_sub",
                   "sample_metric",
                   "nfev_avg_full",
                   "nfev_avg_sub",
                   "nit_avg_full",
                   "nit_avg_sub",
                   "eq_w_const_full",
                   "eq_w_const_sub",
                   "n_const_per_eq_full",
                   "n_const_per_eq_sub",
                   "n_unq_tokens_avg_full",
                   "n_unq_tokens_avg_sub",
                   "duration"
                   ]
        headers += [token_name + '_full' for token_name in token_names]
        headers += [token_name + '_sub' for token_name in token_names]

        f.write("{}\n".format(",".join(headers)))

    return all_r_output_file, hof_output_file, pf_output_file


class cached_property(object):
    """
    Decorator used for lazy evaluation of an object attribute. The property
    should be non-mutable, since it replaces itself.
    """

    def __init__(self, getter):
        self.getter = getter

        functools.update_wrapper(self, getter)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = self.getter(obj)
        setattr(obj, self.getter.__name__, value)
        return value


# Entropy computation in batch
def empirical_entropy(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.
    # Compute entropy
    for i in probs:
        ent -= i * np.log(i)

    return ent

def plot_prob_dists(controllers, step, token_names):
    # Not implemented yet.

    # batch = load_pickle('./log/log_2021-10-12-132147_stop/pickled_batches/dsr_saving_pickles_kDeficit_1_step_96_controller_0.p')
    # batch = load_pickle('./log/log_2021-10-12-132147_stop/pickled_batches/dsr_saving_pickles_kDeficit_1_step_368_controller_0.p')
    # actions = batch.actions
    # priors = batch.priors
    pass

def test_fixed_actions(logdir, from_tokens):

    from numpy import genfromtxt

    actions_file = './test/data/dsr_20_consts_saving_actions_1_full.csv'

    full_actions = genfromtxt(actions_file, delimiter=',', dtype='int32')

    # Separate batches
    full_batches_idx = np.where(full_actions[:, 1] == 100)[0]

    filtered_batches_idx = []

    # choose to filter batches or not:
    prev_step = -1
    for ii in range(len(full_batches_idx)):
        if full_actions[full_batches_idx[ii], 0] % 20 == 0:
            if full_actions[full_batches_idx[ii], 0] != prev_step:
                prev_step = full_actions[full_batches_idx[ii], 0]
            else:
                filtered_batches_idx.append(full_batches_idx[ii])
        else:
            filtered_batches_idx.append(full_batches_idx[ii])

    results_filename = f'{logdir}/results_fixed_actions.csv'

    df_save = pd.DataFrame(columns=['batch', 'duration_proc', 'invalid_avg', 'r', 'nit_avg'])
    df_save.to_csv(results_filename, index=False)

    full_batches_idx = np.array(filtered_batches_idx)

    batches_match_percentage = []
    batches_best_performers = []

    # df_expressions_full = pd.DataFrame()
    # df_expressions_sub = pd.DataFrame()
    for ii in range(len(full_batches_idx)):
        # for ii in range(1):
        # calculate statistics for each batch
        print(f"{ii + 1} of {len(full_batches_idx)}")

        full_batch = full_actions[full_batches_idx[ii] + 1:full_batches_idx[ii] + 1001, :]

        # create programs
        start_proc = time.process_time()
        programs = [from_tokens(a, optimize=2000) for a in full_batch]

        r = np.array([p.r for p in programs])
        duration_proc = time.process_time() - start_proc

        invalid_avg = np.mean([p.invalid for p in programs])
        nit_avg = np.mean([p.nit for p in programs])

        if any(np.isnan(r)):
            # if the const optimisation returns nan constants, the rewards is nan, that is set to min reward here.
            r[np.where(np.isnan(r))[0]] = min(r)

        df_save = pd.DataFrame(
            columns=['batch', 'duration_proc', 'invalid_avg', 'r', 'nit_avg'])
        df_save['batch'] = [full_actions[full_batches_idx[ii], 0]]
        df_save['duration_proc'] = [duration_proc]
        df_save['invalid_avg'] = [invalid_avg]
        df_save['r'] = [max(r)]
        df_save['nit_avg'] = nit_avg

        df_save.to_csv(results_filename, mode='a', header=False, index=False)

