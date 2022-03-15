import numpy as np
import pandas as pd

import dsr
from dsr.library import Library, AD_PlaceholderConstant
from dsr.functions import create_tokens, create_ad_tokens, create_metric_ad_tokens
from dsr.task.regression.dataset import BenchmarkDataset


def make_regression_task(name, function_set, enforce_sum, dataset, dataset_info, metric="inv_nrmse",
    metric_params=(1.0,), extra_metric_test=None, extra_metric_test_params=(),
    reward_noise=0.0, reward_noise_type="r", threshold=1e-12,
    normalize_variance=False, protected=False):
    """
    Factory function for regression rewards. This includes closures for a
    dataset and regression metric (e.g. inverse NRMSE). Also sets regression-
    specific metrics to be used by Programs.

    Parameters
    ----------
    name : str or None
        Name of regression benchmark, if using benchmark dataset.

    function_set : list or None
        List of allowable functions. If None, uses function_set according to
        benchmark dataset.

    dataset : dict, str, or tuple
        If dict: .dataset.BenchmarkDataset kwargs.
        If str: filename of dataset.
        If tuple: (X, y) data

    metric : str
        Name of reward function metric to use.

    metric_params : list
        List of metric-specific parameters.

    extra_metric_test : str
        Name of extra function metric to use for testing.

    extra_metric_test_params : list
        List of metric-specific parameters for extra test metric.

    reward_noise : float
        Noise level to use when computing reward.

    reward_noise_type : "y_hat" or "r"
        "y_hat" : N(0, reward_noise * y_rms_train) is added to y_hat values.
        "r" : N(0, reward_noise) is added to r.

    normalize_variance : bool
        If True and reward_noise_type=="r", reward is multiplied by
        1 / sqrt(1 + 12*reward_noise**2) (We assume r is U[0,1]).

    protected : bool
        Whether to use protected functions.

    threshold : float
        Threshold of NMSE on noiseless data used to determine success.

    Returns
    -------

    task : Task
        Dynamically created Task object whose methods contains closures.
    """

    X_test = y_test = y_test_noiseless = None

    # Benchmark dataset config
    if isinstance(dataset, dict):
        dataset["name"] = name
        benchmark = BenchmarkDataset(**dataset)
        X_train = benchmark.X_train
        y_train = benchmark.y_train
        X_test = benchmark.X_test
        y_test = benchmark.y_test
        y_test_noiseless = benchmark.y_test_noiseless

        # Unless specified, use the benchmark's default function_set
        if function_set is None:
            function_set = benchmark.function_set

    # Dataset filename
    elif isinstance(dataset, str):
        df = pd.read_csv(dataset, header=None) # Assuming data file does not have header rows
        X_train = df.values[:, :-1]
        y_train = df.values[:, -1]

    # sklearn-like (X, y) data
    elif isinstance(dataset, tuple):
        X_train = dataset[0]
        y_train = dataset[1]

    if X_test is None:
        X_test = X_train
        y_test = y_train
        y_test_noiseless = y_test

    X_train_full = X_train
    y_train_full = y_train

    if function_set is None:
        print("WARNING: Function set not provided. Using default set.")
        function_set = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]

    # Save time by only computing these once
    var_y_test = np.var(y_test)
    var_y_test_noiseless = np.var(y_test_noiseless)

    # Define closures for metric
    metric, invalid_reward, max_reward, ad_metric_traversal = make_regression_metric(metric, y_train, *metric_params)
    ad_metric_start_idx = len(ad_metric_traversal) - 1
    if extra_metric_test is not None:
        print("Setting extra test metric to {}.".format(extra_metric_test))
        metric_test, _, _ = make_regression_metric(extra_metric_test, y_test, *extra_metric_test_params) 
    assert reward_noise >= 0.0, "Reward noise must be non-negative."
    if reward_noise:
        assert reward_noise_type in ["y_hat", "r"], "Reward noise type not recognized."
        rng = np.random.RandomState(0)
        y_rms_train = np.sqrt(np.mean(y_train ** 2))
        if reward_noise_type == "y_hat":
            scale = reward_noise * y_rms_train
        elif reward_noise_type == "r":
            scale = reward_noise

    def data_shuffle(seed):
        nonlocal X_train_full, y_train_full

        idx = np.arange(y_train_full.shape[0])
        np.random.seed(seed)
        idx = np.random.permutation(idx)

        X_train_full = X_train_full[idx, :]
        y_train_full = y_train_full[idx]

    def rotate_batch(batch_size):
        nonlocal X_train, y_train, ad_metric_traversal, X_train_full, y_train_full

        # set up train batch
        X_train = X_train_full[:batch_size, :]
        y_train = y_train_full[:batch_size]
        ad_metric_traversal[-2] = AD_PlaceholderConstant(value=y_train, name='y')

        # rotate full dataset
        if batch_size:
            X_train_full = np.roll(X_train_full, -batch_size, axis=0)
            y_train_full = np.roll(y_train_full, -batch_size)

    def reward(p):

        # Compute estimated values
        y_hat, invalid_indices = p.execute(X_train)

        p.invalid_tokens = invalid_indices
        # For invalid expressions, return invalid_reward
        # p.invalid_tokens = [token.invalid for token in p.traversal]
        if p.invalid:
            if invalid_indices is None:
                p.invalid = 1
            else:
                p.invalid = invalid_indices.shape[0]
            # p.invalid = np.sum(p.invalid_tokens, dtype=np.float32)  # overwrite "True" with the number of invalid tokens
            return invalid_reward

        ### Observation noise
        # For reward_noise_type == "y_hat", success must always be checked to 
        # ensure success cases aren't overlooked due to noise. If successful,
        # return max_reward.
        if reward_noise and reward_noise_type == "y_hat":
            if p.evaluate.get("success"):
                return max_reward
            y_hat += rng.normal(loc=0, scale=scale, size=y_hat.shape)

        # Compute metric
        r = metric(y_train, y_hat)

        ### Direct reward noise
        # For reward_noise_type == "r", success can for ~max_reward metrics be
        # confirmed before adding noise. If successful, must return np.inf to
        # avoid overlooking success cases.
        if reward_noise and reward_noise_type == "r":
            if r >= max_reward - 1e-5 and p.evaluate.get("success"):
                return np.inf
            r += rng.normal(loc=0, scale=scale)
            if normalize_variance:
                r /= np.sqrt(1 + 12*scale**2)

        return r

    def set_ad_traversal(p):

        # create ad tokens:
        ad_traversal = create_ad_tokens(p.traversal)
        # update traversal with error metric
        p.ad_traversal = ad_metric_traversal[:-1] + ad_traversal + [ad_metric_traversal[-1]]
        # add ad_const_pos for ad traversal.
        p.ad_const_pos = [pos + len(ad_metric_traversal[:-1]) for pos in p.const_pos]

    def reverse_ad(p):

        # on each evaluation reset the adjoint tokens to 0, except the first to 1.

        # Compute estimated values
        (base_r, jac), invalid_indices = p.ad_reverse(X_train)

        # For invalid expressions, return invalid_reward
        if p.invalid:
            if invalid_indices is None:
                p.invalid_tokens = invalid_indices
                # if the program is invalid but invalid indices is None the tokens are not logged, set invalid to 1
                p.invalid = 1
            else:
                # if invalid weight = 0, p.invalid_tokens will be a dummy array of len == 2

                # Adjust invalid_indices from AD_traversal
                invalid_indices -= ad_metric_start_idx
                invalid_indices = invalid_indices[invalid_indices >= 0]
                p.invalid = invalid_indices.shape[0]
                p.invalid_tokens = invalid_indices
                # p.invalid_tokens = p.invalid_tokens[ad_metric_start_idx:-1]
                # p.invalid = np.sum(p.invalid_tokens, dtype=np.float32)
                # p.invalid = np.sum(p.invalid_tokens[ad_metric_start_idx:-1], dtype=np.float32)
            return invalid_reward, np.zeros(jac.shape)


        # set self.r and self.jac at end of this function

        return base_r, jac


    def evaluate(p):

        # Compute predictions on test data
        y_hat, invalid_indices = p.execute(X_test)
        if invalid_indices is not None:
            nmse_test = None
            nmse_test_noiseless = None
            success = False

        else:
            # NMSE on test data (used to report final error)
            nmse_test = np.mean((y_test - y_hat)**2) / var_y_test

            # NMSE on noiseless test data (used to determine recovery)
            nmse_test_noiseless = np.mean((y_test_noiseless - y_hat)**2) / var_y_test_noiseless

            # Success is defined by NMSE on noiseless test data below a threshold
            success = nmse_test_noiseless < threshold
            
        info = {
            "nmse_test" : nmse_test,
            "nmse_test_noiseless" : nmse_test_noiseless,
            "success" : success
        }

        if extra_metric_test is not None:
            if p.invalid:
                m_test = None
                m_test_noiseless = None
            else:
                m_test = metric_test(y_test, y_hat)
                m_test_noiseless = metric_test(y_test_noiseless, y_hat)     

            info.update(
                {
                extra_metric_test : m_test,
                extra_metric_test + '_noiseless' : m_test_noiseless
                }
            )

        return info

    tokens = create_tokens(n_input_var=X_train.shape[1],
                           function_set=function_set,
                           protected=protected)
    library = Library(tokens)

    # create secondary library without the tensors to pass to the controllers
    tens = []
    if enforce_sum:
        tens_idx = []
        for idx, val in enumerate(dataset_info['input']):
            if val in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']:
                tens.append(val)
                tens_idx.append(idx)

    sec_tokens = create_tokens(n_input_var=X_train.shape[1]-len(tens),
                               function_set=function_set,
                               protected=protected)
    sec_library = Library(sec_tokens)


    # # add to function set
    # ad_function_set = function_set + ['n2', 'sum']
    # # create library for automatic differentiation in reverse mode
    # ad_tokens = create_tokens(n_input_var=X_train.shape[1],
    #                        function_set=ad_function_set,
    #                        protected=protected)
    # ad_library = Library(ad_tokens)




    stochastic = reward_noise > 0.0

    extra_info = {}

    task = dsr.task.Task(reward_function=reward,
                         rotate_batch=rotate_batch,
                         data_shuffle=data_shuffle,
                         ad_reverse=reverse_ad,
                         set_ad_traversal=set_ad_traversal,
                         evaluate=evaluate,
                         library=library,
                         sec_library=sec_library,
                         stochastic=stochastic,
                         extra_info=extra_info)

    return task


def make_regression_metric(name, y_train, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.

    invalid_reward: float or None
        Reward value to use for invalid expression. If None, the training
        algorithm must handle it, e.g. by rejecting the sample.

    max_reward: float
        Maximum possible reward under this metric.
    """

    var_y = np.var(y_train)

    all_metrics = {

        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        "neg_mse" :     (lambda y, y_hat : -np.mean((y - y_hat)**2),
                        0),

        # Negative root mean squared error
        # Range: [-inf, 0]
        # Value = -sqrt(var(y)) when y_hat == mean(y)
        "neg_rmse" :     (lambda y, y_hat : -np.sqrt(np.mean((y - y_hat)**2)),
                        0),

        # Negative normalized mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nmse" :    (lambda y, y_hat : -np.mean((y - y_hat)**2)/var_y,
                        0),

        # Negative normalized root mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nrmse" :   (lambda y, y_hat : -np.sqrt(np.mean((y - y_hat)**2)/var_y),
                        0),

        # (Protected) negative log mean squared error
        # Range: [-inf, 0]
        # Value = -log(1 + var(y)) when y_hat == mean(y)
        "neglog_mse" : (lambda y, y_hat : -np.log(1 + np.mean((y - y_hat)**2)),
                        0),

        # (Protected) inverse mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]*var(y)) when y_hat == mean(y)
        "inv_mse" : (lambda y, y_hat : 1/(1 + args[0]*np.mean((y - y_hat)**2)),
                        1),

        # (Protected) inverse normalized mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nmse" :    (lambda y, y_hat : 1/(1 + args[0]*np.mean((y - y_hat)**2)/var_y),
                        1),

        # (Protected) inverse normalized root mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nrmse" :    (lambda y, y_hat : 1/(1 + args[0]*np.sqrt(np.mean((y - y_hat)**2)/var_y)),
                        1),

        # Fraction of predicted points within p0*abs(y) + p1 band of the true value
        # Range: [0, 1]
        "fraction" :    (lambda y, y_hat : np.mean(abs(y - y_hat) < args[0]*abs(y) + args[1]),
                        2),

        # Pearson correlation coefficient
        # Range: [0, 1]
        "pearson" :     (lambda y, y_hat : scipy.stats.pearsonr(y, y_hat)[0],
                        0),

        # regularised custom Squared Mean Average Percentage Error
        # added a minus sign such that the range is similar to the first 5 entries
        # Range: [-inf, 0]
        "reg_mspe" :     (lambda y, y_hat : -np.mean((y-y_hat)**2 /np.sqrt(0.001**2 + y**2)),
                        0),

        # Spearman correlation coefficient
        # Range: [0, 1]
        "spearman" :    (lambda y, y_hat : scipy.stats.spearmanr(y, y_hat)[0],
                        0)
    }

    assert name in all_metrics, "Unrecognized reward function name."
    assert len(args) == all_metrics[name][1], "For {}, expected {} reward function parameters; received {}.".format(name,all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    # For negative MSE-based rewards, invalid reward is the value of the reward function when y_hat = mean(y)
    # For inverse MSE-based rewards, invalid reward is 0.0
    # For non-MSE-based rewards, invalid reward is the minimum value of the reward function's range
    all_invalid_rewards = {
        "neg_mse" : -np.inf,   # used to be -var_y
        "neg_rmse" : -np.sqrt(var_y),
        "neg_nmse" : -1.0,
        "neg_nrmse" : -1.0,
        "neglog_mse" : -np.log(1 + var_y),
        "inv_mse" : 0.0, #1/(1 + args[0]*var_y),
        "inv_nmse" : 0.0, #1/(1 + args[0]),
        "inv_nrmse" : 0.0, #1/(1 + args[0]),
        "fraction" : 0.0,
        "pearson" : 0.0,
        "reg_mspe" : -1000,
        "spearman" : 0.0
    }
    invalid_reward = all_invalid_rewards[name]

    all_max_rewards = {
        "neg_mse" : 0.0,
        "neg_rmse" : 0.0,
        "neg_nmse" : 0.0,
        "neg_nrmse" : 0.0,
        "neglog_mse" : 0.0,
        "inv_mse" : 1.0,
        "inv_nmse" : 1.0,
        "inv_nrmse" : 1.0,
        "fraction" : 1.0,
        "pearson" : 1.0,
        "reg_mspe" : 0.0,  # ??
        "spearman" : 1.0
    }
    max_reward = all_max_rewards[name]

    # when implementing different metric, note the batch rotation needs y in the -2 position
    all_ad_metric_traversals = {
        'mse' : ['div', 'sum', 'n2', 'sub', 'y', 'n'],
        'inv_nrmse' : ['div', 'one', 'add', 'one', 'sqrt', 'div', 'sum', 'n2', 'sub', 'y', 'n_var_y']
    }

    ad_metric_traversal = create_metric_ad_tokens(all_ad_metric_traversals[name], y=y_train)

    return metric, invalid_reward, max_reward, ad_metric_traversal
