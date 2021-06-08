"""Parallelized, single-point launch script to run DSR or GP on a set of benchmarks."""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
import sys
dsrpath = os.path.abspath(__file__)   # these two lines are to add the dsr dir to path to run it without installing dsr package
sys.path.append(dsrpath[:dsrpath.rfind('dsr')])
import json
import time
from datetime import datetime
import multiprocessing
import logging
from functools import partial
from pkg_resources import resource_filename
import zlib
# import pp
import copy

from subprocess import call
import click
import numpy as np
import pandas as pd
import time
from sympy.parsing.sympy_parser import parse_expr
from sympy import srepr

from dsr import DeepSymbolicOptimizer
from dsr.program import Program
from dsr.task.regression.dataset import BenchmarkDataset
from dsr.baselines import gpsr
from dsr.turbulence.dataprocessing import load_frozen_RANS_dataset
from dsr.turbulence.resultprocessing import plot_results


def train_dsr(name_and_seed, config):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    # Override the benchmark name and output file
    name, seed = name_and_seed
    config["task"]["name"] = name
    config["training"]["output_file"] = "dsr_{}_{}.csv".format(name, seed)

    # Try importing TensorFlow (with suppressed warnings), Controller, and learn
    # When parallelizing across tasks, these will already be imported, hence try/except
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        from dsr.controller import Controller
        from dsr.train import learn
    except ModuleNotFoundError: # Specific subclass of ImportError for when module is not found, probably needs to be excepted first
        print("One or more libraries not found")
        raise ModuleNotFoundError
    except ImportError:
        # Have we already imported tf? If so, this is the error we want to dodge. 
        if 'tf' in globals():
            pass
        else:
            raise ImportError

    # Train the model
    model = DeepSymbolicOptimizer(config)
    start = time.time()
    result = {"name" : name, "seed" : seed} # Name and seed are listed first
    result.update(model.train(seed=seed))
    result["t"] = time.time() - start
    plot_results(result, config)
    result.pop("program")

    return result


def eval_expression(expression, X):

    vardict = {}
    for ii in range(X.shape[1]):
        vardict[f'x{ii+1}'] = X[:, ii]
    locals().update(vardict)

    expression = expression.replace('exp', 'np.exp')
    expression = expression.replace('log', 'np.log')
    expression = expression.replace('sin', 'np.sin')
    expression = expression.replace('cos', 'np.cos')

    yhat = eval(expression)

    return yhat

def find_residual(expression, X, y):

    yhat = eval_expression(expression, X)

    residual = y - yhat

    return residual

def find_real_expression(expression, inputs):
    for ii in range(len(inputs)):
        expression = expression.replace(f'x{ii + 1}', inputs[ii])

    return expression

def train_dsr_greedy(name_and_seed, config):
    """Trains DSR and returns dict of reward, expression, and traversal"""

    # Try importing TensorFlow (with suppressed warnings), Controller, and learn
    # When parallelizing across tasks, these will already be imported, hence try/except
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        from dsr.controller import Controller
        from dsr.train import learn
    except ModuleNotFoundError: # Specific subclass of ImportError for when module is not found, probably needs to be excepted first
        print("One or more libraries not found")
        raise ModuleNotFoundError
    except ImportError:
        # Have we already imported tf? If so, this is the error we want to dodge.
        if 'tf' in globals():
            pass
        else:
            raise ImportError

    # Override the benchmark name
    name, seed = name_and_seed
    config["task"]["name"] = name

    base_tensors = []
    non_base_tensors = []
    for index, value in enumerate(config['task']['dataset_info']['input']):
        if value in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']:
            base_tensors.append(index)
        else:
            non_base_tensors.append(index)

    # set temporary config file to only include one tensor
    configtmp = copy.deepcopy(config)
    configtmp['task'].pop('dataset_info')

    y = config['task']['dataset'][1]

    full_results = {}

    for tensor in base_tensors:
        configtmp["training"]["output_file"] = f"dsr_{name}_{seed}_{config['task']['dataset_info']['input'][tensor]}.csv"
        input_idx = [tensor] + non_base_tensors
        X = config['task']['dataset'][0][:, input_idx]
        configtmp['task']['dataset'] = (X,y)

        # train the model for single tensor
        model = DeepSymbolicOptimizer(configtmp)
        start = time.time()
        result = {"name": name, "seed": seed}  # Name and seed are listed first
        result.update(model.train(seed=seed))
        result["t"] = time.time() - start
        result.pop("program")

        # save dsr_results
        full_results[config['task']['dataset_info']['input'][tensor]] = {}
        full_results[config['task']['dataset_info']['input'][tensor]]['dsr_result'] = result
        full_results[config['task']['dataset_info']['input'][tensor]]['dataset'] = (X,y)
        full_results[config['task']['dataset_info']['input'][tensor]]['input_idx'] = input_idx

        # overwrite y with the newly calculated residual
        y = find_residual(result['expression'], X, y)


    return_result = {"name": name, "seed": seed}

    # reset y to original target output
    y = config['task']['dataset'][1]

    # calculate y-hat, find full expression, determine total time
    yhat = np.zeros(y.shape)
    full_expression = ''
    return_result['t'] = 0
    for key in full_results.keys():
        yhat+= eval_expression(full_results[key]['dsr_result']['expression'],full_results[key]['dataset'][0])
        inputs = [config['task']['dataset_info']['input'][ii] for ii in full_results[key]['input_idx']]
        full_expression += find_real_expression(full_results[key]['dsr_result']['expression'], inputs) + ' + '
        return_result['t'] += full_results[key]['dsr_result']['t']

    return_result['expression'] = full_expression[:-3]

    # save errors
    return_result['reg_mspe'] = -np.mean((y-yhat)**2 /np.sqrt(0.001**2 + y**2))
    return_result['NRMSE'] = np.sqrt(np.mean((y-yhat)**2))/np.std(y)

    # append individual results
    for key in full_results.keys():
        return_result[key+'_reward'] = full_results[key]['dsr_result']['r']
        return_result[key+'_expression'] = full_results[key]['dsr_result']['expression']
        return_result[key+'_t'] = full_results[key]['dsr_result']['t']

    return return_result






def train_gp(name_and_seed, logdir, config_task, config_gp):
    """Trains GP and returns dict of reward, expression, and program"""

    name, seed = name_and_seed
    config_gp["seed"] = seed + zlib.adler32(name.encode("utf-8"))

    start = time.time()

    # Load the dataset
    config_dataset = config_task["dataset"]
    config_dataset["name"] = name
    dataset = BenchmarkDataset(**config_dataset)

    # Fit the GP
    gp = gpsr.GP(dataset=dataset, **config_gp)
    p, logbook = gp.train()

    # Retrieve results
    r = base_r = p.fitness.values[0]
    str_p = str(p)
    nmse_test = gp.nmse_test(p)[0]
    nmse_test_noiseless = gp.nmse_test_noiseless(p)[0]
    success = gp.success(p)

    # Many failure cases right now for converting to SymPy expression
    try:
        expression = repr(parse_expr(str_p.replace("X", "x").replace("add", "Add").replace("mul", "Mul")))
    except:
        expression = "N/A"

    # Save run details
    drop = ["gen", "nevals"]
    df_fitness = pd.DataFrame(logbook.chapters["fitness"]).drop(drop, axis=1)
    df_fitness = df_fitness.rename({"avg" : "fit_avg", "min" : "fit_min"}, axis=1)
    df_fitness["fit_best"] = df_fitness["fit_min"].cummin()
    df_len = pd.DataFrame(logbook.chapters["size"]).drop(drop, axis=1)
    df_len = df_len.rename({"avg" : "l_avg"}, axis=1)
    df = pd.concat([df_fitness, df_len], axis=1, sort=False)
    df.to_csv(os.path.join(logdir, "gp_{}_{}.csv".format(name, seed)), index=False)

    result = {
        "name" : name,
        "seed" : seed,
        "r" : r,
        "base_r" : base_r,
        "nmse_test" : nmse_test,
        "nmse_test_noiseless" : nmse_test_noiseless,
        "success" : success,
        "expression" : expression,
        "traversal" : str_p,
        "t" : time.time() - start
    }

    return result



def main_custom(config_template="config.json",
                method="dsr",
                mc=1,
                output_filename=None,
                n_cores_task=24,
                seed_shift=1):
    """Modified by Jasper Hemmes - 2021"""
    """Now loads custom dataset to run with dsr"""

    """Runs DSR or GP on custom datasets using multiprocessing."""

    # Load the config file
    with open(config_template, encoding='utf-8') as f:
        config = json.load(f)


    # set required configs
    config_task = config["task"]      # Task specification parameters
    config_training = config["training"]    # Training hyperparameters


    # Optional configs
    config_controller = config.get("controller")                        # Controller hyperparameters
    config_language_model_prior = config.get("language_model_prior")    # Language model hyperparameters
    config_gp = config.get("gp")                                        # GP hyperparameters

    # Create output directories
    if output_filename is None:
        output_filename = "Results_{}.csv".format(method)
    config_training["logdir"] = os.path.join(
        config_training["logdir"],
        "log_{}".format(datetime.now().strftime("%Y-%m-%d-%H%M%S")))
    logdir = config_training["logdir"]


    os.makedirs(logdir, exist_ok=True)
    output_filename = os.path.join(logdir, output_filename)

    runs = [config_task["name"]]

    # Generate run-seed pairs for each MC. When passed to the TF RNG,
    # seeds will be added to checksums on the benchmark names
    unique_runs = runs.copy()
    runs *= mc
    seeds = (np.arange(mc) + seed_shift).repeat(len(unique_runs)).tolist()
    names_and_seeds = list(zip(runs, seeds))

    # Edit n_cores_task and/or n_cores_batch
    if n_cores_task == -1:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task > len(runs):
        print("Setting 'n_cores_task' to {} for batch because there are only {} runs.".format(len(runs), len(runs)))
        n_cores_task = len(runs)
    if method == "dsr":
        if config_training["verbose"] and n_cores_task > 1:
            print("Setting 'verbose' to False for parallelized run.")
            config_training["verbose"] = False
        if config_training["n_cores_batch"] != 1 and n_cores_task > 1:
            print("Setting 'n_cores_batch' to 1 to avoid nested child processes.")
            config_training["n_cores_batch"] = 1
    print("Running {} for n={} on runs {}".format(method, mc, unique_runs))

    # Write terminal command and config.json into log directory
    cmd_filename = os.path.join(logdir, "cmd.out")
    with open(cmd_filename, 'w') as f:
        print(" ".join(sys.argv), file=f)
    config_filename = os.path.join(logdir, "config.json")
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)

    # load dataset and overwrite config
    # (needs to happen after the config is written to the logdir, because dataset is not JSON serialisable)
    X, y = load_frozen_RANS_dataset(config_task)

    # Random data
    # np.random.seed(0)
    # X = np.random.random((10, 6))
    # # y = np.exp(X[:, 1]) + X[:, 2] ** 2 - 2.5 * X[:, 0] + X[:, 1] * X[:, 4] + X[:, 3]
    # y = np.exp(X[:, 1]) + X[:, 2] ** 2 - 2.5 * X[:, 0]

    # disabled the greedy algorithm
    # # if the greedy algorithm the information on the dataset must be remembered
    # greedy = False
    # if config_task['dataset']['output'] == 'bDelta':
    #     greedy = True
    #     config['task']['dataset_info'] = config['task']['dataset']

    config["task"]["dataset_info"] = config["task"]["dataset"] # save dataset information for later use
    config["task"]["dataset"] = (X, y)
    config_task = config["task"]      # Set config task again after overwriting dataset

    # Define the work
    if method == "dsr":
        work = partial(train_dsr, config=config)
    elif method == "gp":
        work = partial(train_gp, logdir=logdir, config_task=config_task, config_gp=config_gp)

    # Farm out the work
    write_header = True
    if n_cores_task > 1:
        # logger = multiprocessing.log_to_stderr()
        # logger.setLevel(multiprocessing.SUBDEBUG)
        # logger.setLevel(logging.INFO)

        pool = multiprocessing.Pool(processes=n_cores_task)
        # pool = multiprocessing.Pool(processes=n_cores_task, maxtasksperchild=1) # ?? Jasper Hemmes added maxtasksperchild, didnt fix libgomp
        # for result in pool.imap_unordered(work, names_and_seeds, int(np.ceil(mc/n_cores_task))): Adding chunksize did seem to improve the number of failed runs, however whole chunk needs to complete before the process is done.
        for result in pool.imap_unordered(work, names_and_seeds):
            pd.DataFrame(result, index=[0]).to_csv(output_filename, header=write_header, mode='a', index=False)
            print("Completed {} ({} of {}) in {:.0f} s".format(result["name"], result["seed"]+1-seed_shift, mc, result["t"]))
            write_header = False
    else:
        for name_and_seed in names_and_seeds:
            result = work(name_and_seed)
            pd.DataFrame(result, index=[0]).to_csv(output_filename, header=write_header, mode='a', index=False)
            write_header = False

    print("Results saved to: {}".format(output_filename))



if __name__ == "__main__":

    """Note, allowed entries for input and output in the JSON config file are:
    
    scalars:
    k - turbulent kinetic energy
    grad_u_T1, grad_u_T2, etc.  - The tensor product of the base tensors with grad_u
    inv1, inv2 ect. - the 10 base tensor invariants from pope
    kDeficit - the corrective term to the k-equation"
    
    tensors:
    T1, T2, etc. - Base tensors 
    grad_u - velocity gradient
     
    possible entries for the function_set:"  # ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "const"]"""

    main_custom(config_template="config_kDeficit.json", mc=1, n_cores_task=1)
    # main_custom(config_template="config_bDelta.json", mc=100, n_cores_task=4)

    # main_custom(config_template="config_bDelta.json", mc=100, n_cores_task=8)


