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
    # disabled plotting results since that doesnt work with the bechmarks
    # plot_results(result, config)
    result.pop("program")

    return result

def main_custom(config_template="config.json",
                mc=1,
                output_filename=None,
                n_cores_task=24,
                seed_shift=1,
                benchmark='nguyen-1'):
    """Modified by Jasper Hemmes - 2021"""
    """Now loads custom dataset to run with dsr"""

    """Runs DSR custom datasets using multiprocessing."""

    # Load the config file
    with open(config_template, encoding='utf-8') as f:
        config = json.load(f)

    # overwrite task name with benchmark
    config['task']['name'] = benchmark

    # set required configs
    config_task = config["task"]      # Task specification parameters
    config_training = config["training"]    # Training hyperparameters


    # Optional configs
    config_controller = config.get("controller")                        # Controller hyperparameters
    config_language_model_prior = config.get("language_model_prior")    # Language model hyperparameters
    config_gp = config.get("gp")                                        # GP hyperparameters

    # Create output directories
    if output_filename is None:
        output_filename = "Results_dsr.csv"
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

    if config_training["verbose"] and n_cores_task > 1:
        print("Setting 'verbose' to False for parallelized run.")
        config_training["verbose"] = False
    if config_training["n_cores_batch"] != 1 and n_cores_task > 1:
        print("Setting 'n_cores_batch' to 1 to avoid nested child processes.")
        config_training["n_cores_batch"] = 1
    print("Running dsr for n={} on runs {}".format(mc, unique_runs))

    # Write terminal command and config.json into log directory
    cmd_filename = os.path.join(logdir, "cmd.out")
    with open(cmd_filename, 'w') as f:
        print(" ".join(sys.argv), file=f)
    config_filename = os.path.join(logdir, "config.json")
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)

    # load dataset and overwrite config
    # (needs to happen after the config is written to the logdir, because dataset is not JSON serialisable)
    # X, y = load_frozen_RANS_dataset(config_task)

    np.random.seed(0)
    # generate data dependent on config:
    if benchmark == 'nguyen-1':
        X = np.random.uniform(-1, 1, (20, 1))
        y = X[:, 0]**3 + X[:, 0]**2 + X[:, 0]
    elif benchmark == 'nguyen-2':
        X = np.random.uniform(-1, 1, (20, 1))
        y = X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]
    elif benchmark == 'nguyen-3':
        X = np.random.uniform(-1, 1, (20, 1))
        y = X[:, 0]**5 + X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]
    elif benchmark == 'nguyen-4':
        X = np.random.uniform(-1, 1, (20, 1))
        y = X[:, 0]**6 + X[:, 0]**5 + X[:, 0]**4 + X[:, 0]**3 + X[:, 0]**2 + X[:, 0]
    elif benchmark == 'nguyen-5':
        X = np.random.uniform(-1, 1, (20, 1))
        y = np.sin(X[:, 0]**2) * np.cos(X[:, 0]) - 1
    elif benchmark == 'nguyen-6':
        X = np.random.uniform(-1, 1, (20, 1))
        y = np.sin(X[:, 0]) + np.sin(X[:, 0] + X[:, 0]**2)
    elif benchmark == 'nguyen-7':
        X = np.random.uniform(0, 2, (20, 1))
        y = np.log(X[:, 0] + 1) + np.log(X[:, 0]**2 + 1)
    elif benchmark == 'nguyen-8':
        X = np.random.uniform(0, 4, (20, 1))
        y = np.sqrt(X[:, 0])
    elif benchmark == 'nguyen-9':
        X = np.random.uniform(0, 1, (20, 2))
        y = np.sin(X[:, 0]) + np.sin(X[:, 1]**2)
    elif benchmark == 'nguyen-10':
        X = np.random.uniform(0, 1, (20, 2))
        y = 2*np.sin(X[:, 0]) * np.cos(X[:, 1])
    elif benchmark == 'nguyen-11':
        X = np.random.uniform(0, 1, (20, 2))
        y = X[:, 0]**X[:, 1]
    elif benchmark == 'nguyen-12':
        X = np.random.uniform(0, 1, (20, 2))
        y = X[:, 0]**4 - X[:, 0]**3 + 0.5*X[:, 1]**2 - X[:, 1]

    config["task"]["dataset_info"] = config["task"]["dataset"] # save dataset information for later use
    config["task"]["dataset"] = (X, y)
    config_task = config["task"]      # Set config task again after overwriting dataset

    # Define the work
    work = partial(train_dsr, config=config)

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


    benchmarks =     ['nguyen-1', 'nguyen-2', 'nguyen-3', 'nguyen-4',
                      'nguyen-5', 'nguyen-6', 'nguyen-7', 'nguyen-8',
                      'nguyen-9', 'nguyen-10', 'nguyen-11', 'nguyen-12']

    for benchmark in benchmarks:
        main_custom(config_template="config.json", mc=100, n_cores_task=2, benchmark=benchmark)
    # main_custom(config_template="config_bDelta.json", mc=100, n_cores_task=1)

    # main_custom(config_template="config_bDelta.json", mc=100, n_cores_task=8)

