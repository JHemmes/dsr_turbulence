"""Parallelized, single-point launch script to run DSR or GP on a set of benchmarks."""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
dsrpath = os.path.abspath(__file__)   # these two lines are to add the dsr dir to path to run it without installing dsr package
sys.path.append(dsrpath[:dsrpath.rfind('dsr')])
import json
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
from dsr.turbulence.dataprocessing import load_frozen_RANS_dataset, load_benchmark_dataset
from dsr.turbulence.resultprocessing import plot_results
from dsr.hyperparameters import create_configs_sensitivity

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
    if config['task']['dataset_info']['name'] in ['PH10595', 'CBFS13700', 'CD12600']:
        plot_results(result, config)
    result.pop("program")

    return result

def main_custom(config_template="config.json",
                mc=1,
                output_filename=None,
                n_cores_task=24,
                seed_shift=1):
    """Now loads custom dataset to run with dsr"""

    """Runs DSR custom datasets using multiprocessing."""

    # Load the config file
    with open(config_template, encoding='utf-8') as f:
        config = json.load(f)

    # set required configs
    config_task = config["task"]      # Task specification parameters
    config_training = config["training"]    # Training hyperparameters

    # Create output directories
    if output_filename is None:
        output_filename = "Results_dsr.csv"
    if config_training["logdir"] == './log':
        config_training["logdir"] = os.path.join(config_training["logdir"],
                                                 "log_{}".format(datetime.now().strftime("%Y-%m-%d-%H%M%S")))
    logdir = config_training["logdir"]

    if 'sensitivity' in logdir:  # delete original config file when performing sensiticity analysis
        os.remove(config_template)

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
    if config_task['dataset']['name'] in ['PH10595', 'CBFS13700', 'CD12600']:
        X, y = load_frozen_RANS_dataset(config_task)
    else:
        print('Using benchmark dataset')
        X, y = load_benchmark_dataset(config_task)

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

def sensitivity_analysis(config_template="config.json",
                         mc=1,
                         n_cores_task=24):

    # Load the base config file
    with open(config_template, encoding='utf-8') as f:
        base_config = json.load(f)

    # set output dir:
    if base_config['task']['enforce_sum']:
        base_logdir = './log/sensitivity/tensor'
        base_config['training']['logdir'] = base_logdir
    else:
        base_logdir = './log/sensitivity/scalar'
        base_config['training']['logdir'] = base_logdir
    os.makedirs(base_logdir, exist_ok=True)

    create_configs_sensitivity(base_config)

    config_list = [x for x in os.listdir(base_logdir) if not os.path.isdir(os.path.join(base_logdir, x))]

    for config in config_list:
        main_custom(config_template=os.path.join(base_logdir, config), mc=mc, n_cores_task=n_cores_task)



if __name__ == "__main__":

    """
    Allowed entries for input and output in the JSON config file are:
     
    scalars:
    k - turbulent kinetic energy
    grad_u_T1, grad_u_T2, etc.  - The tensor product of the base tensors with grad_u
    inv1, inv2 ect. - the 10 base tensor invariants from pope
    kDeficit - the corrective term to the k-equation"
    
    tensors:
    T1, T2, etc. - Base tensors 
    grad_u - velocity gradient
     
    possible entries for the function_set:
    ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "const"]
    
    
    possible entries for dataset: 
    'PH10595', 'CBFS13700', 'CD12600', anything else will result in benchmark being used
    
    """


    main_custom(config_template="config_kDeficit.json", mc=100, n_cores_task=12)
    # main_custom(config_template="config_bDelta.json", mc=100, n_cores_task=12)



    # sensitivity_analysis(config_template="config_kDeficit.json", mc=100, n_cores_task=4)
