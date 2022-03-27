"""Defines main training loop for deep symbolic regression."""

import os
import multiprocessing
from itertools import compress
from itertools import combinations
from datetime import datetime
from collections import defaultdict
from copy import copy
import time
from copy import copy

import tensorflow as tf
import pandas as pd
import numpy as np

from dsr.program import Program, from_tokens
from dsr.utils import empirical_entropy, setup_output_files, test_fixed_actions, plot_prob_dists
from dsr.memory import Batch, make_queue

# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set TensorFlow seed
tf.random.set_random_seed(0)


# Work for multiprocessing pool: optimize constants and compute reward
def work(p):
    optimized_constants = p.optimize
    return optimized_constants, p.base_r

def hof_work(p):

    return [p.r, p.base_r, p.count, repr(p.sympy_expr), repr(p), p.evaluate]

def learn(session, controller, pool, tensor_dsr, dataset_name,
          logdir="./log", n_epochs=None, n_samples=1e6,
          batch_size=1000, dataset_batch_size=10, complexity="length", complexity_weight=0.001,
          const_optimizer="minimize", const_params=None,
          epsilon=0.01, n_cores_batch=1, verbose=True,
          output_file=None, baseline=0.5,
          b_jumpstart=True, early_stopping=False,
          t_lim=1000, hof=10,
          optim_opt_full={'maxiter': 100, 'gtol': 1e-5}, optim_opt_sub={'maxiter': 2000, 'gtol': 1e-5},
          save_batch=False, save_controller=False, eval_all=False):
    """
    Executes the main training loop.

    Parameters
    ----------
    sess : tf.Session
        TenorFlow Session object.

    controller : dsr.controller.Controller
        Controller object used to generate Programs.

    pool : multiprocessing.Pool or None
        Pool to parallelize reward computation. For the control task, each
        worker should have its own TensorFlow model. If None, a Pool will be
        generated if n_cores_batch > 1.

    logdir : str, optional
        Name of log directory.

    n_epochs : int or None, optional
        Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).

    batch_size : int, optional
        Number of sampled expressions per epoch.

    complexity : str, optional
        Complexity penalty name.

    complexity_weight : float, optional
        Coefficient for complexity penalty.

    const_optimizer : str or None, optional
        Name of constant optimizer.

    const_params : dict, optional
        Dict of constant optimizer kwargs.

    epsilon : float or None, optional
        Fraction of top expressions used for training. None (or
        equivalently, 1.0) turns off risk-seeking.

    n_cores_batch : int, optional
        Number of cores to spread out over the batch for constant optimization
        and evaluating reward. If -1, uses multiprocessing.cpu_count().

    verbose : bool, optional
        Whether to print progress.

    summary : bool, optional
        Whether to write TensorFlow summaries.

    output_file : str, optional
        Filename to write results for each iteration.

    save_all_r : bool, optional
        Whether to save all rewards for each iteration.

    baseline : str, optional
        Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
        Choices:
        (1) "ewma_R" : b = EWMA(<R>)
        (2) "R_e" : b = R_e
        (3) "ewma_R_e" : b = EWMA(R_e)
        (4) "combined" : b = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the (1-epsilon)-quantile estimate.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional
        Whether to stop early if stopping criteria is reached.

    t_lim : int, optional
        Time limit that stops the training after the limit, value given in hours wall clock time

    hof : int or None, optional
        If not None, number of top Programs to evaluate after training.

    optim_opt_full: dict, optional
        dict containing options controlling constant optimisation for the full batch
        maxiter sets iteration limit
        gtol sets gradient tolerance when optimisation stops

    optim_opt_sub: dict, optional
        dict containing options controlling constant optimisation for the sub-batch
        maxiter sets iteration limit
        gtol sets gradient tolerance when optimisation stops

    save_batch : bool, optional
        Determines whether the batches that provided a new best are saved to a pickle file

    eval_all : bool, optional
        If True, evaluate all Programs. While expensive, this is useful for
        noisy data when you can't be certain of success solely based on reward.
        If False, only the top Program is evaluated each iteration.

    pareto_front : bool, optional
        If True, compute and save the Pareto front at the end of training.

    debug : int, optional
        Debug level, also passed to Controller. 0: No debug. 1: Print initial
        parameter means. 2: Print parameter means each step.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by base_r).
    """

    # Config assertions and warnings
    assert n_samples is None or n_epochs is None, "At least one of 'n_samples' or 'n_epochs' must be None."

    #Create dummy program to find names of tokens in library
    tmp_program = from_tokens(np.array([0]), optimize=False, skip_cache=True)
    token_names = tmp_program.library.names
    if dataset_batch_size:
        if not tensor_dsr:
            tmp_program.task.data_shuffle(int(output_file.split('.')[0].split('_')[-1]))
        tmp_program.task.rotate_batch(dataset_batch_size)
    del tmp_program

    # Create log files and dirs
    hof_output_file, batch_dir, controller_dir = setup_output_files(logdir, output_file, token_names,
                                                                    save_batch, save_controller)

    # Set the complexity functions
    Program.set_complexity_penalty(complexity, complexity_weight)

    # Set the constant optimizer
    const_params = const_params if const_params is not None else {}
    Program.set_const_optimizer(const_optimizer, **const_params)

    if save_batch:
        from utils import save_pickle

    # Create the pool of workers, if pool is not already given
    if pool is None:
        if n_cores_batch == -1:
            n_cores_batch = multiprocessing.cpu_count()
        if n_cores_batch > 1:
            pool = multiprocessing.Pool(n_cores_batch)

    base_r_history = None

    # Main training loop
    p_final = None
    base_r_best = -np.inf
    r_best = -np.inf
    r_max_full = 0
    r_best_full = 0
    r_max_PH = 0
    r_max_CD = 0
    r_max_CBFS = 0
    r_max_PH_NW = 0
    r_max_CD_NW = 0
    r_max_CBFS_NW = 0

    loss_pg = 1
    prev_r_best = None
    prev_base_r_best = None
    ewma = None if b_jumpstart else 0.0 # EWMA portion of baseline
    n_epochs = n_epochs if n_epochs is not None else int(n_samples / batch_size)

    # wall clock time limit:
    program_start = time.process_time()  # start time for
    wall_clock_start = time.time()
    t_lim_seconds = t_lim*3600
    print(f"Time limit set to {t_lim}h")

    for step in range(n_epochs):
        # if step%100 == 0:
        #     plot_prob_dists(controllers, step, token_names)

        # if output_file
        # this can be used to test performance on fixed set of actions:
        # if step == 0 and int(output_file.split('.')[0].split('_')[-1]) == 1:
        #     test_fixed_actions(logdir, from_tokens)

        proc_start = time.process_time()
        wall_start = time.time()
        # Set of str representations for all Programs ever seen
        s_history = set(Program.cache.keys())

        actions, obs, priors = controller.sample(batch_size)

        # Choose to stack actions or not, if there is one controller they should not be stacked
        if tensor_dsr:
            all_means = []
            for a, b in combinations(actions, 2):
                all_means.append(np.mean(a==b))
            sample_metric = np.mean(all_means)

            actions = np.stack(actions, axis=-1)
            actions_original = copy(actions)

        else:
            actions = actions
            obs = obs
            priors = priors

            sample_metric = 1  # Dummy value

        programs = [from_tokens(a, optimize=optim_opt_full) for a in actions]

        # Retrieve metrics
        base_r = np.array([p.base_r for p in programs])
        r = np.array([p.r for p in programs])

        if any(np.isnan(base_r)):
            # if the const optimisation returns nan constants, the rewards is nan, that is set to min reward here.
            base_r[np.where(np.isnan(base_r))[0]] = min(base_r)
            r[np.where(np.isnan(r))[0]] = min(r)

        if eval_all:
            success = [p.evaluate.get("success") for p in programs]
            # Check for success before risk-seeking, but don't break until after
            if any(success):
                p_final = programs[success.index(True)]

        # Update reward history
        if base_r_history is not None:
            for p in programs:
                key = p.str
                if key in base_r_history:
                    base_r_history[key].append(p.base_r)
                else:
                    base_r_history[key] = [p.base_r]

        # Collect full-batch statistics
        l = np.array([len(p.traversal) for p in programs])
        s = [p.str for p in programs] # Str representations of Programs
        invalid = np.array([p.invalid for p in programs])
        nfev = np.array([p.nfev for p in programs])
        n_consts = np.array([len(p.const_pos) for p in programs])
        token_occur = np.array([p.token_occurences for p in programs])
        n_unq_tokens = np.array([p.n_unique_tokens for p in programs])

        r_avg_full = np.mean(r)
        l_avg_full = np.mean(l)
        a_ent_full = np.mean(np.apply_along_axis(empirical_entropy, 0, actions))
        base_r_avg_full = np.mean(base_r)
        n_unique_full = len(set(s))
        n_novel_full = len(set(s).difference(s_history))
        invalid_avg_full = np.mean(invalid.clip(0,1))
        eq_w_const_full = np.mean(n_consts > 0)
        n_const_per_eq_full = np.mean(n_consts)
        nfev_avg_full = np.mean(nfev[nfev > 1])
        nit_avg_full = np.mean([p.nit for p in programs if p.nit > 0])
        token_occur_avg_full = np.mean(token_occur, axis=0)
        n_unq_tokens_avg_full = np.mean(n_unq_tokens)

        # Risk-seeking policy gradient: train on top epsilon fraction of samples
        if epsilon is not None and epsilon < 1.0:
            if any(np.isinf(r)): # code added to remove infinite rewards that mess up the quantile calculation.
                min_noinf = min(r[~np.isinf(r)])
                r[np.isinf(r)] = min_noinf
            quantile = np.nanquantile(r, 1 - epsilon, interpolation="higher")
            keep = base_r >= quantile

        # Redo the optimisation "without" limit only for programs in the top quantile
        for p in list(compress(programs, keep)):
            p.top_quantile = 1  # used in tensorflow to distinguish what programs are in the sub batch.
            p.optimize(optim_opt=optim_opt_sub)
            p.update_rewards()

        # memory heavy traversal no longer needed, replace by lighter version
        for p in programs:
            p.replace_traversal()

        # update base_r and r after new optimisation
        base_r = np.array([p.base_r for p in programs])
        r = np.array([p.r for p in programs])

        if any(np.isnan(base_r)):
            # if the const optimisation returns nan constants, the rewards is nan, that is set to min reward here.
            base_r[np.where(np.isnan(base_r))[0]] = np.nanmin(base_r)
            r[np.where(np.isnan(r))[0]] = np.nanmin(r)

        # Collect newly optimised sub batch statistics
        base_r_avg_sub = np.mean(base_r[keep])
        r_avg_sub = np.mean(r[keep])
        l_avg_sub = np.mean(l[keep])
        a_ent_sub = np.mean(np.apply_along_axis(empirical_entropy, 0, actions[keep]))
        n_unique_sub = len(set(list(compress(s, keep))))
        n_novel_sub = len(set(list(compress(s, keep))).difference(s_history))
        invalid_avg_sub = np.mean(invalid[keep].clip(0,1))
        eq_w_const_sub = np.mean(n_consts[keep] > 0)
        n_const_per_eq_sub = np.mean(n_consts[keep])
        nfev_avg_sub = np.mean([p.nfev for p in programs if (p.nfev > 0) and (p.top_quantile == 1)])
        nit_avg_sub = np.mean([p.nit for p in programs if (p.nit > 0) and (p.top_quantile == 1)])
        token_occur_avg_sub = np.mean(token_occur[keep], axis=0)
        n_unq_tokens_avg_sub = np.mean(n_unq_tokens[keep])

        # Check if there is a new best performer
        base_r_max = max(base_r)
        base_r_best = max(base_r_max, base_r_best)
        r_max = max(r)
        r_best = max(r_max, r_best)

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)

        # val_list = []
        # for controller in controllers:
        #     with controller.sess.graph.as_default():
        #         train_vars = tf.trainable_variables()
        #         var_names = [v.name for v in train_vars]
        #         values = controller.sess.run(var_names)
        #         val_list.append(values)


        # collect program information for training:
        top_quantile = np.array([p.top_quantile for p in programs])
        invalid = invalid.astype(float)

        if tensor_dsr:
            # Compute sequence lengths (here I have used the lenghts of individual functions g samples by each)
            # Set up lists as expected by controller
            actions_lst = [actions_original[:, :, ii] for ii in range(actions.shape[-1])]
            lengths_lst = []
            for ii in range(actions.shape[-1]):
                if ii == 0:
                    lengths_lst.append(np.array([min(len(p.tokens[np.where(p.tokens == ii)[0][0] + 1:]),
                                                     controller.max_length) for p in programs], dtype=np.int32))
                else:
                    lengths_lst.append(np.array([min(len(p.tokens[np.where(p.tokens == ii)[0][0] + 1:
                                                         np.where(p.tokens == ii - 1)[0][0] - min(ii, 2)]),
                                                     controller.max_length) for p in programs], dtype=np.int32))

            lengths_lst.append(np.amax(np.stack(lengths_lst, axis=-1), axis=1))

            # Create the Batch
            sampled_batch = Batch(actions=actions_lst, obs=obs, priors=priors,
                                  lengths=lengths_lst, rewards=r, top_quantile=top_quantile, invalid=invalid)

        else:
            lengths = np.array([min(len(p.traversal), controller.max_length)
                                for p in programs], dtype=np.int32)

            # Create the Batch
            sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                                  lengths=lengths, rewards=r, top_quantile=top_quantile, invalid=invalid)

        if save_batch:
            if prev_r_best is None or r_max > prev_r_best or step%50 == 0:
                batch_filename = os.path.join(batch_dir, f'step_{step}.p')
                save_pickle(batch_filename, sampled_batch)

        # failsafe to ensure positive PG-loss
        b_train = min(baseline, quantile)

        # Train the controller
        loss_ent, loss_inv, loss_pg = controller.train_step(b_train, loss_pg, sampled_batch)

        # Assess best program on full dataset for output file
        if dataset_batch_size:
            Program.task.rotate_batch(None)

            # set best program of batch
            p_max = programs[np.argmax(r)]

            # reoptimise constants for full dataset
            p_max.optimize(optim_opt=optim_opt_sub)
            p_max.update_rewards()
            p_max.full_set = True
            r_max_full = p_max.r

            # rotate batch to select next batch for training
            Program.task.rotate_batch(dataset_batch_size)

            # update new best program
            new_r_best = False
            new_base_r_best = False

            if r_best_full == 0 or r_max_full > r_best_full:
                # note, if there will be a difference between base_r and r in the future this needs revising
                r_best_full = r_max_full

                new_r_best = True
                p_r_best = p_max

                new_base_r_best = True
                p_base_r_best = p_max

            prev_r_best = r_best
            prev_base_r_best = base_r_best

        else:

            # update new best program
            new_r_best = False
            new_base_r_best = False

            if prev_r_best is None or r_max > prev_r_best:
                new_r_best = True
                p_r_best = programs[np.argmax(r)]

            if prev_base_r_best is None or base_r_max > prev_base_r_best:
                new_base_r_best = True
                p_base_r_best = programs[np.argmax(base_r)]

            prev_r_best = r_best
            prev_base_r_best = base_r_best


        if dataset_name in ['PH10595', 'CBFS13700', 'CD12600']:
            # when using turbulence data, asses programs on all cases

            p_max = programs[np.argmax(r)]

            Program.task.rotate_batch(None, data_set='PH')
            r_max_PH = p_max.task.reward_function(p_max)

            Program.task.rotate_batch(None, data_set='CD')
            r_max_CD = p_max.task.reward_function(p_max)

            Program.task.rotate_batch(None, data_set='CBFS')
            r_max_CBFS = p_max.task.reward_function(p_max)
            #
            # Program.task.rotate_batch(None, data_set='PH_NW')
            # r_max_PH_NW = p_max.task.reward_function(p_max)
            #
            # Program.task.rotate_batch(None, data_set='CD_NW')
            # r_max_CD_NW = p_max.task.reward_function(p_max)
            #
            # Program.task.rotate_batch(None, data_set='CBFS_NW')
            # r_max_CBFS_NW = p_max.task.reward_function(p_max)

            Program.task.rotate_batch(dataset_batch_size)

            # test = np.array([ 9,  9, 11,  1,  0,  7,  7,  1,  5, 13])
            # p_test = from_tokens(test, optimize=optim_opt_sub, skip_cache=True)
            # p_test.traversal[-1].value[0] = 0.054435153499558075
            # print(p_test.task.reward_function(p_test))

        if output_file is not None:
            proc_duration = time.process_time() - proc_start
            wall_duration = time.time() - wall_start
            # If the outputted stats are changed dont forget to change the column names in utils
            stats = [[
                         base_r_best,
                         r_best_full,
                         r_max_full,
                         base_r_max,
                         base_r_avg_full,
                         base_r_avg_sub,
                         r_best,
                         r_max,
                         programs[np.argmax(r)].sympy_expr,
                         r_max_PH,
                         r_max_CD,
                         r_max_CBFS,
                         r_max_PH_NW,
                         r_max_CD_NW,
                         r_max_CBFS_NW,
                         r_avg_full,
                         r_avg_sub,
                         l_avg_full,
                         l_avg_sub,
                         ewma,
                         loss_pg,  # avg_pg_loss,
                         loss_inv,  # avg_inv_loss,
                         loss_ent,  # avg_ent_loss,
                         n_unique_full,
                         n_unique_sub,
                         n_novel_full,
                         n_novel_sub,
                         np.round(a_ent_full, 2),
                         np.round(a_ent_sub, 2),
                         invalid_avg_full,
                         invalid_avg_sub,
                         np.round(sample_metric, 3),
                         np.round(nfev_avg_full, 3),
                         np.round(nfev_avg_sub, 3),
                         np.round(nit_avg_full, 3),
                         np.round(nit_avg_sub, 3),
                         np.round(eq_w_const_full, 3),
                         np.round(eq_w_const_sub, 3),
                         np.round(n_const_per_eq_full, 3),
                         np.round(n_const_per_eq_sub, 3),
                         np.round(n_unq_tokens_avg_full, 3),
                         np.round(n_unq_tokens_avg_sub, 3),
                         np.round(wall_duration, 2),
                         np.round(proc_duration, 2)
                         ]]  # changed this array to a list, changed save routine to pandas to allow expression string
            stats[0] += list(np.round(token_occur_avg_full, 2))
            stats[0] += list(np.round(token_occur_avg_sub, 2))
            df_append = pd.DataFrame(stats)
            df_append.to_csv(os.path.join(logdir, output_file), mode='a', header=False, index=False)

        # Print new best expression
        if verbose:
            if new_r_best and new_base_r_best:
                if p_r_best == p_base_r_best:
                    print("\nNew best overall")
                    p_r_best.print_stats()
                else:
                    print("\nNew best reward")
                    p_r_best.print_stats()
                    print("...and new best base reward")
                    p_base_r_best.print_stats()

            elif new_r_best:
                print("\nNew best reward")
                p_r_best.print_stats()

            elif new_base_r_best:
                print("\nNew best base reward")
                p_base_r_best.print_stats()

        # Stop if early stopping criteria is met
        if eval_all and any(success):
            # all_r = all_r[:(step + 1)]  # all_r is no longer saved to reduce memory
            print("Early stopping criteria met; breaking early.")
            break
        if early_stopping and p_base_r_best.evaluate.get("success"):
            # all_r = all_r[:(step + 1)]  # all_r is no longer saved to reduce memory
            print("Early stopping criteria met; breaking early.")
            break

        if verbose and step > 0 and step % 10 == 0:
            print("Completed {} steps".format(step))

        if len(Program.cache) > 5000:
            # if the cache contains more than x programs, tidy cache.
            Program.tidy_cache(hof)

        if (time.process_time() - program_start) > t_lim_seconds:
            # if the wall clock time exceeds time limit, save controller and stop iterations
            print(f"Time limit of {t_lim}h exceeded; breaking early")
            break

        if (time.time() - wall_clock_start) > 71*3600:
            # when running on cluster, copy files after 71H of wall_clock time
            command = f'cp -r {os.path.join(os.getcwd(), "log")} /home/jghemmes/'
            try:
                os.system(command)
            except:
                print('Copying files failed, ignore this message if you are not running on cluster.')

    # Save the hall of fame
    if hof is not None and hof > 0:

        Program.task.rotate_batch(None)

        Program.tidy_cache(batch_size)
        programs = list(Program.cache.values())  # unique Programs in cache

        Program.clear_cache()

        for p in programs:
            p.optimize(optim_opt=optim_opt_sub)
            p.update_rewards()
            # # overwrite cached base_r
            # if (p.base_r != p.ad_r) and p.ad_r is not None:
            #     p.base_r = p.ad_r
            # else:
            #     p.base_r = p.task.reward_function(p)
            #
            # # overwrite cached r
            # p.r = p.base_r - p.complexity

        base_r = np.array([p.base_r for p in programs])

        if any(np.isnan(base_r)):
            # if the const optimisation returns nan constants, the rewards is nan, that is set to min reward here.
            base_r[np.where(np.isnan(base_r))[0]] = np.nanmin(base_r)
            # r[np.where(np.isnan(r))[0]] = np.nanmin(r)

        i_hof = np.argsort(base_r)[-hof:][::-1] # Indices of top hof Programs
        hof = [programs[i] for i in i_hof]

        if verbose:
            print("Evaluating the hall of fame...")
        if pool is not None:
            results = pool.map(hof_work, hof)
        else:
            results = list(map(hof_work, hof))

        eval_keys = list(results[0][-1].keys())
        columns = ["r", "base_r", "count", "expression", "traversal"] + eval_keys
        hof_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
        df = pd.DataFrame(hof_results, columns=columns)
        if hof_output_file is not None:
            print("Saving Hall of Fame to {}".format(hof_output_file))
            df.to_csv(hof_output_file, header=True, index=False)

        p_final = programs[np.argmax(base_r)]

    # save tensorflow checkpoint of network state
    if save_controller:
        controller_file = os.path.join(controller_dir, 'controller.ckpt')
        print("Saving Controller Checkpoint to {}".format(controller_file))
        controller.save(controller_file)

    # Close the pool
    if pool is not None:
        pool.close()

    # Return statistics of best Program
    p = p_final if p_final is not None else p_base_r_best

    result = {
        "r" : p.r,
        "base_r" : p.base_r,
    }
    result.update(p.evaluate)
    result.update({
        "expression" : repr(p.sympy_expr),
        "traversal" : repr(p),
        "program" : p
    })

    return result
