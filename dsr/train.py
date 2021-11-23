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
from dsr.utils import empirical_entropy, is_pareto_efficient, setup_output_files, test_fixed_actions, plot_prob_dists
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

def pf_work(p):
    return [p.complexity_eureqa, p.r, p.base_r, p.count, repr(p.sympy_expr), repr(p), p.evaluate]

def learn(session, controller, pool, tensor_dsr,
          logdir="./log", n_epochs=None, n_samples=1e6,
          batch_size=1000, complexity="length", complexity_weight=0.001,
          const_optimizer="minimize", const_params=None, alpha=0.1,
          epsilon=0.01, n_cores_batch=1, verbose=True, summary=True,
          output_file=None, save_all_r=False, baseline="ewma_R",
          b_jumpstart=True, early_stopping=False, hof=10, save_batch=False, eval_all=False,
          pareto_front=False, debug=0):
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

    alpha : float, optional
        Coefficient of exponentially-weighted moving average of baseline.

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

    hof : int or None, optional
        If not None, number of top Programs to evaluate after training.

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

    # ?? disabled when changed to multiple sessions
    # # Create the summary writer
    # if summary:
    #     timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    #     summary_dir = os.path.join("summary", timestamp)
    #     for sess in sessions: # doenst work properly for multiple sessions
    #         writer = tf.summary.FileWriter(summary_dir, sess.graph)

    #Create dummy program to find names of tokens in library
    tmp_program = from_tokens(np.array([0]), optimize=False, skip_cache=True)
    token_names = tmp_program.library.names
    del tmp_program

    # Create log file
    if output_file is not None:
        all_r_output_file, hof_output_file, pf_output_file = setup_output_files(logdir, output_file, token_names)
    else:
        all_r_output_file = hof_output_file = pf_output_file = None

    # Set the complexity functions
    Program.set_complexity_penalty(complexity, complexity_weight)

    # Set the constant optimizer
    const_params = const_params if const_params is not None else {}
    Program.set_const_optimizer(const_optimizer, **const_params)

    if save_batch:
        # if batches are saved import pickle, set up output folder and define save_pickle function
        import pickle
        pickle_dir = os.path.join(logdir, 'pickled_batches')
        os.mkdir(pickle_dir)
        def save_pickle(path, data):
            # function saves data
            with open(path, 'wb') as f:
                pickle.dump(data, f)

    # ?? disabled when changed to multiple sessions
    # if debug:
    #     tvars = tf.trainable_variables()
    #     def print_var_means():
    #         tvars_vals = sess.run(tvars)
    #         for var, val in zip(tvars, tvars_vals):
    #             print(var.name, "mean:", val.mean(),"var:", val.var())

    # Create the pool of workers, if pool is not already given
    if pool is None:
        if n_cores_batch == -1:
            n_cores_batch = multiprocessing.cpu_count()
        if n_cores_batch > 1:
            pool = multiprocessing.Pool(n_cores_batch)


    # ?? disabled when changed to multiple sessions
    priority_queue = None
    # Create the priority queue
    # k = controller.pqt_k
    # if controller.pqt and k is not None and k > 0:
    #     priority_queue = make_queue(priority=True, capacity=k)
    # else:
    #     priority_queue = None

    # ?? disabled when changed to multiple sessions
    # if debug >= 1:
    #     print("\nInitial parameter means:")
    #     print_var_means()

    base_r_history = None

    # Main training loop
    p_final = None
    base_r_best = -np.inf
    r_best = -np.inf
    prev_r_best = None
    prev_base_r_best = None
    ewma = None if b_jumpstart else 0.0 # EWMA portion of baseline
    n_epochs = n_epochs if n_epochs is not None else int(n_samples / batch_size)
    # all_r = np.zeros(shape=(n_epochs, batch_size), dtype=np.float32)

    for step in range(n_epochs):
        #
        # if step%100 == 0:
        #     plot_prob_dists(controllers, step, token_names)

        # if output_file
        # this can be used to test performance on fixed set of actions:
        # if step == 0 and int(output_file.split('.')[0].split('_')[-1]) == 1:
        #     test_fixed_actions(logdir, from_tokens)

        start_time = time.process_time()
        # Set of str representations for all Programs ever seen
        s_history = set(Program.cache.keys())

        # Sample batch of expressions from controller
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: [(batch_size, max_length)] * 3
        # Shape of priors: (batch_size, max_length, n_choices)

        # raw_actions = []

        actions, obs, priors = controller.sample(batch_size)

        # disabled for now since initialised with different seeds:

        # raw_actions.append(action)
        # if tensor_dsr:
        #     # for tensor_dsr actions need to be shuffled if controllers are initialised with the same seed.
        #     shuffler = np.random.permutation(batch_size)
        #     action = action[shuffler]
        #     ob = [item[shuffler] for item in ob]
        #     prior = prior[shuffler]

        # Choose to stack actions or not, if there is one controller they should not be stacked
        # Also calculate percentage of how much the sampled actions differ if there are multiple RNNs:
        if tensor_dsr:
            all_means = []
            for a, b in combinations(actions, 2):
                all_means.append(np.mean(a==b))
            sample_metric = np.mean(all_means)

            actions = np.stack(actions, axis=-1)
            # obs = np.stack(obs, axis=-1)
            # priors = np.stack(priors, axis=-1)


            # sample_metric = 1  # Dummy value, disabled for now since raw_actions is disabled

            actions_original = copy(actions)

        else:
            actions = actions
            obs = obs
            priors = priors

            sample_metric = 1  # Dummy value

        # import pickle
        # def load_pickle(path):
        #     # function loads saved charge points
        #     with open(path, 'rb') as f:
        #         data = pickle.load(f)
        #     return data
        #
        # actions = [np.array([ 9,  1,  9,  9,  7,  9,  0,  2,  9, 12,  9,  0, 11,  3,  9,  9,  0,
        #         7,  8, 11, 10, 13, 12,  0,  1, 13, 13, 13, 13,  0], dtype=np.int32)]
        #
        # actions = [np.array([ 9,  1,  9,  9,  7,  9,  0,  2,  9, 12,  9,  0, 11,  3,  9,  9,  0,
        #         7,  8, 11, 10, 13, 12,  1,  1, 13, 13, 13, 13,  0], dtype=np.int32)]

        programs = [from_tokens(a, optimize=100) for a in actions]

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
                # ?? added by Jasper Hemmes
                min_noinf = min(r[~np.isinf(r)])
                r[np.isinf(r)] = min_noinf
            quantile = np.nanquantile(r, 1 - epsilon, interpolation="higher")
            keep = base_r >= quantile
            #
            # programs = list(compress(programs, keep))
            #
            # actions = actions[keep]
            # obs = [o[keep] for o in obs]
            # priors = priors[keep]
            # n_consts = n_consts[keep]
            #
            # l = l[keep]
            # s = list(compress(s, keep))

        # Redo the optimisation "without" limit only for programs in the top quantile
        for p in list(compress(programs, keep)):
            p.top_quantile = 1  # used in tensorflow to distinguish what programs are in the sub batch.
            p.optimize(2000)

        # memory heavy traversal no longer needed, replace by lighter version
        for p in programs:
            p.replace_traversal()

        # update base_r and r after new optimisation
        base_r = np.array([p.base_r for p in programs])
        r = np.array([p.r for p in programs])

        if any(np.isnan(base_r)):
            # if the const optimisation returns nan constants, the rewards is nan, that is set to min reward here.
            base_r[np.where(np.isnan(base_r))[0]] = min(base_r)
            r[np.where(np.isnan(r))[0]] = min(r)

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

        # Compute baseline
        if baseline == "ewma_R":
            ewma = np.mean(r) if ewma is None else alpha*np.mean(r) + (1 - alpha)*ewma
            b_train = ewma
        elif baseline == "R_e": # Default
            ewma = -1
            b_train = quantile

        # val_list = []
        # for controller in controllers:
        #     with controller.sess.graph.as_default():
        #         train_vars = tf.trainable_variables()
        #         var_names = [v.name for v in train_vars]
        #         values = controller.sess.run(var_names)
        #         val_list.append(values)


        # collect program information for training:
        top_quantile = np.array([p.top_quantile for p in programs])
        # top_quantile = np.array(list(range(1000)))[keep]
        invalid = invalid.astype(float)

        if tensor_dsr:
            # Compute sequence lengths (here I have used the lenghts of individual functions g samples by each)
            #
            # batch_ph = {
            #     "actions": (tf.placeholder(tf.int32, [None, max_length]),
            #                 tf.placeholder(tf.int32, [None, max_length]),
            #                 tf.placeholder(tf.int32, [None, max_length]),
            #                 tf.placeholder(tf.int32, [None, max_length])),
            #     "obs": ((tf.placeholder(tf.int32, [None, max_length]),
            #              tf.placeholder(tf.int32, [None, max_length]),
            #              tf.placeholder(tf.int32, [None, max_length])),
            #             (tf.placeholder(tf.int32, [None, max_length]),
            #              tf.placeholder(tf.int32, [None, max_length]),
            #              tf.placeholder(tf.int32, [None, max_length])),
            #             (tf.placeholder(tf.int32, [None, max_length]),
            #              tf.placeholder(tf.int32, [None, max_length]),
            #              tf.placeholder(tf.int32, [None, max_length])),
            #             (tf.placeholder(tf.int32, [None, max_length]),
            #              tf.placeholder(tf.int32, [None, max_length]),
            #              tf.placeholder(tf.int32, [None, max_length]))),
            #     "priors": tf.placeholder(tf.float32, [None, max_length, n_choices * n_tensors]),
            #     "lengths": (tf.placeholder(tf.int32, [None, ]),
            #                 tf.placeholder(tf.int32, [None, ]),
            #                 tf.placeholder(tf.int32, [None, ]),
            #                 tf.placeholder(tf.int32, [None, ]),
            #                 tf.placeholder(tf.int32, [None, ])),
            #     "rewards": tf.placeholder(tf.float32, [None], name="r"),
            #     "top_quantile": tf.placeholder(tf.float32, [None, ]),
            #     "invalid": tf.placeholder(tf.float32, [None, ])
            # }

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

        # Update and sample from the priority queue
        if priority_queue is not None:
            priority_queue.push_best(sampled_batch, programs)
            pqt_batch = priority_queue.sample_batch(controller.pqt_batch_size)
        else:
            pqt_batch = None

        if save_batch:
            if prev_r_best is None or r_max > prev_r_best:
                batch_filename = f"{output_file.split('.')[0]}_step_{step}.p"
                save_pickle(os.path.join(pickle_dir, batch_filename), sampled_batch)
            if step%50 == 0:
                batch_filename = f"{output_file.split('.')[0]}_step_{step}.p"
                save_pickle(os.path.join(pickle_dir, batch_filename), sampled_batch)

        # Train the controller
        b_train = 0.5
        summaries, loss_ent, loss_inv, loss_pg = controller.train_step(b_train, sampled_batch, pqt_batch)

        if output_file is not None:
            duration = time.process_time() - start_time
            # If the outputted stats are changed dont forget to change the column names in utils
            stats = [[
                         base_r_best,
                         base_r_max,
                         base_r_avg_full,
                         base_r_avg_sub,
                         r_best,
                         r_max,
                         programs[np.argmax(r)].sympy_expr,
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
                         np.round(duration, 2)
                         ]]  # changed this array to a list, changed save routine to pandas to allow expression string
            stats[0] += list(np.round(token_occur_avg_full, 2))
            stats[0] += list(np.round(token_occur_avg_sub, 2))
            df_append = pd.DataFrame(stats)
            df_append.to_csv(os.path.join(logdir, output_file), mode='a', header=False, index=False)


        # ?? disabled when changed to multiple sessions since writer is disabled. Also means that summaries above is unused.
        # if summary:
        #     writer.add_summary(summaries, step)
        #     writer.flush()

        # Update new best expression
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
        #
        # if debug >= 2:
        #     print("\nParameter means after step {} of {}:".format(step+1, n_epochs))
        #     print_var_means()

        if len(Program.cache) > 5000:
            # if the cache contains more than x function, tidy cache.
            Program.tidy_cache(hof)
    #
    # if save_all_r:
    #     with open(all_r_output_file, 'ab') as f:
    #         np.save(f, all_r)

    # Save the hall of fame
    if hof is not None and hof > 0:
        programs = list(Program.cache.values()) # All unique Programs found during training

        base_r = [p.base_r for p in programs]
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
        
    # Print error statistics of the cache
    # ?? Jasper Hemmes. Disabled this because when the cache is cleared it makes no sense to print these stats.
    # Possibly make this available again, saving stats each time chache is cleared ??
    # n_invalid = 0
    # error_types = defaultdict(lambda : 0)
    # error_nodes = defaultdict(lambda : 0)
    # for p in Program.cache.values():
    #     if p.invalid:
    #         n_invalid += p.count
    #         error_types[p.error_type] += p.count
    #         error_nodes[p.error_node] += p.count
    # if n_invalid > 0:
    #     total_samples = (step + 1)*batch_size # May be less than n_samples if breaking early
    #     print("Invalid expressions: {} of {} ({:.1%}).".format(n_invalid, total_samples, n_invalid/total_samples))
    #     print("Error type counts:")
    #     for error_type, count in error_types.items():
    #         print("  {}: {} ({:.1%})".format( error_type, count, count/n_invalid))
    #     print("Error node counts:")
    #     for error_node, count in error_nodes.items():
    #         print("  {}: {} ({:.1%})".format(error_node, count, count/n_invalid))
    #
    # # Print the priority queue at the end of training
    # if verbose and priority_queue is not None:
    #     for i, item in enumerate(priority_queue.iter_in_order()):
    #         print("\nPriority queue entry {}:".format(i))
    #         p = Program.cache[item[0]]
    #         p.print_stats()

    # Compute the pareto front
    if pareto_front:
        if verbose:
            print("Evaluating the pareto front...")
        all_programs = list(Program.cache.values())
        costs = np.array([(p.complexity_eureqa, -p.r) for p in all_programs])
        pareto_efficient_mask = is_pareto_efficient(costs) # List of bool
        pf = list(compress(all_programs, pareto_efficient_mask))
        pf.sort(key=lambda p : p.complexity_eureqa) # Sort by complexity

        if pool is not None:
            results = pool.map(pf_work, pf)
        else:
            results = list(map(pf_work, pf))

        eval_keys = list(results[0][-1].keys())
        columns = ["complexity", "r", "base_r", "count", "expression", "traversal"] + eval_keys
        pf_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
        df = pd.DataFrame(pf_results, columns=columns)
        if pf_output_file is not None:
            print("Saving Pareto Front to {}".format(pf_output_file))
            df.to_csv(pf_output_file, header=True, index=False)

        # Look for a success=True case within the Pareto front
        for p in pf:
            if p.evaluate.get("success"):
                p_final = p
                break

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
