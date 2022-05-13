
import os
import platform
import json
import numpy as np
from sympy.physics.units import Dimension
from sympy.physics.units.systems.si import dimsys_SI
from sympy import log, exp
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

from dsr.turbulence.model_selection.foam_results_processing import add_scatter
from dsr.turbulence.resultprocessing import load_iterations
from dsr.turbulence.dataprocessing import load_frozen_RANS_dataset

def fetch_iteration_metrics(logdir, finished=True):

    results = load_iterations(logdir)

    available_metric = results[list(results.keys())[0]].columns.values
    plot_metrics = [metric for metric in plot_metrics if metric in available_metric]

    # if finished:
    n_iter = 0
    for key, value in results.items():
        if value.shape[0] > n_iter:
            n_iter = value.shape[0]

    plot_dict = {}
    for metric in plot_metrics:
        plot_dict[metric] = []

    for key in results:
        for metric in plot_metrics:
            # if finished:
            if len(results[key][metric].values) == n_iter:
                plot_dict[metric].append(results[key][metric].values)
            else:
                # extend array to full length
                short_arr = results[key][metric].values
                extended_arr = short_arr[-1]*np.ones(n_iter)
                extended_arr[:short_arr.shape[0]] = short_arr
                plot_dict[metric].append(extended_arr)

    # if the r_max_full or r_best_full are nonzero, overwrite base_r_best and base_r_max with full dataset stats
    if 'r_max_full' in plot_metrics:
        if not (plot_dict['r_max_full'][0] == 0).all():
            plot_dict['base_r_max'] = plot_dict['r_max_full']
        del plot_dict['r_max_full']

    if 'r_best_full' in plot_metrics:
        if not (plot_dict['r_best_full'][0] == 0).all():
            plot_dict['base_r_best'] = plot_dict['r_best_full']
        del plot_dict['r_best_full']

    return plot_dict

def convert_expression(expression, inputs):

    c_names = {
        'grad_u_T1': 'grad_u_T1',
        'grad_u_T2': 'grad_u_T2',
        'grad_u_T3': 'grad_u_T3',
        'grad_u_T4': 'grad_u_T4',
        'k': 'k_',
        'inv1': 'inv1',
        'inv2': 'inv2',
        'T1': 'T1',
        'T2': 'T2',
        'T3': 'T3',
        'T4': 'T4'
    }

    for ii in range(len(inputs)):
        expression = expression.replace(f'x{ii+1}', c_names[inputs[ii]])
    return expression


def check_expression_dim(expression, dim_dict):

    if 'E' in expression:
        expression = expression.replace('E', 'exp(1)')

    try:
        expr_dim = eval(expression, dim_dict)
    except NameError:
        # sometimes the expressions contain faulty symbols
        # in that case return different "dimension", investigate if the rewards of these expressions are very good
        return (9, 9, 9, 9, 9, 9, 9)


    try:
        dims = dimsys_SI.get_dimensional_dependencies(expr_dim)
    except TypeError:
        return (10, 10, 10, 10, 10, 10, 10)

    #process dims: [mass, lenght, time, temp, quantity, current, luminousintensity]
    OF_dims = [0, 0, 0, 0, 0, 0, 0]
    for key in dims:
        # note only length and time implemented currently
        if key == 'time':
            OF_dims[2] = dims[key]
        elif key == 'length':
            OF_dims[1] = dims[key]

    return tuple(OF_dims)

def count_tokens(token_str, ntok):
    if isinstance(token_str, str):
        token_str = token_str.replace('[', '')
        token_str = token_str.replace(']', '')
        split_str = token_str.split(' ')
        tokens = [val for val in split_str if val.isnumeric()]
        return len(tokens)
    else:
        return ntok


def summarise_results(logdir):

    dirlist = os.listdir(logdir)
    try:
        dirlist.remove('results.csv')
    except ValueError:
        pass

    dim_dict = {'exp': exp,
                'log': log}

    m = Dimension('length')
    s = Dimension('time')

    input_dims = {"grad_u_T1": 1 / s,
                  "grad_u_T2": 1 / s,
                  "grad_u_T3": 1 / s,
                  "grad_u_T4": 1 / s,
                  "k": (m ** 2) / (s ** 2),
                  "inv1": m / m,
                  "inv2": m / m,
                  "T1": m / m,
                  "T2": m / m,
                  "T3": m / m,
                  "T4": m / m}

    df_results = pd.DataFrame()

    for run in dirlist:

        if '.csv' in run:
            continue

        run_dir = os.path.join(logdir, run)
        print(f'Working on: {run}')
        with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
            config_run = json.load(f)

        output = config_run['task']['dataset']['output'][:4]
        case = ''.join([letter for letter in config_run['task']['dataset']['name'] if not letter.isnumeric()])
        sw = config_run['task']['dataset']['skip_wall']
        ntok = config_run['prior']['length']['max_']

        run_name = f'{output}_{case}_sw{sw}_{ntok}tokens'

        results = load_iterations(os.path.join(logdir, run))

        df_joined = pd.DataFrame()

        for key in results:
            df_joined = pd.concat([df_joined, results[key]], axis=0, ignore_index=True)

        inputs = config_run['task']['dataset']['input']
        for ii in range(len(inputs)):
            dim_dict[f'x{ii+1}'] = input_dims[inputs[ii]]


        df_joined = df_joined[~df_joined['batch_r_max_expression'].isna()]

        df_joined['r_sum'] = df_joined.apply(lambda x: x['r_max_PH'] + x['r_max_CD'] + x['r_max_CBFS'], axis=1)

        if output == 'kDef':
            df_joined['dimensions'] = df_joined.apply(
                lambda x: check_expression_dim(x['batch_r_max_expression'], dim_dict), axis=1)

            target_dim = (0, 2, -3, 0, 0, 0, 0)
        if output == 'bDel':
            target_dim = (0, 0, 0, 0, 0, 0, 0)
            df_joined['dimensions'] = [(0, 0, 0, 0, 0, 0, 0) for _ in df_joined.index]

        df_joined = df_joined.drop_duplicates(subset=['batch_r_max_expression'])
        df_joined['converted_expression'] = df_joined.apply(lambda x: convert_expression(x['batch_r_max_expression'], inputs), axis=1)

        df_joined['name'] = run_name
        df_joined['output'] = output
        df_joined['training_case'] = case
        df_joined['skip_wall'] = sw

        if 'tokens' in df_joined.columns:
            df_joined['ntokens'] = df_joined.apply(lambda x: count_tokens(x['tokens'], ntok), axis=1)
        else:
            df_joined['ntokens'] = ntok

        df_right_dim = df_joined[df_joined['dimensions'] == target_dim]
        df_right_dim = df_right_dim.drop_duplicates(subset=['batch_r_max_expression'])
        df_right_dim['correct_dim'] = True

        # add best on all cases
        df_best = df_right_dim.sort_values('r_sum', ascending=False).head(70)
        df_best['rank'] = np.arange(len(df_best))
        df_best['ranked_by'] = 'r_sum'
        df_results = pd.concat([df_results, df_best], axis=0, ignore_index=True)

        # add best on all cases
        df_best = df_right_dim.sort_values(f'r_max_{case}', ascending=False).head(70)
        df_best['rank'] = np.arange(len(df_best))
        df_best['ranked_by'] = f'r_max_{case}'
        df_results = pd.concat([df_results, df_best], axis=0, ignore_index=True)


        save_arr = df_joined['r_max_PH'].values
        np.savetxt(os.path.join(logdir, f'LR{config_run["controller"]["learning_rate"]}_ent{config_run["controller"]["entropy_weight"]}_rewards.csv'), save_arr, delimiter=',')

        plt.figure()
        plt.hist(df_joined['r_max_PH'], bins=20)
        plt.title(f'{df_joined["r_max_PH"].max()}')
        plt.savefig(f'../logs_completed/aa_plots/aatmp_len{config_run["prior"]["length"]["max_"]}_LR{config_run["controller"]["learning_rate"]}_ent{config_run["controller"]["entropy_weight"]}.png')


        # df_wrong_dim = df_joined[df_joined['dimensions'] != target_dim]
        # df_wrong_dim = df_wrong_dim.drop_duplicates(subset=['batch_r_max_expression'])
        # df_wrong_dim['correct_dim'] = False
        #
        # # add best on all cases
        # df_best = df_wrong_dim.sort_values('r_sum', ascending=False).head(70)
        # df_best['rank'] = np.arange(len(df_best))
        # df_best['ranked_by'] = 'r_sum'
        # df_results = pd.concat([df_results, df_best], axis=0, ignore_index=True)
        #
        # # add best on all cases
        # df_best = df_wrong_dim.sort_values(f'r_max_{case}', ascending=False).head(70)
        # df_best['rank'] = np.arange(len(df_best))
        # df_best['ranked_by'] = f'r_max_{case}'
        # df_results = pd.concat([df_results, df_best], axis=0, ignore_index=True)


    save_cols = ['name','rank', 'ranked_by', 'r_max_PH', 'r_max_CD', 'r_max_CBFS', 'r_sum', 'batch_r_max_expression',
                 'dimensions', 'training_case', 'skip_wall', 'ntokens', 'correct_dim', 'converted_expression']
    df_save = df_results[save_cols]
    df_save = df_save.drop_duplicates(subset=['batch_r_max_expression'])

    df_save.to_csv(os.path.join(logdir, 'results.csv'),index=False)

def plot_reward_dsitribution(logdir):

    dirlist = os.listdir(logdir)

    distributions = {}

    for file in dirlist:
        if '.csv' not in file or 'results' in file:
            continue

        distributions[file] = np.genfromtxt(os.path.join(logdir, file))

    bins = np.arange(0.56, 0.68, 0.005)

    markersize = 25
    lw = 2
    width = 12
    figsize = (width, 3*width/4)
    cm = 1 / 2.54  # centimeters in inches

    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    plt.hist(distributions['LR0.005_ent0.05_rewards.csv'], bins=bins, label=r'$\alpha = 0.005$, $\lambda_H = 0.05$')
    plt.hist(distributions['LR0.005_ent0.01_rewards.csv'], bins=bins, label=r'$\alpha = 0.005$, $\lambda_H = 0.01$')
    plt.hist(distributions['LR0.005_ent0.005_rewards.csv'], bins=bins, label=r'$\alpha = 0.005$, $\lambda_H = 0.005$')
    plt.hist(distributions['LR0.01_ent0.005_rewards.csv'], bins=bins, label=r'$\alpha = 0.010$, $\lambda_H = 0.005$ (BSL)')
    plt.xlabel(r'$r_{max}$', fontsize=12)
    plt.ylabel(r'$n$ unique expressions', fontsize=12)
    ax = plt.gca()
    order = [3, 2, 1, 0]
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order], prop={'size': 9.5})# ,ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9}
    plt.savefig(f'../logs_completed/aa_plots/reward_distribution.eps', format='eps', bbox_inches='tight')


def plot_ntokens_CFDerror():
    #
    #
    # result_file = '../logs_completed/kDef_CD/kDef_CD_selected_models_CFD_results.csv'
    # results_CD = pd.read_csv(result_file)
    # results_CD['CFD_sumerr'] = results_CD['PH_nmse'] + results_CD['CD_nmse'] + results_CD['CBFS_nmse']
    #
    # result_file = '../logs_completed/kDef_PH/kDef_PH_selected_models_CFD_results.csv'
    # results_PH = pd.read_csv(result_file)
    # results_PH['CFD_sumerr'] = results_PH['PH_nmse'] + results_PH['CD_nmse'] + results_PH['CBFS_nmse']
    #
    # result_file = '../logs_completed/kDef_CBFS/kDef_CBFS_selected_models_CFD_results.csv'
    # results_CBFS = pd.read_csv(result_file)
    # results_CBFS['CFD_sumerr'] = results_CBFS['PH_nmse'] + results_CBFS['CD_nmse'] + results_CBFS['CBFS_nmse']

    # does not plot the
    # tokens_PH = []
    # tokens_CD = []
    # tokens_CBFS = []
    # emin_PH = []
    # emin_CD = []
    # emin_CBFS = []
    #
    # for token in sorted(results_PH['ntokens'].unique()):
    #     df_token = results_PH[results_PH['ntokens'] == token].reset_index()
    #     df_row = df_token.iloc[df_token['PH_nmse'].idxmin()]
    #     tokens_PH.append(token)
    #     emin_PH.append(df_row['PH_nmse'])
    #     # emin_CD.append(df_row['CD_nmse'])
    #     # emin_CBFS.append(df_row['CBFS_nmse'])
    #
    # for token in sorted(results_CD['ntokens'].unique()):
    #     df_token = results_CD[results_CD['ntokens'] == token].reset_index()
    #     df_row = df_token.iloc[df_token['CD_nmse'].idxmin()]
    #     tokens_CD.append(token)
    #     emin_CD.append(df_row['CD_nmse'])
    #
    # for token in sorted(results_CBFS['ntokens'].unique()):
    #     df_token = results_CBFS[results_CBFS['ntokens'] == token].reset_index()
    #     df_row = df_token.iloc[df_token['CBFS_nmse'].idxmin()]
    #     tokens_CBFS.append(token)
    #     emin_CBFS.append(df_row['CBFS_nmse'])
    #
    # tokens_PH = np.array(tokens_PH)
    # tokens_CD = np.array(tokens_CD)
    # tokens_CBFS = np.array(tokens_CBFS)
    # emin_PH = np.array(emin_PH)
    # emin_CD = np.array(emin_CD)
    # emin_CBFS = np.array(emin_CBFS)
    #
    # markersize = 25
    # lw = 2
    # width = 12
    # figsize = (width, 3*width/4)
    # cm = 1 / 2.54  # centimeters in inches
    #
    # plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    # plt.xlabel(r"$n_{tokens}$")
    # plt.ylabel(r'$\varepsilon (U) / \varepsilon(U_0)$')
    # plt.xticks(np.arange(0,25,2))
    # plt.yticks(np.arange(0,1,0.1))
    # ax = plt.gca()
    # ax.set_axisbelow(True)
    # plt.grid('both', linestyle=':')
    # plt.plot(tokens_CD[emin_CD < 1], emin_CD[emin_CD < 1], label='$CD_{12600}$', c='C1', linestyle='--', linewidth=lw, marker='^')
    # plt.plot(tokens_CBFS[emin_CBFS < 1], emin_CBFS[emin_CBFS < 1], label='$CBFS_{13700}$', c='C2', linestyle=':', linewidth=lw, marker='v')
    # plt.plot(tokens_PH[emin_PH < 1], emin_PH[emin_PH < 1], label='$PH_{10595}$', c='C0', linestyle=(0, (3, 1, 1, 1)), linewidth=lw, marker='d')
    #
    # order = [2, 0, 1]
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order]) # ,ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9}
    #
    # plt.savefig(f'../logs_completed/aa_plots/ntokens_CFD_err.eps', format='eps', bbox_inches='tight')

    cases = ['PH', 'CD', 'CBFS']

    for case in cases:
        result_file = f'../logs_completed/kDef_{case}/kDef_{case}_selected_models_CFD_results.csv'
        df_models = pd.read_csv(result_file)
        df_models['CFD_sumerr'] = df_models['PH_nmse'] + df_models['CD_nmse'] + df_models['CBFS_nmse']

        ################### I think the plot above is not exactly right.
        tokens_PH = []
        # tokens_CD = []
        # tokens_CBFS = []
        emin_PH = []
        emin_CD = []
        emin_CBFS = []

        for token in sorted(df_models['ntokens'].unique()):
            df_token = df_models[df_models['ntokens'] == token].reset_index()
            df_row = df_token.iloc[df_token[f'{case}_nmse'].idxmin()]
            tokens_PH.append(token)
            emin_PH.append(df_row['PH_nmse'])
            emin_CD.append(df_row['CD_nmse'])
            emin_CBFS.append(df_row['CBFS_nmse'])

        # for token in sorted(results_CD['ntokens'].unique()):
        #     df_token = results_CD[results_CD['ntokens'] == token].reset_index()
        #     df_row = df_token.iloc[df_token['CD_nmse'].idxmin()]
        #     tokens_CD.append(token)
        #     emin_CD.append(df_row['CD_nmse'])
        #
        # for token in sorted(results_CBFS['ntokens'].unique()):
        #     df_token = results_CBFS[results_CBFS['ntokens'] == token].reset_index()
        #     df_row = df_token.iloc[df_token['CBFS_nmse'].idxmin()]
        #     tokens_CBFS.append(token)
        #     emin_CBFS.append(df_row['CBFS_nmse'])

        tokens_PH = np.array(tokens_PH)
        # tokens_CD = np.array(tokens_CD)
        # tokens_CBFS = np.array(tokens_CBFS)
        emin_PH = np.array(emin_PH)
        emin_CD = np.array(emin_CD)
        emin_CBFS = np.array(emin_CBFS)

        markersize = 25
        lw = 2
        width = 7
        figsize = (width, 3*width/4)
        cm = 1 / 2.54  # centimeters in inches

        plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
        plt.xlabel(r"$n_{tokens}$")
        plt.ylabel(r'$\varepsilon (U) / \varepsilon(U_0)$')
        plt.xticks(np.arange(0,25,2))
        plt.yticks(np.arange(0,1,0.1))
        ax = plt.gca()
        ax.set_axisbelow(True)
        plt.grid('both', linestyle=':')
        plt.plot(tokens_PH[emin_CD < 1], emin_CD[emin_CD < 1], label='$CD_{12600}$', c='C1', linestyle='--', linewidth=lw, marker='^')
        plt.plot(tokens_PH[emin_CBFS < 1], emin_CBFS[emin_CBFS < 1], label='$CBFS_{13700}$', c='C2', linestyle=':', linewidth=lw, marker='v')
        plt.plot(tokens_PH[emin_PH < 1], emin_PH[emin_PH < 1], label='$PH_{10595}$', c='C0', linestyle=(0, (3, 1, 1, 1)), linewidth=lw, marker='d')

        order = [2, 0, 1]
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order], prop={'size': 8}, ncol=3, loc='center', bbox_to_anchor=(0.5, 1.15)) # ,ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9}

        plt.savefig(f'../logs_completed/aa_plots/ntokens_CFD_err_{case}.eps', format='eps', bbox_inches='tight')

    ######## This is the old scatterplot in the report that i was not super happy about.
    # markersize = 25
    # lw = 1.5
    # width = 12
    # figsize = (width, 3*width/4)
    # cm = 1 / 2.54  # centimeters in inchesr
    #
    # plt.figure(figsize=tuple([val * cm for val in list(figsize)]))
    # add_scatter(results_PH['ntokens'].values, results_PH, 'CFD_sumerr', 'C0', markersize, lw, r'$PH_{10595}$')
    # add_scatter(results_CD['ntokens'].values - 1/3, results_CD, 'CFD_sumerr', 'C1', markersize, lw, r'$CD_{12600}$')
    # add_scatter(results_CBFS['ntokens'].values + 1/3, results_CBFS, 'CFD_sumerr', 'C2', markersize, lw, r'$CBFS_{13700}$')
    #
    # plt.yticks(np.arange(0, 3, 0.1))
    # plt.xticks(np.arange(0, 25, 5))
    # plt.ylim([0.7, 1.5])
    # ax = plt.gca()
    # ax.set_axisbelow(True)
    # plt.grid('both', linestyle=':')
    # plt.ylabel(r'$\varepsilon _{sum}$')
    # plt.xlabel(r"$n_{tokens}$")
    # plt.axhline(y=0.246319164597 + 0.16591760527490615 + 0.46520543797084507, color='C0', linestyle=(0, (5, 1)), label=r'SpaRTA $M^{(1)}$', linewidth = lw) # densely dashed
    # plt.axhline(y=0.246319164597 + 0.16591760527490615 + 0.46520543797084507, color='C1', linestyle=(0, (1, 1)), label=r'SpaRTA $M^{(2)}$', linewidth = lw)
    # plt.axhline(y=0.2081585409088 + 0.20329225923 + 0.499579335406, color='C2', linestyle=(0, (3, 1, 1, 1, 1, 1)), label=r'SpaRTA $M^{(3)}$', linewidth = lw)
    # plt.scatter(10, 10, c='none', edgecolors='grey', s=markersize, linewidth=lw,
    #             label='Incorrect dimensionality')
    # plt.legend(prop={'size': 8}, loc='center right', bbox_to_anchor=(1.55, 0.5))
    # plt.savefig(f'../logs_completed/aa_plots/tokens_vs_CFDerrorsum.eps', format='eps', bbox_inches='tight')

    #
    # # analyse convergence of log or exp functions:
    # results_CBFS['contains_logexp'] = results_CBFS.apply(lambda x: ('exp' in x['batch_r_max_expression']) or 'log' in (x['batch_r_max_expression']), axis=1)
    # results_CD['contains_logexp'] = results_CD.apply(lambda x: ('exp' in x['batch_r_max_expression']) or 'log' in (x['batch_r_max_expression']), axis=1)
    # results_PH['contains_logexp'] = results_PH.apply(lambda x: ('exp' in x['batch_r_max_expression']) or 'log' in (x['batch_r_max_expression']), axis=1)
    #
    #
    # limit = 10
    # n_bl = sum((results_PH['ntokens'] <= limit)) # number of expression below limit
    # n_bl_converged = sum((results_PH['ntokens'] <= limit) & (results_PH['CFD_sumerr'] < 3))
    #
    # n_bl += sum((results_CBFS['ntokens'] <= limit)) # number of expression below limit
    # n_bl_converged += sum((results_CBFS['ntokens'] <= limit) & (results_CBFS['CFD_sumerr'] < 3))
    #
    # n_bl += sum((results_CD['ntokens'] <= limit)) # number of expression below limit
    # n_bl_converged += sum((results_CD['ntokens'] <= limit) & (results_CD['CFD_sumerr'] < 3))
    #
    # n_al = sum((results_PH['ntokens'] > limit)) # number of expression above limit
    # n_al_converged = sum((results_PH['ntokens'] > limit) & (results_PH['CFD_sumerr'] < 3))
    #
    # n_al += sum((results_CBFS['ntokens'] > limit)) # number of expression above limit
    # n_al_converged += sum((results_CBFS['ntokens'] > limit) & (results_CBFS['CFD_sumerr'] < 3))
    #
    # n_al += sum((results_CD['ntokens'] > limit)) # number of expression above limit
    # n_al_converged += sum((results_CD['ntokens'] > limit) & (results_CD['CFD_sumerr'] < 3))
    #
    #
    # n_cor = sum(results_PH['correct_dim'] == True)
    # n_incor = sum(results_PH['correct_dim'] == False)
    #
    #
    # results_PH.to_csv('../logs_completed/tmp_ph.csv')
    # results_CD.to_csv('../logs_completed/tmp_cd.csv')
    # results_CBFS.to_csv('../logs_completed/tmp_cbfs.csv')


def write_OF_model_file(expression, model_nr, model_type):

    example_file = '../logs_completed/models/model0000.C'

    if 'E' in expression:
        expression = expression.replace('E', 'exp(1)')

    if '**' in expression:
        print(f'found ** in {model_nr}')

    if model_type == 'bDel':
        c_name = 'bijDelta'
        inputs = ["T1", "T2", "T3", "T4", "inv1", "inv2"]

    elif model_type == 'kDef':
        c_name = 'kDeficit'
        inputs = ["grad_u_T1", "grad_u_T2", "grad_u_T3", "grad_u_T4", "k", "inv1", "inv2"]

    with open(example_file) as f:
        lines = f.readlines()

    header_lines = lines[:-2]
    header_lines.append(f'    Info << "{c_name}: DSR model {model_nr}" << endl;\n')

    if 'x' in expression:
        # needs replacing
        expression = convert_expression(expression, inputs)

    header_lines.append(f'    {c_name}_ = {expression};\n')

    with open(f'../logs_completed/models/{model_type}{model_nr:04d}.C', 'w') as newfile:
        for line in header_lines:
            newfile.write(line)

    newfile.close()

def write_selected_models_to_C(path):

    df_selected_models = pd.read_csv(path)

    model_type = np.unique([name[:4] for name in df_selected_models['name'].values])[0]

    for index, row in df_selected_models.iterrows():

        filename = f'model_{row["model_nr"]}'
        row.to_csv(os.path.join('../logs_completed/models/model_info_files', filename), header=False)

        write_OF_model_file(row['batch_r_max_expression'], row['model_nr'], model_type)




    for index, row in df_selected_models.iterrows():
        print(f'    else if ({model_type}_model_nr_.value() == {row["model_nr"]}) {{')
        print(f'        #include "{model_type}{row["model_nr"]:04d}.C"')
        print('    }')


def plot_ntokens_r_max(logdir):

    dirlist = os.listdir(logdir)

    tokens = []
    r_max_PH = []
    r_max_CD = []
    r_max_CBFS = []

    dim_dict = {'exp': exp,
                'log': log}

    m = Dimension('length')
    s = Dimension('time')

    input_dims = {"grad_u_T1": 1 / s,
                  "grad_u_T2": 1 / s,
                  "grad_u_T3": 1 / s,
                  "grad_u_T4": 1 / s,
                  "k": (m ** 2) / (s ** 2),
                  "inv1": m / m,
                  "inv2": m / m,
                  "T1": m / m,
                  "T2": m / m,
                  "T3": m / m,
                  "T4": m / m}

    for run in dirlist:

        if '.csv' in run:
            continue

        run_dir = os.path.join(logdir, run)
        print(f'Working on: {run}')
        with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
            config_run = json.load(f)

        output = config_run['task']['dataset']['output'][:4]
        case = ''.join([letter for letter in config_run['task']['dataset']['name'] if not letter.isnumeric()])
        sw = config_run['task']['dataset']['skip_wall']
        ntok = config_run['prior']['length']['max_']

        run_name = f'{output}_{case}_sw{sw}_{ntok}tokens'

        results = load_iterations(os.path.join(logdir, run))

        df_joined = pd.DataFrame()

        for key in results:
            df_joined = pd.concat([df_joined, results[key]], axis=0, ignore_index=True)

        df_joined['r_sum'] = df_joined.apply(lambda x: x['r_max_PH'] + x['r_max_CD'] + x['r_max_CBFS'], axis=1)

        inputs = config_run['task']['dataset']['input']
        for ii in range(len(inputs)):
            dim_dict[f'x{ii + 1}'] = input_dims[inputs[ii]]

        df_joined['dimensions'] = df_joined.apply(lambda x: check_expression_dim(x['batch_r_max_expression'], dim_dict), axis=1)

        if output == 'kDef':
            target_dim = (0, 2, -3, 0, 0, 0, 0)
        if output == 'bDel':
            target_dim = (0, 0, 0, 0, 0, 0, 0)

        df_joined = df_joined[df_joined['dimensions'] == target_dim]

        df_joined['name'] = run_name
        df_joined['output'] = output
        df_joined['training_case'] = case
        df_joined['skip_wall'] = sw

        if 'tokens' in df_joined.columns:
            df_joined['ntokens'] = df_joined.apply(lambda x: count_tokens(x['tokens'], ntok), axis=1)
        else:
            df_joined['ntokens'] = ntok

        tokens.append(df_joined['ntokens'].values)
        r_max_PH.append(df_joined['r_max_PH'].values)
        r_max_CD.append(df_joined['r_max_CD'].values)
        r_max_CBFS.append(df_joined['r_max_CBFS'].values)

    tokens = np.concatenate(tokens, axis=0)
    r_max_PH = np.concatenate(r_max_PH, axis=0)
    r_max_CD = np.concatenate(r_max_CD, axis=0)
    r_max_CBFS = np.concatenate(r_max_CBFS, axis=0)

    sorted_tokens = []
    sorted_r_max_PH = []
    sorted_r_max_CD = []
    sorted_r_max_CBFS = []

    for token in np.unique(tokens):
        sorted_tokens.append(token)
        best_model_PH = np.argmax(r_max_PH[tokens == token])
        sorted_r_max_PH.append(r_max_PH[tokens == token][best_model_PH])
        sorted_r_max_CD.append(r_max_CD[tokens == token][best_model_PH])
        sorted_r_max_CBFS.append(r_max_CBFS[tokens == token][best_model_PH])

    markersize = 25
    lw = 2
    width = 7
    figsize = (width, 3*width/4)
    cm = 1 / 2.54  # centimeters in inches

    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    plt.xlabel(r"$n_{tokens}$")
    plt.ylabel(r"$r_{max}$")
    plt.xticks(np.arange(0,25,2))
    plt.yticks(np.arange(0,1,0.05))
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid('both', linestyle=':')
    plt.plot(sorted_tokens, sorted_r_max_CD, label='$CD_{12600}$', c='C1', linestyle='--', linewidth=lw, marker='^')
    plt.plot(sorted_tokens, sorted_r_max_CBFS, label='$CBFS_{13700}$', c='C2', linestyle=':', linewidth=lw, marker='v')
    plt.plot(sorted_tokens, sorted_r_max_PH, label='$PH_{10595}$', c='C0', linestyle=(0, (3, 1, 1, 1)), linewidth=lw, marker='d')

    order = [2, 0, 1]
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order], ncol=3, loc='center', bbox_to_anchor=(0.5, 1.15), prop={'size': 8}) # ,ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9}

    plt.savefig(f'../logs_completed/aa_plots/ntokens_r_max{logdir.split("/")[-1]}.eps', format='eps', bbox_inches='tight')

def search_tokens(df_row, df_joined):

    if df_row['tokens_search'] != '[0 0 0]':
        return df_row['tokens_search']

    limit = 0.00001
    df_filtered = df_joined[(df_joined['r_sum'] > df_row['r_sum'] - limit) & (df_joined['r_sum'] < df_row['r_sum'] + limit)]

    df_filtered = df_filtered[~df_filtered['tokens'].isna()]

    if len(df_filtered) == 0:
        return "[0 0 0]"

    for index, row in df_filtered.iterrows():
        if row['batch_r_max_expression'] == df_row['batch_r_max_expression']:
            return row['tokens']

    df_filtered = df_filtered[(df_filtered['r_max_PH'] > df_row['r_max_PH'] - limit) & (df_filtered['r_max_PH'] < df_row['r_max_PH'] + limit)]
    df_filtered = df_filtered[(df_filtered['r_max_CD'] > df_row['r_max_CD'] - limit) & (df_filtered['r_max_CD'] < df_row['r_max_CD'] + limit)]
    df_filtered = df_filtered[(df_filtered['r_max_CBFS'] > df_row['r_max_CBFS'] - limit) & (df_filtered['r_max_CBFS'] < df_row['r_max_CBFS'] + limit)]

    if len(df_filtered) == 0:
        return "[0 0 0]"

    custom_return = "[0 0 0]"

    print(df_row['batch_r_max_expression'])

    return custom_return



def add_tokens(logdir, selected_models_file):

    df_models = pd.read_csv(selected_models_file)

    df_models['train_n_tokens'] = df_models['name'].apply(lambda x: int(x.split('_')[-1].replace('tokens', '')))

    if 'tokens_search' not in df_models.columns:
        df_models['tokens_search'] = '[0 0 0]'

    dirlist = os.listdir(logdir)

    for run in dirlist:

        if '.csv' in run or '.json' in run or run == 'results':
            continue
        # if run == 'results':
        #     continue

        run_dir = os.path.join(logdir, run)
        print(f'Working on: {run}')
        with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
            config_run = json.load(f)

        output = config_run['task']['dataset']['output'][:4]
        case = ''.join([letter for letter in config_run['task']['dataset']['name'] if not letter.isnumeric()])
        sw = config_run['task']['dataset']['skip_wall']
        ntok = config_run['prior']['length']['max_']

        run_name = f'{output}_{case}_sw{sw}_{ntok}tokens'

        results = load_iterations(os.path.join(logdir, run))

        df_joined = pd.DataFrame()

        for key in results:
            df_joined = pd.concat([df_joined, results[key]], axis=0, ignore_index=True)

        if 'tokens' not in df_joined.columns:
            continue

        df_joined['r_sum'] = df_joined.apply(lambda x: x['r_max_PH'] + x['r_max_CD'] + x['r_max_CBFS'], axis=1)

        df_models['tokens_search'] = df_models.apply(lambda x: search_tokens(x, df_joined), axis=1)

    filename = selected_models_file[:-4] + '_added_tokens.csv'
    df_models.to_csv(filename, index=False)

def categorise_tokens(df, type):
    if type == 'kDef':
       names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'add', 'sub', 'mul', 'div', 'exp', 'log', 'const']

    results = {name: 0 for name in names}

    for index, row in df.iterrows():
        tokens = row['tokens']
        if tokens == '[0 0 0]':
            continue

        tokens = tokens.replace('[', '')
        tokens = tokens.replace(']', '')

        tokens = [int(token) for token in tokens.split(' ') if token.isnumeric()]

        for token in tokens:
            results[names[token]] += 1

    return results

def plot_token_distribution():

    df_models = pd.DataFrame()
    for case in ['PH', 'CD', 'CBFS']:
        file = f'../logs_completed/kDef_{case}/kDef_{case}_selected_models_CFD_results.csv'
        # file = f'../logs_completed/bDel_{case}/bDel_{case}_selected_models_CFD_results_full_bDelta.csv'
        df_case = pd.read_csv(file)
        df_models = pd.concat([df_models, df_case], axis=0, ignore_index=True)

    # df_models = pd.read_csv(file)
    df_models.loc[df_models['tokens'].isna(), 'tokens'] = '[0 0 0]'

    df_models['CFD_err_sum'] = df_models['CD_nmse'] + df_models['PH_nmse'] + df_models['CBFS_nmse']

    # df_improved = df_models[df_models['CFD_err_sum'] < 3]
    # df_worse = df_models[df_models['CFD_err_sum'] >= 3]

    df_improved = df_models[(df_models['CD_nmse'] < 1 ) & (df_models['PH_nmse'] < 1 ) & (df_models['CBFS_nmse'] < 1 )]
    df_worse = df_models[(df_models['CD_nmse'] >= 1 ) & (df_models['PH_nmse'] >= 1 ) & (df_models['CBFS_nmse'] >= 1 )]

    results_all = categorise_tokens(df_models, 'kDef')
    results_improved = categorise_tokens(df_improved, 'kDef')
    results_worse = categorise_tokens(df_worse, 'kDef')

    token_labels = [r"$T^{(1)} \bar{S}_{ij}$",
                    r"$T^{(2)} \bar{S}_{ij}$",
                    r"$T^{(3)} \bar{S}_{ij}$",
                    r"$T^{(4)} \bar{S}_{ij}$",
                    r"$k$",
                    r"$\theta_1$",
                    r"$\theta_2$",
                    'add', 'sub', 'mul', 'div', 'exp', 'log', 'const']

    base_x = np.arange(len(results_all.keys())) + 1
    barwidth = 0.20
    shift = 0.25

    figsize = (20, 5)
    cm = 1 / 2.54  # centimeters in inches

    plt.figure(figsize=tuple([val * cm for val in list(figsize)]))
    plt.bar(base_x - shift, np.array(list(results_all.values())) / (sum(results_all.values())), color='C0', width=barwidth, label='All')
    plt.bar(base_x , np.array(list(results_improved.values())) / (sum(results_improved.values())), color='C1', width=barwidth,  label=r'Improved flow field')  #     , label=r'$\varepsilon_{sum} < 3$')
    plt.bar(base_x + shift, np.array(list(results_worse.values())) / (sum(results_worse.values())), color='C2', width=barwidth, label=r'Worsened flow field')  # , label=r'$\varepsilon_{sum} \geq 3$')
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.set_xticks(base_x)
    ax.set_yticks(np.arange(0, 0.25, 0.05))
    ax.set_xticklabels(token_labels)
    plt.ylabel(r'$p(token)$')
    ax.tick_params(axis='x', which='major', labelsize=9)
    plt.grid('both', linestyle=':')
    plt.legend()  # prop={'size': 8}, loc='center right', bbox_to_anchor=(1.55, 0.5))
    plt.savefig(f'../logs_completed/aa_plots/token_distribution.eps', format='eps', bbox_inches='tight')


def tensor_expr_eval(expr, vardict, y, tensor_var, full_expression):


    vardict['np'] = np

    expr = expr.replace('exp', 'np.exp')
    expr = expr.replace('E', 'np.exp(1)')
    expr = expr.replace('log', 'np.log')

    full_expression = full_expression.replace('exp', 'np.exp')
    full_expression = full_expression.replace('E', 'np.exp(1)')
    full_expression = full_expression.replace('log', 'np.log')


    g = eval(expr, vardict)

    absmean = np.mean(abs(g))
    g_mean = np.mean(g)

    cor_mat = np.corrcoef(g*vardict[tensor_var], y)

    return cor_mat[1,0], absmean, g_mean

def split_expression(expression):

    split_expr = []
    brackets = 0
    sub_expr = ''
    prev_char = ''
    for char in expression:
        if char == '(':
            brackets += 1
        if char == ')':
            brackets -= 1

        if char == '+' or char == '-':
            if prev_char == 'e' or prev_char == '':
                sub_expr += char
                continue
            if brackets == 0:
                split_expr.append(sub_expr)
                sub_expr = ''

        prev_char = char

        sub_expr += char
    split_expr.append(sub_expr)

    if len(split_expr) != 4:
        if 'x1' not in expression:
            split_expr.insert(0, '0')
        elif 'x2' not in expression:
            split_expr.insert(1, '0')
        elif 'x3' not in expression:
            split_expr.insert(2, '0')
        elif 'x4' not in expression:
            split_expr.append('0')

    if len(split_expr) != 4:
        print('error')


    return split_expr


def bDelta_magnitudes(model_file):

    #
    # Possibly use this for all bDelta models!
    # df_models = pd.DataFrame()
    # for case in ['PH', 'CD', 'CBFS']:
    #     file = f'../logs_completed/kDef_{case}/kDef_{case}_selected_models_CFD_results.csv'
    #     # file = f'../logs_completed/bDel_{case}/bDel_{case}_selected_models_CFD_results_full_bDelta.csv'
    #     df_case = pd.read_csv(file)
    #     df_models = pd.concat([df_models, df_case], axis=0, ignore_index=True)

    # sort by training reward?

    df_models = pd.read_csv(model_file)
    df_models[['x1term','x2term','x3term','x4term']] = df_models.apply(lambda x: split_expression(x['batch_r_max_expression']), axis=1, result_type='expand')

    # Load the config file
    with open('/'.join(model_file.split('/')[:-1] + ['config.json']), encoding='utf-8') as f:
        config = json.load(f)

    # config['task']['dataset']['skip_wall'] = False
    X, y = load_frozen_RANS_dataset(config['task']['dataset'])

    vardict = {}

    for ii in range(X.shape[1]):
        vardict[f'x{ii+1}'] = X[:,ii]

    for ii in range(4):
        strings = df_models[f'x{ii+1}term'].values
        strings = [str.replace(f'x{ii+1}*', '') for str in strings]
        strings = [str.replace(f'*x{ii+1}', '') for str in strings]
        strings = [str.replace(f'x{ii+1}', '0') for str in strings]
        df_models[f'x{ii+1}term'] = strings

        df_models[[f'x{ii+1}corr', f'x{ii+1}absmean', f'x{ii+1}mean']] = df_models.apply(lambda x: tensor_expr_eval(x[f'x{ii+1}term'], vardict, y, f'x{ii+1}', x['batch_r_max_expression']), axis=1, result_type='expand')

    if len(df_models['training_case'].unique()) == 1:
        training_case = df_models['training_case'].unique()[0]
    if training_case == 'PH':
        sort_by_CFD = ['PH_nmse', 'CD_nmse', 'CBFS_nmse']
        sort_by_r_max = ['r_max_PH', 'r_max_CD', 'r_max_CBFS']
        rlabel = r'\; PH_{10595}$'
        # rlabel = r'$r_{max} \; PH_{10595}$'

    elif training_case == 'CD':
        sort_by_CFD = ['CD_nmse', 'PH_nmse', 'CBFS_nmse']
        sort_by_r_max = ['r_max_CD', 'r_max_PH', 'r_max_CBFS']
        rlabel = '\; CD_{12600}$'
        # rlabel = '$r_{max}  \; CD_{12600}$'

    else:
        sort_by_CFD = ['CBFS_nmse', 'PH_nmse', 'CD_nmse']
        sort_by_r_max = ['r_max_CBFS', 'r_max_PH', 'r_max_CD']
        rlabel = '\; CBFS_{13700}$'

    df_sorted = df_models.sort_values(sort_by_r_max,
                                      ascending=[False, False, False], ignore_index=True)

    x = np.arange(df_sorted.shape[0]) + 1

    markersize = 30
    lw = 1.5
    figsize = (24, 6)
    cm = 1 / 2.54  # centimeters in inches

    # plot r_max:
    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    # plt.plot(x, df_sorted[f'r_max_{training_case}'] )
    add_scatter(x, df_sorted, 'x1corr', 'C0', markersize, lw, r'$g^{(1)} T^{(1)} $', 'd')
    add_scatter(x, df_sorted, 'x2corr', 'C1', markersize, lw, r'$g^{(2)} T^{(2)} $', '^')
    add_scatter(x, df_sorted, 'x3corr', 'C2', markersize, lw, r'$g^{(3)} T^{(3)} $', 'v')
    add_scatter(x, df_sorted, 'x4corr', 'C3', markersize, lw, r'$g^{(4)} T^{(4)} $', 'o')

    plt.ylabel(r'$\rho_c$')
    plt.xlabel('Models')
    # plt.ylim([-18, 18])
    plt.xticks(np.arange(0,100,10))
    ax = plt.gca()
    ax.xaxis.grid(linestyle=':')
    ax.yaxis.grid(linestyle=':')
    plt.legend(prop={'size': 8})

    ax2 = ax.twinx()
    ax2.set_ylabel(r'$r_{max} $')
    ax2.plot(x, df_sorted[f'r_max_{training_case}'] , label=r'$r_{max}' + rlabel, zorder=-1, c='black', linestyle=(0, (1, 1)))
    if training_case == 'CBFS':
        ax2.set_yticks([0.55, 0.60, 0.65])

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    lines = lines_1 + lines_2
    labels = labels_1 + labels_2

    ax.legend(lines, labels, prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=5)

    plt.savefig(f'../logs_completed/aa_plots/bDelCorr_r_max_{training_case}.eps', format='eps', bbox_inches='tight')

    df_sorted = df_models.sort_values(sort_by_CFD,
                                      ascending=[True, True, True], ignore_index=True)

    # plot CFD errors:
    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    # plt.plot(x, df_sorted[f'r_max_{training_case}'] )
    add_scatter(x, df_sorted, 'x1corr', 'C0', markersize, lw, r'$g^{(1)} T^{(1)} $', 'd')
    add_scatter(x, df_sorted, 'x2corr', 'C1', markersize, lw, r'$g^{(2)} T^{(2)} $', '^')
    add_scatter(x, df_sorted, 'x3corr', 'C2', markersize, lw, r'$g^{(3)} T^{(3)} $', 'v')
    add_scatter(x, df_sorted, 'x4corr', 'C3', markersize, lw, r'$g^{(4)} T^{(4)} $', 'o')

    plt.ylabel(r'$\rho_c$')
    plt.xlabel('Models')
    # plt.ylim(ylim)
    plt.xticks(np.arange(0,100,10))
    ax = plt.gca()
    ax.xaxis.grid(linestyle=':')
    ax.yaxis.grid(linestyle=':')
    plt.legend(prop={'size': 8})

    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\varepsilon (U) / \varepsilon(U_0)$')
    ax2.plot(x, df_sorted[f'{training_case}_nmse'] , label=r'$\varepsilon (U) / \varepsilon(U_0)' + rlabel, zorder=-1, c='black', linestyle=(0, (1, 1)))

    ax2.set_ylim([0, 4])

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    lines = lines_1 + lines_2
    labels = labels_1 + labels_2

    ax.legend(lines, labels, prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=5)

    plt.savefig(f'../logs_completed/aa_plots/bDelCorr_CFD_err_{training_case}.eps', format='eps', bbox_inches='tight')


if __name__ == "__main__":

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    matplotlib.use('tkagg')

    #
    #
    # models_path = '../logs_completed/bDel_CD/bDel_CD_selected_models.csv'
    # models_path = '../logs_completed/kDef_CBFS/kDef_CBFS_selected_models.csv'
    # models_path = '../logs_completed/kDef_CBFS/kDef_CBFS_selected_models.csv'
    # write_selected_models_to_C(models_path)
    #
    # logdir = '../logs_completed/kDef_PH_entweight_10tok'
    # summarise_results(logdir)
    logdir = '../logs_completed/kDef_PH_entweight_10tok'
    plot_reward_dsitribution(logdir)
    #
    # logdir = '../logs_completed/kDef_PH_ntokens'
    # plot_ntokens_r_max(logdir)

    # plot_ntokens_CFDerror() # files are hardcoded in the function itself
    #
    # plot_token_distribution()
    # add_tokens(f'../logs_completed/bDel_PH',
    #
    # bDelta_magnitudes(f'../logs_completed/bDel_PH/bDel_PH_selected_models_CFD_results_full_bDelta.csv')
    # bDelta_magnitudes(f'../logs_completed/bDel_CD/bDel_CD_selected_models_CFD_results_full_bDelta.csv')
    # bDelta_magnitudes(f'../logs_completed/bDel_CBFS/bDel_CBFS_selected_models_CFD_results_full_bDelta.csv')

    # bDelta_magnitudes(f'../logs_completed/bDel_PH/bDel_PH_selected_models_CFD_results_20percent_bDelta_weight.csv')
    # bDelta_magnitudes(f'../logs_completed/bDel_CD/bDel_CD_selected_models_CFD_results_20percent_bDelta_weight.csv')
    # bDelta_magnitudes(f'../logs_completed/bDel_CBFS/bDel_CBFS_selected_models_CFD_results_20percent_bDelta_weight.csv')

   #
   #  # below to add tokens to model file retrospectively
   #  for case in ['PH', 'CD', 'CBFS']:
   #      add_tokens(f'../logs_completed/kDef_{case}',
   #                 f'../logs_completed/kDef_{case}/kDef_{case}_selected_models_CFD_results.csv')


    print('end')