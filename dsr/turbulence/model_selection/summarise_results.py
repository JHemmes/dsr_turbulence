
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

from dsr.turbulence.resultprocessing import load_iterations
from dsr.turbulence.model_selection.foam_results_processing import add_scatter

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

        df_joined['dimensions'] = df_joined.apply(lambda x: check_expression_dim(x['batch_r_max_expression'], dim_dict), axis=1)

        df_joined['r_sum'] = df_joined.apply(lambda x: x['r_max_PH'] + x['r_max_CD'] + x['r_max_CBFS'], axis=1)

        if output == 'kDef':
            target_dim = (0, 2, -3, 0, 0, 0, 0)
        if output == 'bDel':
            target_dim = (0, 0, 0, 0, 0, 0, 0)

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
        df_best = df_right_dim.sort_values('r_sum', ascending=False).head(30)
        df_best['rank'] = np.arange(len(df_best))
        df_best['ranked_by'] = 'r_sum'
        df_results = pd.concat([df_results, df_best], axis=0, ignore_index=True)

        # add best on all cases
        df_best = df_right_dim.sort_values(f'r_max_{case}', ascending=False).head(30)
        df_best['rank'] = np.arange(len(df_best))
        df_best['ranked_by'] = f'r_max_{case}'
        df_results = pd.concat([df_results, df_best], axis=0, ignore_index=True)

        df_wrong_dim = df_joined[df_joined['dimensions'] != target_dim]
        df_wrong_dim = df_wrong_dim.drop_duplicates(subset=['batch_r_max_expression'])
        df_wrong_dim['correct_dim'] = False

        # add best on all cases
        df_best = df_wrong_dim.sort_values('r_sum', ascending=False).head(30)
        df_best['rank'] = np.arange(len(df_best))
        df_best['ranked_by'] = 'r_sum'
        df_results = pd.concat([df_results, df_best], axis=0, ignore_index=True)

        # add best on all cases
        df_best = df_wrong_dim.sort_values(f'r_max_{case}', ascending=False).head(30)
        df_best['rank'] = np.arange(len(df_best))
        df_best['ranked_by'] = f'r_max_{case}'
        df_results = pd.concat([df_results, df_best], axis=0, ignore_index=True)


    save_cols = ['name','rank', 'ranked_by', 'r_max_PH', 'r_max_CD', 'r_max_CBFS', 'r_sum', 'batch_r_max_expression',
                 'dimensions', 'training_case', 'skip_wall', 'ntokens', 'correct_dim', 'converted_expression']
    df_save = df_results[save_cols]
    df_save.to_csv(os.path.join(logdir, 'results.csv'),index=False)

def scatter_ntokens_CFDerror(result_file):

    results = pd.read_csv(result_file)

    #
    # markersize = 25
    # lw = 1.5
    # width = 10
    # figsize = (width, 3*width/4)
    # cm = 1 / 2.54  # centimeters in inches
    #
    # plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    # plt.scatter(results['ntokens'].values, results['PH_nmse'].values, s=markersize)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xticks(np.arange(0,25,5))
    # plt.ylim([0, 0.3])
    # ax = plt.gca()
    # ax.set_axisbelow(True)
    # plt.grid('both', linestyle=':')
    # plt.ylabel(r'$\varepsilon (U) / \varepsilon(U_0)$')
    # plt.xlabel(r"$n_{tokens}$")
    #
    #
    # plt.figure()
    # plt.title('CD')
    # plt.scatter(results['ntokens'].values, results['CD_nmse'].values)
    # plt.ylim([0, 1])
    #
    #
    # plt.figure()
    # plt.title('CBFS')
    # plt.scatter(results['ntokens'].values, results['CBFS_nmse'].values)
    # plt.ylim([0, 1])


    result_file = '../logs_completed/kDef_CD/kDef_CD_selected_models_CFD_results.csv'
    results_CD = pd.read_csv(result_file)
    results_CD['CFD_sumerr'] = results_CD['PH_nmse'] + results_CD['CD_nmse'] + results_CD['CBFS_nmse']

    result_file = '../logs_completed/kDef_PH/kDef_PH_selected_models_CFD_results.csv'
    results_PH = pd.read_csv(result_file)
    results_PH['CFD_sumerr'] = results_PH['PH_nmse'] + results_PH['CD_nmse'] + results_PH['CBFS_nmse']

    result_file = '../logs_completed/kDef_CBFS/kDef_CBFS_selected_models_CFD_results.csv'
    results_CBFS = pd.read_csv(result_file)
    results_CBFS['CFD_sumerr'] = results_CBFS['PH_nmse'] + results_CBFS['CD_nmse'] + results_CBFS['CBFS_nmse']

    limit = 10
    n_bl = sum((results_PH['ntokens'] <= limit)) # number of expression below limit
    n_bl_converged = sum((results_PH['ntokens'] <= limit) & (results_PH['CFD_sumerr'] < 3))

    n_bl += sum((results_CBFS['ntokens'] <= limit)) # number of expression below limit
    n_bl_converged += sum((results_CBFS['ntokens'] <= limit) & (results_CBFS['CFD_sumerr'] < 3))

    n_bl += sum((results_CD['ntokens'] <= limit)) # number of expression below limit
    n_bl_converged += sum((results_CD['ntokens'] <= limit) & (results_CD['CFD_sumerr'] < 3))

    n_al = sum((results_PH['ntokens'] > limit)) # number of expression above limit
    n_al_converged = sum((results_PH['ntokens'] > limit) & (results_PH['CFD_sumerr'] < 3))

    n_al += sum((results_CBFS['ntokens'] > limit)) # number of expression above limit
    n_al_converged += sum((results_CBFS['ntokens'] > limit) & (results_CBFS['CFD_sumerr'] < 3))

    n_al += sum((results_CD['ntokens'] > limit)) # number of expression above limit
    n_al_converged += sum((results_CD['ntokens'] > limit) & (results_CD['CFD_sumerr'] < 3))

    print(f'% converged below {limit}: {100* n_bl_converged/n_bl}')
    print(f'% converged above {limit}: {100* n_al_converged/n_al}')
    markersize = 25
    lw = 1.5
    width = 12
    figsize = (width, 3*width/4)
    cm = 1 / 2.54  # centimeters in inchesr

    plt.figure(figsize=tuple([val * cm for val in list(figsize)]))
    add_scatter(results_PH['ntokens'].values, results_PH, 'CFD_sumerr', 'C0', markersize, lw, r'$PH_{10595}$')
    add_scatter(results_CD['ntokens'].values - 1/3, results_CD, 'CFD_sumerr', 'C1', markersize, lw, r'$CD_{12600}$')
    add_scatter(results_CBFS['ntokens'].values + 1/3, results_CBFS, 'CFD_sumerr', 'C2', markersize, lw, r'$CBFS_{13700}$')

    plt.yticks(np.arange(0, 3, 0.1))
    plt.xticks(np.arange(0, 25, 5))
    plt.ylim([0.7, 1.5])
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid('both', linestyle=':')
    plt.ylabel(r'$\varepsilon _{sum}$')
    plt.xlabel(r"$n_{tokens}$")
    plt.axhline(y=0.246319164597 + 0.16591760527490615 + 0.46520543797084507, color='C0', linestyle=(0, (5, 1)), label=r'SpaRTA $M^{(1)}$', linewidth = lw) # densely dashed
    plt.axhline(y=0.246319164597 + 0.16591760527490615 + 0.46520543797084507, color='C1', linestyle=(0, (1, 1)), label=r'SpaRTA $M^{(2)}$', linewidth = lw)
    plt.axhline(y=0.2081585409088 + 0.20329225923 + 0.499579335406, color='C2', linestyle=(0, (3, 1, 1, 1, 1, 1)), label=r'SpaRTA $M^{(3)}$', linewidth = lw)
    plt.scatter(10, 10, c='none', edgecolors='grey', s=markersize, linewidth=lw,
                label='Incorrect dimensionality')
    plt.legend(prop={'size': 8}, loc='center right', bbox_to_anchor=(1.55, 0.5))
    plt.savefig(f'../logs_completed/aa_plots/tokens_vs_CFDerrorsum.eps', format='eps', bbox_inches='tight')


    # analyse convergence of log or exp functions:
    results_CBFS['contains_logexp'] = results_CBFS.apply(lambda x: ('exp' in x['batch_r_max_expression']) or 'log' in (x['batch_r_max_expression']), axis=1)
    results_CD['contains_logexp'] = results_CD.apply(lambda x: ('exp' in x['batch_r_max_expression']) or 'log' in (x['batch_r_max_expression']), axis=1)
    results_PH['contains_logexp'] = results_PH.apply(lambda x: ('exp' in x['batch_r_max_expression']) or 'log' in (x['batch_r_max_expression']), axis=1)


    limit = 10
    n_bl = sum((results_PH['ntokens'] <= limit)) # number of expression below limit
    n_bl_converged = sum((results_PH['ntokens'] <= limit) & (results_PH['CFD_sumerr'] < 3))

    n_bl += sum((results_CBFS['ntokens'] <= limit)) # number of expression below limit
    n_bl_converged += sum((results_CBFS['ntokens'] <= limit) & (results_CBFS['CFD_sumerr'] < 3))

    n_bl += sum((results_CD['ntokens'] <= limit)) # number of expression below limit
    n_bl_converged += sum((results_CD['ntokens'] <= limit) & (results_CD['CFD_sumerr'] < 3))

    n_al = sum((results_PH['ntokens'] > limit)) # number of expression above limit
    n_al_converged = sum((results_PH['ntokens'] > limit) & (results_PH['CFD_sumerr'] < 3))

    n_al += sum((results_CBFS['ntokens'] > limit)) # number of expression above limit
    n_al_converged += sum((results_CBFS['ntokens'] > limit) & (results_CBFS['CFD_sumerr'] < 3))

    n_al += sum((results_CD['ntokens'] > limit)) # number of expression above limit
    n_al_converged += sum((results_CD['ntokens'] > limit) & (results_CD['CFD_sumerr'] < 3))


    n_cor = sum(results_PH['correct_dim'] == True)
    n_incor = sum(results_PH['correct_dim'] == False)


    results_PH.to_csv('../logs_completed/tmp_ph.csv')
    results_CD.to_csv('../logs_completed/tmp_cd.csv')
    results_CBFS.to_csv('../logs_completed/tmp_cbfs.csv')


def write_OF_model_file(expression, model_nr):

    example_file = '../logs_completed/models/model0000.C'

    with open(example_file) as f:
        lines = f.readlines()

    header_lines = lines[:-2]

    header_lines.append(f'    Info << "Using DSR model {model_nr}" << endl;\n')

    if 'x' in expression:
        # needs replacing
        print('Make sure you got the right inputs!!!!!')
        inputs = ["grad_u_T1", "grad_u_T2", "grad_u_T3", "grad_u_T4", "k", "inv1", "inv2"] ## kDeficit
        expression = convert_expression(expression, inputs)

    header_lines.append(f'    kDeficit_ = {expression};\n')

    with open(f'../logs_completed/models/model{model_nr:04d}.C', 'w') as newfile:
        for line in header_lines:
            newfile.write(line)

    newfile.close()

def write_selected_models_to_C(path):

    df_selected_models = pd.read_csv(path)

    for index, row in df_selected_models.iterrows():

        filename = f'model_{row["model_nr"]}'
        row.to_csv(os.path.join('../logs_completed/models/model_info_files', filename), header=False)

        write_OF_model_file(row['batch_r_max_expression'], row['model_nr'])

    for index, row in df_selected_models.iterrows():
        print(f'    else if (model_nr_.value() == {row["model_nr"]}) {{')
        print(f'        #include "model{row["model_nr"]:04d}.C"')
        print('    }')

def scatter_ntokens_r_max(logdir):

    dirlist = os.listdir(logdir)

    tokens = []
    r_max = []
    r_sum = []

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

        df_joined['name'] = run_name
        df_joined['output'] = output
        df_joined['training_case'] = case
        df_joined['skip_wall'] = sw

        if 'tokens' in df_joined.columns:
            df_joined['ntokens'] = df_joined.apply(lambda x: count_tokens(x['tokens'], ntok), axis=1)
        else:
            df_joined['ntokens'] = ntok

        tokens.append(df_joined['ntokens'].values)
        r_max.append(df_joined['r_max_PH'].values)
        r_sum.append(df_joined['r_sum'].values)

    tokens = np.concatenate(tokens, axis=0)
    r_max = np.concatenate(r_max, axis=0)
    r_sum = np.concatenate(r_sum, axis=0)

    markersize = 25
    lw = 1.5
    width = 10
    figsize = (width, 3*width/4)
    cm = 1 / 2.54  # centimeters in inches

    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    plt.xlabel(r"$n_{tokens}$")
    plt.ylabel(r"$r \;(\tau)$")
    plt.xticks(np.arange(0,25,5))
    plt.yticks(np.arange(0,1,0.05))
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid('both', linestyle=':')
    plt.scatter(tokens, r_max, s=markersize)
    plt.savefig(f'../logs_completed/aa_plots/ntokens_r_max.pdf', format='pdf', bbox_inches='tight')


    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    plt.xlabel(r"$n_{tokens}$")
    plt.ylabel(r"$r_{sum} \;(\tau)$")
    plt.xticks(np.arange(0,25,5))
    plt.yticks(np.arange(0,5,0.5))
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid('both', linestyle=':')
    plt.scatter(tokens, r_sum, s=markersize)
    plt.savefig(f'../logs_completed/aa_plots/ntokens_r_sum.pdf', format='pdf', bbox_inches='tight')

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
        tokens = row['tokens_search']
        if tokens == '[0 0 0]':
            continue

        tokens = tokens.replace('[', '')
        tokens = tokens.replace(']', '')

        tokens = [int(token) for token in tokens.split(' ') if token.isnumeric()]

        for token in tokens:
            results[names[token]] += 1

    return results

def plot_token_distribution():

    for case in ['PH', 'CD', 'CBFS']:
        file = f'../logs_completed/kDef_{case}/kDef_{case}_selected_models_CFD_results_added_tokens.csv'
        df_case = pd.read_csv(file)
        df_models = pd.concat([df_models, df_case], axis=0, ignore_index=True)

    df_models = pd.read_csv(file)

    df_models['CFD_err_sum'] = df_models['CD_nmse'] + df_models['PH_nmse'] + df_models['CBFS_nmse']

    df_improved = df_models[df_models['CFD_err_sum'] < 3]
    df_worse = df_models[df_models['CFD_err_sum'] >= 3]

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
    plt.bar(base_x , np.array(list(results_improved.values())) / (sum(results_improved.values())), color='C1', width=barwidth, label=r'$\varepsilon_{sum} < 3$')
    plt.bar(base_x + shift, np.array(list(results_worse.values())) / (sum(results_worse.values())), color='C2', width=barwidth, label=r'$\varepsilon_{sum} \geq 3$')
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.set_xticks(base_x)
    ax.set_yticks(np.arange(0, 0.25, 0.05))
    ax.set_xticklabels(token_labels)
    ax.tick_params(axis='x', which='major', labelsize=9)
    plt.grid('both', linestyle=':')
    plt.legend()  # prop={'size': 8}, loc='center right', bbox_to_anchor=(1.55, 0.5))
    plt.savefig(f'../logs_completed/aa_plots/token_distribution.eps', format='eps', bbox_inches='tight')

if __name__ == "__main__":

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    matplotlib.use('tkagg')

    #
    #
    # models_path = '../logs_completed/all_CBFS/kDef_CBFS_selected_models.csv'
    # write_selected_models_to_C(models_path)
    #
    # logdir = '../logs_completed/all_CD'
    # summarise_results(logdir)
    #
    # logdir = '../logs_completed/kDef_PH_ntokens'
    # scatter_ntokens_r_max(logdir)
    #
    # result_file = '../logs_completed/kDef_PH/kDef_PH_selected_models_CFD_results.csv'
    # scatter_ntokens_CFDerror(result_file)
   #
    plot_token_distribution()
   #  add_tokens(f'../logs_completed/kDef_PH_ntokens',
   #             f'../logs_completed/kDef_PH_ntokens/kDef_PH_selected_models_CFD_results_added_tokens.csv')
   #
   #  # below to add tokens to model file retrospectively
   #  for case in ['PH', 'CD', 'CBFS']:
   #      add_tokens(f'../logs_completed/kDef_{case}',
   #                 f'../logs_completed/kDef_{case}/kDef_{case}_selected_models_CFD_results.csv')


    print('end')