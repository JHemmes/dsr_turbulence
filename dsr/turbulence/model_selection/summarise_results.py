
import os
import platform
import json
import numpy as np
from sympy.physics.units import Dimension
from sympy.physics.units.systems.si import dimsys_SI
from sympy import log, exp

import pandas as pd

from dsr.turbulence.resultprocessing import load_iterations


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


if __name__ == "__main__":

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    #
    #
    # models_path = '../logs_completed/all_PH/selected_models.csv'
    # write_selected_models_to_C(models_path)

    logdir = '../logs_completed/all_PH'
    summarise_results(logdir)

    print('end')




