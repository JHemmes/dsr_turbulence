import os
import sys

import matplotlib
import pandas as pd
import platform
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import fluidfoam
import scipy.interpolate as interp
import pickle
from matplotlib.ticker import FormatStrFormatter

def custom_readsymmtensor(path):

    with open(path) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        line = line.strip(')\n')
        line = line.strip('(')
        split_line = line.split(' ')
        if len(split_line) == 6:
            data.append([float(val) for val in split_line])

    return np.array(data)

def custom_wallShearStress(path):

    with open(path) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if ')' in line and '(' in line:
            line = line.strip(')\n')
            line = line.strip('(')
            split_line = line.split(' ')
            if len(split_line) == 3:
                data.append([float(val) for val in split_line])

    return np.array(data)


def read_case_results(case_dir):
    dirlist_case = os.listdir(case_dir)

    # get last written solution:
    sol_dirs = [int(sol) for sol in dirlist_case if sol.isnumeric()]
    last_sol_dir = f'{max(sol_dirs)}'

    if last_sol_dir == '0':
        return 'Diverged', last_sol_dir

    sol_files = [f for f in os.listdir(os.path.join(case_dir,last_sol_dir)) if os.path.isfile(os.path.join(case_dir,last_sol_dir, f))]

    tensors = ['tauij']
    vectors = ['U', 'U_LES']
    scalars = ['k', 'k_LES', 'omega', 'nut', 'p', 'phi']

    results = {}
    for file in sol_files:
        if file in scalars:
            results[file] = fluidfoam.readscalar(case_dir, last_sol_dir, file)
        if file in vectors:
            results[file] = fluidfoam.readvector(case_dir, last_sol_dir, file)
        if file in tensors:
            results[file] = custom_readsymmtensor(os.path.join(case_dir, last_sol_dir, file))
        if file == 'wallShearStress':
            results[file] = custom_wallShearStress(os.path.join(case_dir, last_sol_dir, file))


    # get line_results from postprocessing dir:
    pp_dir = os.path.join(case_dir, 'postProcessing')
    post_processed = False
    try:
        pp_dirlist = os.listdir(pp_dir)
        post_processed = True
    except FileNotFoundError:
        print('DIVERGED OR DID NOT RUN!')
        print(case_dir)

    results['pp'] = {}
    if post_processed:
        for d in pp_dirlist:

            if d == 'wallShearStress':
                continue

            if d == 'residuals':
                try:
                    data = np.genfromtxt(os.path.join(pp_dir, d, '0', 'residuals_1.dat'))
                except OSError:
                    data = np.genfromtxt(os.path.join(pp_dir, d, '0', 'residuals.dat'))
                results['pp'][d] = data
            else:
                results['pp'][d] = {}
                pp_files = os.listdir(os.path.join(pp_dir, d, last_sol_dir))
                for file in pp_files:
                    data = np.genfromtxt(os.path.join(pp_dir, d, last_sol_dir, file))
                    results['pp'][d][file.split('.')[0]] = data

    return results, last_sol_dir

def reshape_to_mesh(array):
    if len(array) == 120*130:
        return np.reshape(array, (120, 130), order='F')
    elif len(array) == 140*100:
        return np.reshape(array, (140, 100), order='F')
    elif len(array) == 140*150:
        return np.reshape(array, (140, 150), order='F')
    else:
        print('Unknown mesh')

def find_model_info(dir):

    # find model name:
    dirlist = os.listdir(dir)
    for file in dirlist:
        if 'model' in file:
            model_file = file
            break

    with open(os.path.join(dir, model_file)) as f:
        lines = f.readlines()

    model_info = {'dir': dir}
    if len(lines) > 10:
        for line in lines:
            line = line.strip('\n')
            split_line = line.split(',')
            if split_line[0] == 'dimensions':
                model_info[split_line[0]] = ','.join([val.strip('"') for val in split_line[1:]])
            else:
                model_info[split_line[0]] = split_line[1]

    if len(model_info) == 1:  # no model info found
        name = dir.split('/')[-1]
        if name[:7] == 'Re37000':
            name = name[8:]
    else:
        model_type = model_info['name'].split('_')[0]
        name = f'{model_info["training_case"]}_{model_type}_{model_info["model_nr"]}'


    return name, model_info

def process_OF_results(selected_model_file=False):

    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/dsr_models/'
    dirlist = os.listdir(base_dir)
    dirlist.pop(dirlist.index('common'))
    dirlist.pop(dirlist.index('base_dir'))
    # dirlist.pop(dirlist.index('base_dir_no_dimension_check'))

    cases = ['CD', 'PH', 'CBFS']

    kOmegaSST = {}  # will contain standard kOmegaSST results
    hifi_data = {}

    mse = lambda x, y: sum(sum((x - y)**2))/(x.shape[0]*x.shape[1])

    for case in cases:
        # get baseline kOmegaSST data
        case_result, final_iteration = read_case_results(os.path.join(base_dir, 'kOmegaSST', case))
        kOmegaSST[case] = case_result

        # get high fidelity data
        hifi_data[case] = {
            'field':np.genfromtxt(os.path.join(base_dir, 'common', f'{case}_field.csv'), delimiter=','),
            'lines':np.genfromtxt(os.path.join(base_dir, 'common', f'{case}_lines.csv'), delimiter=',')
        }

        # if case == 'CBFS':
        #     # clip domain for MSE caluculation:
        #     xmin = -10 # 0
        #     xmax = 20  # 9
        #     ymin = -5  # 0
        #     ymax = 100 # 3
        #     x = hifi_data[case]['field'][:, 0]
        #     y = hifi_data[case]['field'][:, 1]
        #     keep_points = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
        # else:
        #     keep_points = np.ones(hifi_data[case]['field'][:, 0].shape) == 1
        #
        U = np.moveaxis(hifi_data[case]['field'][:, 2:], -1, 0)
        keep_points = ~np.isnan(U).any(axis=0)
        hifi_data[case]['keep'] = keep_points
        hifi_data[case]['U'] = U[:, hifi_data[case]['keep']]

        hifi_data[case]['mse_kOmegaSST'] = mse(hifi_data[case]['U'],
                                               kOmegaSST[case]['U'][:2, hifi_data[case]['keep']])

        # keep_points = ~np.any(np.isnan(hifi_data[case]['field']), axis=1)
        # hifi_data[case]['keep'] = keep_points
        # hifi_data[case]['U'] = np.moveaxis(hifi_data[case]['field'][hifi_data[case]['keep'], 2:], -1, 0)
        # hifi_data[case]['mse_kOmegaSST'] = mse(hifi_data[case]['U'],
        #                                        kOmegaSST[case]['U'][:2, hifi_data[case]['keep']])

    save_lists = {
        'model_order': [],
        'PH_mse': [],
        'PH_iter': [],
        'CD_mse': [],
        'CD_iter': [],
        'CBFS_mse': [],
        'CBFS_iter': []
    }

    if selected_model_file:
        df_models = pd.read_csv(selected_model_file)

    results = {}
    for dir in dirlist:

        if 'Re37000' in dir:
            continue
        
        if 'combined' in dir:
            continue

        name, model_info = find_model_info(os.path.join(base_dir, dir))

        if len(model_info) > 10:
            if int(model_info['model_nr']) not in df_models['model_nr'].values:
                continue


        results[name] = {'model_info': model_info}

        for case in cases:

            case_result, final_iteration = read_case_results(os.path.join(base_dir, dir, case))
            if case_result == 'Diverged':
                results[name][case] = {'norm_mse': 1000000,
                                       'final_iteration': final_iteration}
            else:
                results[name][case] = {'norm_mse': mse(hifi_data[case]['U'],
                                                       case_result['U'][:2, hifi_data[case]['keep']]),
                                       'final_iteration': final_iteration}
                results[name][case]['norm_mse'] = results[name][case]['norm_mse']/hifi_data[case]['mse_kOmegaSST']

        if len(model_info) > 10:
            if int(model_info['model_nr']) in df_models['model_nr'].values:
                save_lists['model_order'].append(int(model_info['model_nr']))
                for case in cases:
                    save_lists[f'{case}_mse'].append(results[name][case]['norm_mse'])
                    save_lists[f'{case}_iter'].append(results[name][case]['final_iteration'])

    if selected_model_file:
        # perform additional check to ensure the right data is with the right model:
        df_models['model_check'] = [x for _,x in sorted(zip(save_lists['model_order'],
                                                            [f'{ii}' for ii in save_lists['model_order']]))]

        for case in cases:
            df_models[f'{case}_nmse'] = [x for _, x in sorted(zip(save_lists['model_order'],
                                                                  save_lists[f'{case}_mse']))]
            df_models[f'{case}_iter'] = [x for _, x in sorted(zip(save_lists['model_order'],
                                                                  save_lists[f'{case}_iter']))]

        file_name = selected_model_file.split('.')[0] + '_CFD_results.csv'
        df_models.to_csv(file_name, index=False)

    return results, hifi_data

def process_combined_models():

    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/dsr_models/'
    dirlist = os.listdir(base_dir)
    dirlist.pop(dirlist.index('common'))
    dirlist.pop(dirlist.index('base_dir'))
    # dirlist.pop(dirlist.index('base_dir_no_dimension_check'))

    cases = ['CD', 'PH', 'CBFS']

    kOmegaSST = {}  # will contain standard kOmegaSST results
    hifi_data = {}

    mse = lambda x, y: sum(sum((x - y)**2))/(x.shape[0]*x.shape[1])

    for case in cases:
        # get baseline kOmegaSST data
        case_result, final_iteration = read_case_results(os.path.join(base_dir, 'kOmegaSST', case))
        kOmegaSST[case] = case_result

        # get high fidelity data
        hifi_data[case] = {
            'field':np.genfromtxt(os.path.join(base_dir, 'common', f'{case}_field.csv'), delimiter=','),
            'lines':np.genfromtxt(os.path.join(base_dir, 'common', f'{case}_lines.csv'), delimiter=',')
        }

        # if case == 'CBFS':
        #     # clip domain for MSE caluculation:
        #     xmin = -10 # 0
        #     xmax = 20  # 9
        #     ymin = -5  # 0
        #     ymax = 100 # 3
        #     x = hifi_data[case]['field'][:, 0]
        #     y = hifi_data[case]['field'][:, 1]
        #     keep_points = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
        # else:
        #     keep_points = np.ones(hifi_data[case]['field'][:, 0].shape) == 1

        U = np.moveaxis(hifi_data[case]['field'][:, 2:], -1, 0)
        keep_points = ~np.isnan(U).any(axis=0)
        hifi_data[case]['keep'] = keep_points
        hifi_data[case]['U'] = U[:, hifi_data[case]['keep']]

        hifi_data[case]['mse_kOmegaSST'] = mse(hifi_data[case]['U'],
                                               kOmegaSST[case]['U'][:2, hifi_data[case]['keep']])

        # keep_points = ~np.any(np.isnan(hifi_data[case]['field']), axis=1)
        # hifi_data[case]['keep'] = keep_points
        # hifi_data[case]['U'] = np.moveaxis(hifi_data[case]['field'][hifi_data[case]['keep'], 2:], -1, 0)
        # hifi_data[case]['mse_kOmegaSST'] = mse(hifi_data[case]['U'],
        #                                        kOmegaSST[case]['U'][:2, hifi_data[case]['keep']])

    save_lists = {
        'model_order': [],
        'PH_mse': [],
        'PH_iter': [],
        'CD_mse': [],
        'CD_iter': [],
        'CBFS_mse': [],
        'CBFS_iter': []
    }

    results = {}
    for dir in dirlist:

        if 'Re37000' in dir:
            continue

        name, model_info = find_model_info(os.path.join(base_dir, dir))

        if len(model_info) > 10:
            continue


        results[name] = {'model_info': model_info}

        for case in cases:

            case_result, final_iteration = read_case_results(os.path.join(base_dir, dir, case))
            if case_result == 'Diverged':
                results[name][case] = {'norm_mse': 1000000,
                                       'final_iteration': final_iteration}
            else:
                results[name][case] = {'norm_mse': mse(hifi_data[case]['U'],
                                                       case_result['U'][:2, hifi_data[case]['keep']]),
                                       'final_iteration': final_iteration}
                results[name][case]['norm_mse'] = results[name][case]['norm_mse']/hifi_data[case]['mse_kOmegaSST']

    df_results = pd.DataFrame()

    for model in results:
        if 'combined' not in model:
            continue

        df_row = pd.DataFrame()
        df_row['kDef_model'] = [model.split('_')[1]]
        df_row['bDel_model'] = [model.split('_')[2]]

        for case in cases:
            df_row[f'{case}_nmse'] = [results[model][case]['norm_mse']]
            df_row[f'{case}_iter'] = [results[model][case]['final_iteration']]

        df_results = pd.concat([df_results, df_row], axis=0, ignore_index=True)
        # file_name = selected_model_file.split('.')[0] + '_CFD_results.csv'
        # df_models.to_csv(file_name, index=False)




    # if selected_model_file:
    #     # perform additional check to ensure the right data is with the right model:
    #     df_models['model_check'] = [x for _,x in sorted(zip(save_lists['model_order'],
    #                                                         [f'{ii}' for ii in save_lists['model_order']]))]
    #
    #     for case in cases:
    #         df_models[f'{case}_nmse'] = [x for _, x in sorted(zip(save_lists['model_order'],
    #                                                               save_lists[f'{case}_mse']))]
    #         df_models[f'{case}_iter'] = [x for _, x in sorted(zip(save_lists['model_order'],
    #                                                               save_lists[f'{case}_iter']))]
    #
    #     file_name = selected_model_file.split('.')[0] + '_CFD_results.csv'
    #     df_models.to_csv(file_name, index=False)
    df_results = df_results.sort_values(['kDef_model', 'bDel_model'], ascending=[True, True], ignore_index=True)
    file_name = '../logs_completed/aa_plots/combined_models_CFD_results.csv'
    df_results.to_csv(file_name, index=False)

def add_scatter(x, df, plot_col, color, markersize, lw, label_once, marker):

    first_label = True
    for ii in range(len(x)):
        label = None
        if df['correct_dim'].values[ii]:
            face_col = color
            if first_label:
                label = label_once
                first_label = False
        else:
            face_col = 'none'
        plt.scatter(x[ii], df[plot_col].values[ii], c=face_col, edgecolors=color, s=markersize,
                    linewidth=lw, label=label, marker=marker)

def results_scatter(selected_model_file):

    df_models = pd.read_csv(selected_model_file)

    if len(df_models['training_case'].unique()) == 1:
        training_case = df_models['training_case'].unique()[0]

    if training_case == 'PH':
        sort_by_CFD = ['PH_nmse', 'CD_nmse', 'CBFS_nmse']
        sort_by_r_max = ['r_max_PH', 'r_max_CD', 'r_max_CBFS']
    elif training_case == 'CD':
        sort_by_CFD = ['CD_nmse', 'PH_nmse', 'CBFS_nmse']
        sort_by_r_max = ['r_max_CD', 'r_max_PH', 'r_max_CBFS']
    else:
        sort_by_CFD = ['CBFS_nmse', 'PH_nmse', 'CD_nmse']
        sort_by_r_max = ['r_max_CBFS', 'r_max_PH', 'r_max_CD']

    for col in ['PH_nmse', 'CBFS_nmse', 'CD_nmse']:
        df_models.loc[df_models[col] == 'Diverged', col] = 1000
        df_models.loc[df_models[col] > 1000, col] = 1000

    df_sorted = df_models.sort_values(sort_by_CFD,
                                      ascending=[True, True, True], ignore_index=True)

    x = np.arange(df_sorted.shape[0]) + 1

    best_sparta = {'PH': {'CD': 0.246319164597, 'PH': 0.16591760527490615, 'CBFS': 0.40664727572452286},
                   'CD': {'CD': 0.22708994859535858, 'PH': 0.25459681231336095, 'CBFS': 0.38039621733135776},
                   'CBFS': {'CD': 0.2081585409088, 'PH': 0.20329225923, 'CBFS': 0.5792927510006327}
    }

    # prepare info for filename:
    if len(np.unique([name[:4] for name in df_sorted['name'].values])) == 1:
        model_type = np.unique([name[:4] for name in df_sorted['name'].values])[0]
    else:
        ValueError('Too many models in the input file')
    if len(df_sorted['training_case'].unique()) == 1:
        training_case = df_sorted['training_case'].unique()[0]
    else:
        ValueError('Too many models in the input file')

    # df_sorted['CFD_sum'] = df_sorted['PH_nmse'] + df_sorted['CD_nmse'] + df_sorted['CBFS_nmse']

    markersize = 30
    lw = 1.5
    figsize = (24, 6)
    cm = 1 / 2.54  # centimeters in inches

    if model_type == 'bDel':
        ylim = [0, 3]
    elif model_type == 'kDef':
        ylim = [0,1]

    # plot CFD errors:
    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    add_scatter(x, df_sorted, 'PH_nmse', 'C0', markersize, lw, r'$PH_{10595}$', 'd')
    add_scatter(x, df_sorted, 'CD_nmse', 'C1', markersize, lw, r'$CD_{12600}$', '^')
    add_scatter(x, df_sorted, 'CBFS_nmse', 'C2', markersize, lw, r'$CBFS_{13700}$', 'v')

    plt.ylabel(r'$\varepsilon (U) / \varepsilon(U_0)$')
    plt.xlabel('Models')
    plt.ylim(ylim)
    plt.xticks(np.arange(0,100,10))
    ax = plt.gca()
    ax.xaxis.grid(linestyle=':')
    # plt.scatter(10, 10, c='none', edgecolors='grey', s=markersize, linewidth=lw,
    #             label='Incorrect dimensionality')
    plt.axhline(y=best_sparta[training_case]['PH'], color='C0', linestyle=(0, (5, 1)), label=r'SpaRTA $PH_{10595}$', linewidth = lw) # densely dashed
    plt.axhline(y=best_sparta[training_case]['CD'], color='C1', linestyle=(0, (1, 1)), label=r'SpaRTA $CD_{12600}$', linewidth = lw)
    plt.axhline(y=best_sparta[training_case]['CBFS'], color='C2', linestyle=(0, (3, 1, 1, 1, 1, 1)), label=r'SpaRTA $CBFS_{13700}$', linewidth = lw)
    plt.legend(prop={'size': 8})
    plt.savefig(f'../logs_completed/aa_plots/{training_case}_{model_type}_CFDerror.eps', format='eps', bbox_inches='tight')

    # sort by training reward.
    df_sorted = df_models.sort_values(sort_by_r_max,
                                      ascending=[False, False, False], ignore_index=True)
    markersize = 25
    lw = 1.5
    figsize = (20, 5)
    cm = 1 / 2.54  # centimeters in inches
    x = np.arange(df_sorted.shape[0]) + 1
    best_sparta = {'kDef':{
        'CD': 0.5684797739549532,
        'PH': 0.5462239021080454,
        'CBFS': 0.5159889361354046
    }, 'bDel':{
        'CD': 0.5690503044684558,
        'PH': 0.571695779816266,
        'CBFS': 0.5310863754366221
    }}

    if model_type == 'bDel':
        ylim = [0.4, 0.7]
    elif model_type == 'kDef':
        ylim = [0.4, 0.9]

    ########## bDel TRAINING REWARDS SPARTA:
    #           PH                      CD                      CBFS
    # M1        0.498833622810127       0.4999638861137987      0.4999614012046586
    # M2        0.571695779816266       0.5690503044684558      0.5310863754366221
    # M3        0.498833622810127       0.4999638861137987      0.4999614012046586

    ########## kDef TRAINING REWARDS SPARTA:
    #           PH                      CD                      CBFS
    # M1        0.546223902108045       0.5385473577589898      0.5072504830064989
    # M2        0.448964230868368       0.5323072387386726      0.49652636811025874
    # M3        0.536941200253387       0.5684797739549532      0.5159889361354046

    # plot inv_NRMSE errors:
    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    add_scatter(x, df_sorted, 'r_max_PH', 'C0', markersize, lw, r'$PH_{10595}$', 'd')
    add_scatter(x, df_sorted, 'r_max_CD', 'C1', markersize, lw, r'$CD_{12600}$', '^')
    add_scatter(x, df_sorted, 'r_max_CBFS', 'C2', markersize, lw, r'$CBFS_{13700}$', 'v')

    plt.ylabel(r'$r_{max}$')
    plt.xlabel('Models')
    plt.ylim(ylim)
    plt.xticks(np.arange(0,100,10))
    ax = plt.gca()
    ax.xaxis.grid(linestyle=':')
    # plt.scatter(10, 10, c='none', edgecolors='grey', s=markersize, linewidth=lw,
    #             label='Incorrect dimensionality')
    plt.axhline(y=best_sparta[model_type]['PH'], color='C0', linestyle=(0, (5, 1)), label=r'SpaRTA $PH_{10595}$', linewidth = lw) # densely dashed
    plt.axhline(y=best_sparta[model_type]['CD'], color='C1', linestyle=(0, (1, 1)), label=r'SpaRTA $CD_{12600}$', linewidth = lw)
    plt.axhline(y=best_sparta[model_type]['CBFS'], color='C2', linestyle=(0, (3, 1, 1, 1, 1, 1)), label=r'SpaRTA $CBFS_{13700}$', linewidth = lw)
    plt.legend(prop={'size': 8}, loc='center right', bbox_to_anchor=(1.3, 0.5))
    # plt.legend(prop={'size': 8}, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=7)
    plt.savefig(f'../logs_completed/aa_plots/{training_case}_{model_type}_r_max.eps', format='eps', bbox_inches='tight')


def bDelta_scatter():

    modelfiles = ['../logs_completed/bDel_PH/bDel_PH_selected_models_CFD_results_full_bDelta.csv',
                  '../logs_completed/bDel_CD/bDel_CD_selected_models_CFD_results_full_bDelta.csv',
                  '../logs_completed/bDel_CBFS/bDel_CBFS_selected_models_CFD_results_full_bDelta.csv']

    markersize = 30
    lw = 1.5
    figsize = (24, 6)
    cm = 1 / 2.54  # centimeters in inches

    ylim = [0, 3]

    # plot CFD errors:
    fig = plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    gs = fig.add_gridspec(1, 3, wspace=0.05)
    axes = gs.subplots(sharey='row')

    for ii in range(len(axes)):
        ax = axes[ii]
        selected_model_file = modelfiles[ii]

        df_models = pd.read_csv(selected_model_file)

        if len(df_models['training_case'].unique()) == 1:
            training_case = df_models['training_case'].unique()[0]

        if training_case == 'PH':
            sort_by_CFD = ['PH_nmse', 'CD_nmse', 'CBFS_nmse']
            xlabel = r'Models trained on $PH_{10595}$'
        elif training_case == 'CD':
            sort_by_CFD = ['CD_nmse', 'PH_nmse', 'CBFS_nmse']
            xlabel = r'Models trained on $CD_{12600}$'
        else:
            sort_by_CFD = ['CBFS_nmse', 'PH_nmse', 'CD_nmse']
            xlabel = r'Models trained on $CBFS_{13700}$'


        for col in ['PH_nmse', 'CBFS_nmse', 'CD_nmse']:
            df_models.loc[df_models[col] == 'Diverged', col] = 1000
            df_models.loc[df_models[col] > 1000, col] = 1000

        df_sorted = df_models.sort_values(sort_by_CFD,
                                          ascending=[True, True, True], ignore_index=True)

        # df_sorted['sum'] = df_sorted['PH_nmse'] + df_sorted['CD_nmse'] + df_sorted['CBFS_nmse']
        # sum(df_sorted['sum'] < 1000)
        df_sorted = df_sorted.head(30)
        x = np.arange(df_sorted.shape[0]) + 1

        best_sparta = {'PH': {'CD': 0.246319164597, 'PH': 0.16591760527490615, 'CBFS': 0.40664727572452286},
                       'CD': {'CD': 0.22708994859535858, 'PH': 0.25459681231336095, 'CBFS': 0.38039621733135776},
                       'CBFS': {'CD': 0.2081585409088, 'PH': 0.20329225923, 'CBFS': 0.5792927510006327}
        }

        first_label = True
        for ii in range(len(x)):
            label = None
            if df_sorted['correct_dim'].values[ii]:
                face_col = 'C0'
                if first_label:
                    label = r'$PH_{10595}$'
                    first_label = False
            else:
                face_col = 'none'
            ax.scatter(x[ii], df_sorted['PH_nmse'].values[ii], c=face_col, edgecolors='C0', s=markersize,
                        linewidth=lw, label=label, marker='d')

        first_label = True
        for ii in range(len(x)):
            label = None
            if df_sorted['correct_dim'].values[ii]:
                face_col = 'C1'
                if first_label:
                    label = r'$CD_{12600}$'
                    first_label = False
            else:
                face_col = 'none'
            ax.scatter(x[ii], df_sorted['CD_nmse'].values[ii], c=face_col, edgecolors='C1', s=markersize,
                        linewidth=lw, label=label, marker='^')

        first_label = True
        for ii in range(len(x)):
            label = None
            if df_sorted['correct_dim'].values[ii]:
                face_col = 'C2'
                if first_label:
                    label = r'$CBFS_{13700}$'
                    first_label = False
            else:
                face_col = 'none'
            ax.scatter(x[ii], df_sorted['CBFS_nmse'].values[ii], c=face_col, edgecolors='C2', s=markersize,
                        linewidth=lw, label=label, marker='v')

        # add_scatter(x, df_sorted, 'PH_nmse', 'C0', markersize, lw, r'$PH_{10595}$', 'd')
        # add_scatter(x, df_sorted, 'CD_nmse', 'C1', markersize, lw, r'$CD_{12600}$', '^')
        # add_scatter(x, df_sorted, 'CBFS_nmse', 'C2', markersize, lw, r'$CBFS_{13700}$', 'v')

        ax.set_xticks(np.arange(0,len(x),10))
        # ax = plt.gca()
        ax.xaxis.grid(linestyle=':')
        # plt.scatter(10, 10, c='none', edgecolors='grey', s=markersize, linewidth=lw,
        #             label='Incorrect dimensionality')
        ax.axhline(y=best_sparta[training_case]['PH'], color='C0', linestyle=(0, (5, 1)), label=r'SpaRTA $PH_{10595}$', linewidth = lw) # densely dashed
        ax.axhline(y=best_sparta[training_case]['CD'], color='C1', linestyle=(0, (1, 1)), label=r'SpaRTA $CD_{12600}$', linewidth = lw)
        ax.axhline(y=best_sparta[training_case]['CBFS'], color='C2', linestyle=(0, (3, 1, 1, 1, 1, 1)), label=r'SpaRTA $CBFS_{13700}$', linewidth = lw)
        ax.set_xlabel(xlabel, fontsize=13)

    axes[0].set_ylabel(r'$\varepsilon (U) / \varepsilon(U_0)$', fontsize=13)
    # plt.xlabel('Models')
    plt.ylim(ylim)
    plt.legend(prop={'size': 10}, loc='center right', bbox_to_anchor=(1.8, 0.5))
    plt.savefig(f'../logs_completed/aa_plots/bDelta_CFD_err_subplots.eps', format='eps', bbox_inches='tight')

def combined_models_scatter():

    df_20 = pd.read_csv('../logs_completed/aa_plots/combined_models_CFD_results_20_percent.csv')
    # df_full = pd.read_csv('../logs_completed/aa_plots/combined_models_CFD_results_20_percent.csv')
    df_full = pd.read_csv('../logs_completed/aa_plots/combined_models_CFD_results_full_bDelta.csv')
    #
    # # check differences for training case:
    # df_full = df_full[(df_full['kDef_model'] >= 300) & (df_full['kDef_model'] < 400)]


    df_full['sum'] = df_full['PH_nmse'] + df_full['CD_nmse'] + df_full['CBFS_nmse']
    df_full['correct_dim'] = True

    sort_by_r_max = ['r_max_CBFS', 'r_max_PH', 'r_max_CD']

    for col in ['PH_nmse', 'CBFS_nmse', 'CD_nmse']:
        df_20.loc[df_20[col] == 'Diverged', col] = 1000
        df_20.loc[df_20[col] > 1000, col] = 1000
        df_full.loc[df_full[col] == 'Diverged', col] = 1000
        df_full.loc[df_full[col] > 1000, col] = 1000

    df_sorted = df_full.sort_values(['sum'],
                                    ascending=[True], ignore_index=True)

    x = np.arange(df_sorted.shape[0]) + 1

    # best_sparta = {'PH': {'CD': 0.246319164597, 'PH': 0.16591760527490615, 'CBFS': 0.40664727572452286},
    #                'CD': {'CD': 0.22708994859535858, 'PH': 0.25459681231336095, 'CBFS': 0.38039621733135776},
    #                'CBFS': {'CD': 0.2081585409088, 'PH': 0.20329225923, 'CBFS': 0.5792927510006327}
    # }
    best_sparta = {'PH': 0.16591760527490615, 'CD': 0.2081585409088, 'CBFS': 0.38039621733135776}

    markersize = 30
    lw = 1.5
    figsize = (24, 6)
    cm = 1 / 2.54  # centimeters in inches

    # plot CFD errors:
    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    add_scatter(x, df_sorted, 'PH_nmse', 'C0', markersize, lw, r'$PH_{10595}$', 'd')
    add_scatter(x, df_sorted, 'CD_nmse', 'C1', markersize, lw, r'$CD_{12600}$', '^')
    add_scatter(x, df_sorted, 'CBFS_nmse', 'C2', markersize, lw, r'$CBFS_{13700}$', 'v')

    plt.ylabel(r'$\varepsilon (U) / \varepsilon(U_0)$')
    plt.xlabel('Models')
    plt.ylim([0,1])
    plt.xticks(np.arange(0,100,10))
    ax = plt.gca()
    ax.xaxis.grid(linestyle=':')
    # plt.scatter(10, 10, c='none', edgecolors='grey', s=markersize, linewidth=lw,
    #             label='Incorrect dimensionality')
    plt.axhline(y=best_sparta['PH'], color='C0', linestyle=(0, (5, 1)), label=r'SpaRTA $PH_{10595}$', linewidth = lw) # densely dashed
    plt.axhline(y=best_sparta['CD'], color='C1', linestyle=(0, (1, 1)), label=r'SpaRTA $CD_{12600}$', linewidth = lw)
    plt.axhline(y=best_sparta['CBFS'], color='C2', linestyle=(0, (3, 1, 1, 1, 1, 1)), label=r'SpaRTA $CBFS_{13700}$', linewidth = lw)
    plt.legend(prop={'size': 8})
    plt.savefig(f'../logs_completed/aa_plots/combined_models_CFDerror.eps', format='eps', bbox_inches='tight')


def plot_selection(plot_list, cases):

    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/dsr_models'

    kOmegaSST = {}  # will contain standard kOmegaSST results
    hifi_data = {}

    mse = lambda x, y: sum(sum((x - y) ** 2)) / (x.shape[0] * x.shape[1])

    for case in cases:
        # get baseline kOmegaSST data
        case_result, final_iteration = read_case_results(os.path.join(base_dir, 'kOmegaSST', case))
        kOmegaSST[case] = case_result

        # get high fidelity data
        hifi_data[case] = {
            'field': np.genfromtxt(os.path.join(base_dir, 'common', f'{case}_field.csv'), delimiter=','),
            'lines': np.genfromtxt(os.path.join(base_dir, 'common', f'{case}_lines.csv'), delimiter=',')
        }

        if case == 'CBFS':
            # clip domain for MSE caluculation:
            xmin = 0
            xmax = 9
            ymin = 0
            ymax = 3
            x = hifi_data[case]['field'][:, 0]
            y = hifi_data[case]['field'][:, 1]
            keep_points = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
        else:
            keep_points = np.ones(hifi_data[case]['field'][:, 0].shape) == 1

        hifi_data[case]['keep'] = keep_points
        hifi_data[case]['U'] = np.moveaxis(hifi_data[case]['field'][hifi_data[case]['keep'], 2:], -1, 0)
        hifi_data[case]['mse_kOmegaSST'] = mse(hifi_data[case]['U'],
                                               kOmegaSST[case]['U'][:2, hifi_data[case]['keep']])

    results = {}
    for dir in plot_list:

        name, model_info = find_model_info(os.path.join(base_dir, dir))

        results[name] = {'model_info': model_info}

        for case in cases:
            case_result, final_iteration = read_case_results(os.path.join(base_dir, dir, case))

            results[name][case] = case_result

            print(f'dir: {dir}')
            print(f'case: {case}')
            print(mse(hifi_data[case]['U'], case_result['U'][:2, hifi_data[case]['keep']]) / hifi_data[case]['mse_kOmegaSST'])
            #
            # results[name][case] = {'norm_mse':
            #                                    ,
            #                        'final_iteration': final_iteration}
            # results[name][case]['norm_mse'] = results[name][case]['norm_mse'] / hifi_data[case]['mse_kOmegaSST']
            #

    dsr1 = 'PH_kDef_138'
    dsr2 = 'PH_kDef_118'
    dsr3 = 'combined_282_662'
    #
    # for key in results.keys():
    #     if key.split('_')[-1][0] == '1':
    #         dsr1 = key
    #     if key.split('_')[-1][0] == '2':
    #         dsr2 = key
    #     if key.split('_')[-1][0] == '3':
    #         dsr3 = key
    #
    # if None in [dsr1, dsr2, dsr3]:
    #     raise FileNotFoundError('did not manage to find correct models')

    for case in cases:
        if case == 'PH':
            best_sparta = 'sparta_model1'
            sparta_label = r'$M^{(1)}_{SpaRTA}$'
            label = 'LES'
            xlim = ylim = [None, None]
            best_dsr = dsr1
            dsr_label = r'$M^{(1)}_{dsr}$'

        elif case == 'CD':
            best_sparta = 'sparta_model3'
            sparta_label = r'$M^{(3)}_{SpaRTA}$'
            label = 'DNS'
            xlim = [5.5, 12.5]
            ylim = [None, None]
            best_dsr = dsr2
            dsr_label = r'$M^{(2)}_{dsr}$'

        elif case == 'CBFS':
            label = 'LES'
            sparta_label = r'$M^{(1)}_{SpaRTA}$'
            best_sparta = 'sparta_model2'
            xlim = [0, 9]
            ylim = [0, 3]
            best_dsr = dsr3
            dsr_label = r'$M^{(3)}_{dsr}$'

        mesh_x_flat, mesh_y_flat = hifi_data[case]['field'][:, 0],  hifi_data[case]['field'][:, 1]
        mesh_x = reshape_to_mesh(mesh_x_flat)
        # mesh_y = reshape_to_mesh(mesh_y_flat)
        n_points = mesh_x.shape[0]

        figsize = (26, 9)
        cm = 1 / 2.54  # centimeters in inches
        plt.figure(figsize=tuple([val * cm for val in list(figsize)]))

        plt.plot(mesh_x_flat[:n_points], mesh_y_flat[:n_points], c='Black')
        plt.plot(mesh_x_flat[-n_points:], mesh_y_flat[-n_points:], c='Black')
        plt.plot([mesh_x_flat[0], mesh_x_flat[-n_points]], [mesh_y_flat[0], mesh_y_flat[-n_points]], c='Black')
        plt.plot([mesh_x_flat[n_points - 1], mesh_x_flat[-1]], [mesh_y_flat[n_points - 1], mesh_y_flat[-1]], c='Black')

        ax = plt.gca()
        ax.set_aspect('equal')
        # add LES results:
        u_scale = 1
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.xlabel(r'$U_{x}/U_b + x/H$')
        plt.ylabel(r'$y/H$')

        for x in np.unique(hifi_data[case]['lines'][:, 0].round()):
            plot_bool = (hifi_data[case]['lines'][:, 0] > x - 0.1) & (hifi_data[case]['lines'][:, 0] < x + 0.1)
            plt.plot(x + hifi_data[case]['lines'][:, 2][plot_bool],
                     hifi_data[case]['lines'][:, 1][plot_bool], c='Black', marker='o', markevery=7, label=label)
            if label:
                label = None

        linewidth = 2
        add_u_profile(results[best_sparta][case]['pp'], 'C1', '--', sparta_label, u_scale, linewidth)
        add_u_profile(results[best_dsr][case]['pp'], 'C0', (0, (3, 1, 1, 1)), dsr_label, u_scale, linewidth)
        add_u_profile(results['kOmegaSST'][case]['pp'], 'C2', ':', r'$k-\omega$ SST', u_scale, linewidth)
        # add_u_profile(results[dsr2][case]['pp'], 'C1', ':', r'$M^{(2)}_{dsr}$', u_scale, linewidth)
        # add_u_profile(results[dsr3][case]['pp'], 'C2', ':', r'$M^{(3)}_{dsr}$', u_scale, linewidth)

        order = [2, 1, 3, 0]
        # order = [1, 0]

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles= [handles[idx] for idx in order], labels=[labels[idx] for idx in order],
            ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9})

        ax = plt.gca()
        ax.set_xticks(np.arange(8), minor=True)
        ax.xaxis.grid(True, which='both', linestyle=':')
        # plt.grid('minor', linestyle=":")

        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.savefig(f'../logs_completed/aa_plots/Ux_{case}.eps', format='eps', bbox_inches='tight')

def plot_experimental():

    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/dsr_models'
    plot_list = ['Re37000_dsr_138', 'Re37000_sparta_model1', 'Re37000_kOmegaSST']
    best_dsr = plot_list[0]

    hifi_data = {}

    # get high fidelity data
    hifi_data['PH'] = {
        'field': np.genfromtxt(os.path.join(base_dir, 'common', f'PH_field.csv'), delimiter=','),
        'lines': np.genfromtxt(os.path.join(base_dir, 'common', f'PH_exp.csv'), delimiter=',')}

    case = 'PH'
    results = {}
    for dir in plot_list:

        name, model_info = find_model_info(os.path.join(base_dir, dir))

        results[name] = {'model_info': model_info}

        case_result, final_iteration = read_case_results(os.path.join(base_dir, dir))

        results[name][case] = case_result

        if dir == best_dsr:
            best_dsr = name

    best_sparta = 'sparta_model1'
    sparta_label = r'$M^{(1)}_{SpaRTA}$'
    dsr_label = r'$M^{(1)}_{dsr}$'

    mesh_x_flat, mesh_y_flat = hifi_data[case]['field'][:, 0], hifi_data[case]['field'][:, 1]
    mesh_x = reshape_to_mesh(mesh_x_flat)
    # mesh_y = reshape_to_mesh(mesh_y_flat)
    n_points = mesh_x.shape[0]

    figsize = (26, 9)
    cm = 1 / 2.54  # centimeters in inches
    plt.figure(figsize=tuple([val * cm for val in list(figsize)]))

    plt.plot(mesh_x_flat[:n_points], mesh_y_flat[:n_points], c='Black')
    plt.plot(mesh_x_flat[-n_points:], mesh_y_flat[-n_points:], c='Black')
    plt.plot([mesh_x_flat[0], mesh_x_flat[-n_points]], [mesh_y_flat[0], mesh_y_flat[-n_points]], c='Black')
    plt.plot([mesh_x_flat[n_points - 1], mesh_x_flat[-1]], [mesh_y_flat[n_points - 1], mesh_y_flat[-1]], c='Black')

    ax = plt.gca()
    ax.set_aspect('equal')
    # add LES results:
    u_scale = 1

    label = 'Experimental'
    for x in np.unique(hifi_data[case]['lines'][:, 0].round()):
        plot_bool = (hifi_data[case]['lines'][:, 0] > x - 0.1) & (hifi_data[case]['lines'][:, 0] < x + 0.1)
        plt.plot(x + hifi_data[case]['lines'][:, 2][plot_bool],
                 hifi_data[case]['lines'][:, 1][plot_bool], c='Black', marker='o', markevery=7, label=label)
        if label:
            label = None

    linewidth = 2
    add_u_profile(results[best_sparta][case]['pp'], 'C1', '--', sparta_label, u_scale, linewidth)
    add_u_profile(results[best_dsr][case]['pp'], 'C0', (0, (3, 1, 1, 1)), dsr_label, u_scale, linewidth)
    add_u_profile(results['kOmegaSST'][case]['pp'], 'C2', ':', r'$k-\omega$ SST', u_scale, linewidth)
    # add_u_profile(results[dsr2][case]['pp'], 'C1', ':', r'$M^{(2)}_{dsr}$', u_scale, linewidth)
    # add_u_profile(results[dsr3][case]['pp'], 'C2', ':', r'$M^{(3)}_{dsr}$', u_scale, linewidth)
    # plt.legend(ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9})

    ax = plt.gca()
    ax.set_xticks(np.arange(8), minor=True)
    ax.xaxis.grid(True, which='both', linestyle=':')
    # plt.grid('minor', linestyle=":")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    plt.xlabel(r'$U_{x}/U_b + x/H$')
    plt.ylabel(r'$y/H$')

    order = [2, 1, 3, 0]
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order],
               ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9})

    plt.savefig(f'../logs_completed/aa_plots/Re37000_Ux.eps', format='eps', bbox_inches='tight')


def add_u_profile(lines, color, linestyle, label, u_scale, linewidth):

        for line in lines:
            if 'single' in line:
                data = lines[line]['line_U']
                plt.plot(data[:, 0] + u_scale * data[:, 3], data[:, 1], c=color,
                         linewidth= linewidth, linestyle=linestyle, markevery=5, label=label)
                if label:
                    label = None

def add_k_profile(lines, color, linestyle, label, scale, linewidth):

        for line in lines:
            if 'single' in line:
                data = lines[line]['line_p_k_omega_kDeficit']
                plt.plot(data[:, 0] + scale * data[:, 4], data[:, 1], c=color,
                         linewidth= linewidth, linestyle=linestyle, markevery=5, label=label)
                if label:
                    label = None


def interpolate_tauij_field(case_dir, mesh_x_flat, mesh_y_flat):

    x_list = []
    y_list = []
    case_result, final_iteration = read_case_results(case_dir)

    for line in case_result['pp']:
        if line == 'residuals':
            continue

        x_list.append(case_result['pp'][line]['line_U'][:, 0])
        y_list.append(case_result['pp'][line]['line_U'][:, 1])

    x_target = np.concatenate(x_list)
    y_target = np.concatenate(y_list)

    x_target[(x_target < 0.1) & (x_target > -0.01)] = 0.05  # to avoid nans

    tauxy_lines = interp.griddata((mesh_x_flat, mesh_y_flat),
                                  case_result['tauij'][:, 1], (x_target, y_target), method='linear')

    return tauxy_lines, x_target, y_target

def plot_Cf(dsr_PH, dsr_CD, dsr_CBFS):

    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/dsr_models/'


    # x         y_wall    C_p       tau_w     delta*    theta     delta99 CBFS data!


    for ii in range(3):
        if ii == 0:
            case = 'PH'
            dsr_model_dir = dsr_PH
            sparta_model_dir = 'sparta_model1'
            frozen = pickle.load(open('turbulence/frozen_data/PH10595_frozen_var.p', 'rb'))
            set_aspect = True
            sparta_label = r'$M^{(1)}_{SpaRTA}$'
            hifi_label = 'LES'
            xlim = [0, 9]
            ylim = [-0.007, 0.02]
            yticks = [0.00, 0.01, 0.02]
            dsr_label = r'$M^{(1)}_{dsr}$'

            markevery = 7
            interpolate = True
            hifi_file = '/home/jasper/OpenFOAM/jasper-7/run/dsr_models/common/cf_PH.csv'

        if ii == 1:
            case = 'CD'
            dsr_model_dir = dsr_CD
            sparta_model_dir = 'sparta_model1'
            frozen = pickle.load(open('turbulence/frozen_data/CD12600_frozen_var.p', 'rb'))
            set_aspect = False
            markevery = 7
            interpolate = True

            sparta_label = r'$M^{(3)}_{SpaRTA}$'
            hifi_label = 'DNS'
            xlim = [2, 12]
            ylim = [-0.006, 0.025]
            yticks = [0.00, 0.01, 0.02]
            dsr_label = r'$M^{(2)}_{dsr}$'
            hifi_file = '/home/jasper/OpenFOAM/jasper-7/run/dsr_models/common/cf_CD.csv'



        if ii == 2:
            case = 'CBFS'
            dsr_model_dir = dsr_CBFS
            markevery = 15

            sparta_model_dir = 'sparta_model2'
            frozen = pickle.load(open('turbulence/frozen_data/CBFS13700_frozen_var.p', 'rb'))
            set_aspect = False

            hifi_label = 'LES'
            sparta_label = r'$M^{(1)}_{SpaRTA}$'
            xlim = [-5, 15]
            ylim = [-0.003, 0.011]
            yticks = [0.00, 0.01]

            dsr_label = r'$M^{(3)}_{dsr}$'
            interpolate = False

            hifi_data = np.genfromtxt('/home/jasper/OpenFOAM/jasper-7/run/dsr_models/common/cf_CBFS.dat')
            hifi_cf = 2*hifi_data[:, 3]
            hifi_x = hifi_data[:, 0]



        case_dir = os.path.join(base_dir, dsr_model_dir, case)   #f'/home/jasper/OpenFOAM/jasper-7/run/dsr_models/{dsr_model_dir}/{case}'

        case_result, final_iteration = read_case_results(case_dir)

        mesh_x_flat, mesh_y_flat, mesh_z_flat = fluidfoam.readof.readmesh(case_dir)

        mesh_x = reshape_to_mesh(mesh_x_flat)
        mesh_y = reshape_to_mesh(mesh_y_flat)
        # bottom_x = np.insert(np.cumsum(np.sqrt(np.diff(mesh_x[:,0])**2 + np.diff(mesh_y[:,0])**2)), 0, 0)
        n_points = mesh_x.shape[0]

        kOmegaSST_dir = os.path.join(base_dir, 'kOmegaSST', case)   #f'/home/jasper/OpenFOAM/jasper-7/run/dsr_models/{dsr_model_dir}/{case}'

        kOmegaSST_result, _ = read_case_results(kOmegaSST_dir)

        sparta_dir = os.path.join(base_dir, sparta_model_dir, case)   #f'/home/jasper/OpenFOAM/jasper-7/run/dsr_models/{dsr_model_dir}/{case}'

        sparta_result, _ = read_case_results(sparta_dir)


        if interpolate:

            hifi_cf = np.genfromtxt(hifi_file, delimiter=',')
            hifi_cf = hifi_cf[hifi_cf[:, 0].argsort()]

            X_Y_Spline = interp.make_interp_spline(hifi_cf[:, 0], hifi_cf[:, 1])

            hifi_x = mesh_x[:, 0]
            hifi_cf = X_Y_Spline(hifi_x)


        figsize = (24, 8)
        cm = 1 / 2.54  # centimeters in inches
        plt.figure(figsize=tuple([val * cm for val in list(figsize)]))

        ax = plt.gca()
        # ax.xaxis.grid(True, which='both', linestyle=':')
        # ax.yaxis.grid(True, which='both', linestyle=':')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(which='both', linestyle=':')
        # ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


        kOmegaSSTlabel = r'$k-\omega$ SST'
        linewidth = 2

        plt.plot(hifi_x,  hifi_cf, c='Black', marker='o', markevery=markevery, label=hifi_label)
        plt.plot(mesh_x[:,0], -2*sparta_result['wallShearStress'][:n_points, 0], color='C1', linestyle='--', label=sparta_label, linewidth=linewidth)
        plt.plot(mesh_x[:,0], -2*case_result['wallShearStress'][:n_points, 0], color='C0', linestyle=(0, (3, 1, 1, 1)), label=dsr_label, linewidth=linewidth)
        plt.plot(mesh_x[:,0], -2*kOmegaSST_result['wallShearStress'][:n_points, 0], color='C2', linestyle=':', label=kOmegaSSTlabel, linewidth=linewidth)

        plt.xlabel(r'$x/H$')
        plt.ylabel(r'$C_f$')
        plt.yticks(yticks)
        # plt.yticks(np.arange(0, ylim[1], 0.01))

        order = [2, 1, 3, 0]
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles= [handles[idx] for idx in order], labels=[labels[idx] for idx in order],
            ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9})
        plt.savefig(f'../logs_completed/aa_plots/Cf_{case}.eps', format='eps', bbox_inches='tight')




def calc_and_plot_shear_stress(dsr_PH, dsr_CD, dsr_CBFS):

    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/dsr_models/'

    for ii in range(3):
        if ii == 0:
            case = 'PH'
            dsr_model_dir = dsr_PH
            sparta_model_dir = 'sparta_model1'
            frozen = pickle.load(open('turbulence/frozen_data/PH10595_frozen_var.p', 'rb'))
            set_aspect = True
            sparta_label = r'$M^{(1)}_{SpaRTA}$'
            hifi_label = 'LES'
            xlim = ylim = [None, None]
            dsr_label = r'$M^{(1)}_{dsr}$'

        if ii == 1:
            case = 'CD'
            dsr_model_dir = dsr_CD
            sparta_model_dir = 'sparta_model1'
            frozen = pickle.load(open('turbulence/frozen_data/CD12600_frozen_var.p', 'rb'))
            set_aspect = False

            sparta_label = r'$M^{(3)}_{SpaRTA}$'
            hifi_label = 'DNS'
            xlim = [5.5, 12.5]
            ylim = [0, 1]
            dsr_label = r'$M^{(2)}_{dsr}$'

        if ii == 2:
            case = 'CBFS'
            dsr_model_dir = dsr_CBFS
            sparta_model_dir = 'sparta_model3'
            frozen = pickle.load(open('turbulence/frozen_data/CBFS13700_frozen_var.p', 'rb'))
            set_aspect = False

            hifi_label = 'LES'
            sparta_label = r'$M^{(1)}_{SpaRTA}$'
            xlim = [0, 7]
            ylim = [0, 1.2]
            dsr_label = r'$M^{(3)}_{dsr}$'

        case_dir = os.path.join(base_dir, dsr_model_dir, case)   #f'/home/jasper/OpenFOAM/jasper-7/run/dsr_models/{dsr_model_dir}/{case}'

        mesh_x_flat, mesh_y_flat, mesh_z_flat = fluidfoam.readof.readmesh(case_dir)

        # interpolate tauij
        dsr_tauxy_lines, dsr_x_target, dsr_y_target = interpolate_tauij_field(case_dir, mesh_x_flat, mesh_y_flat)

        case_dir_sparta = os.path.join(base_dir, sparta_model_dir, case)

        sparta_tauxy_lines, sparta_x_target, sparta_y_target = interpolate_tauij_field(case_dir_sparta, mesh_x_flat, mesh_y_flat)

        case_dir_kOmegaSST = os.path.join(base_dir, 'kOmegaSST', case)

        komg_tauxy_lines, komg_x_target, komg_y_target = interpolate_tauij_field(case_dir_kOmegaSST, mesh_x_flat, mesh_y_flat)

        data_i = frozen['data_i']
        hifi_tauij = data_i['tauij'][0,1,:]

        hifi_tauxy_lines = interp.griddata((mesh_x_flat, mesh_y_flat),
                                       hifi_tauij, (dsr_x_target, dsr_y_target), method='linear')

        mesh_x = reshape_to_mesh(mesh_x_flat)
        n_points = mesh_x.shape[0]

        ####################### To plot
        # mesh_um = reshape_to_mesh(data_i['um'])
        # mesh_vm = reshape_to_mesh(data_i['vm'])
        #
        # CBFS_ICx = mesh_um[0,:]
        # CBFS_ICy = mesh_vm[0,:]
        #
        # for ii in range(len(CBFS_ICx)):
        #     print(f'({CBFS_ICx[ii]} {CBFS_ICy[ii]} 0)')

        ###################### Cf investigation


        hifi_tauxy = data_i['tauij'][0, 1, :]
        # hifi_tauxy = data_i['grad_u'][0, 1, :]

        # hifi_tauxy = data_i['uv']
        mesh_hifi_tau = reshape_to_mesh(hifi_tauxy)
        mesh_y = reshape_to_mesh(mesh_y_flat)

        plt.figure()
        plt.contourf(mesh_x, mesh_y, mesh_hifi_tau, levels=30, cmap='Reds')


        case_result, final_iteration = read_case_results(case_dir)
        #
        mesh_case_tau = reshape_to_mesh(case_result['tauij'][:,1])

        case_result_sparta, _ = read_case_results(os.path.join(base_dir, 'sparta_model1', case) )
        #
        mesh_sparta_tau = reshape_to_mesh(case_result_sparta['tauij'][:,1])

        case_result_kOmegaSST, _ = read_case_results(case_dir_kOmegaSST)
        #
        mesh_kOmegaSST_tau= reshape_to_mesh(case_result_kOmegaSST['tauij'][:,1])


        bottom_x = np.insert(np.cumsum(np.sqrt(np.diff(mesh_x[:,0])**2 + np.diff(mesh_y[:,0])**2)), 0, 0)


        plt.figure()
        # plt.plot(mesh_x_flat[:n_points], hifi_tauij[:n_points])
        # plt.plot(mesh_x[:,0], mesh_hifi_tau[:,0])
        # plt.plot(mesh_x[:,0], mesh_y[:,1])
        plt.plot(bottom_x, -mesh_kOmegaSST_tau[:,0], label='kOmegaSST')
        plt.plot(bottom_x, -mesh_sparta_tau[:,0], label='sparta model 1')
        plt.plot(bottom_x, -mesh_case_tau[:,0], label='dsr model 138')
        plt.legend()
        plt.grid()

        from dsr.turbulence.dataprocessing import calc_sij_rij

        sij, rij = calc_sij_rij(data_i['grad_u'], data_i['omega_frozen'], True)

        k = data_i['k']
        omega = data_i['omega_frozen']
        nut_frozen = data_i['nut_frozen']
        calc_tau  = -2.0 * k *  nut_frozen / k * omega * sij # + data_i['bDelta'] # + 1 / 3 * I);
        hifi_tauxy = calc_tau[0, 1, :]
        mesh_hifi_tau = reshape_to_mesh(hifi_tauxy)
        #
        tau_wall = -data_i['uv']
        mesh_hifi_tau = reshape_to_mesh(tau_wall)
        # # #
        # # tau_wall = data_i['grad_u'] * nut_frozen
        # # hifi_tauxy = tau_wall[0, 1, :]
        # # mesh_hifi_tau = reshape_to_mesh(hifi_tauxy)
        # #
        tau = data_i['tauij'][0,1,:]
        mesh_hifi_tau = reshape_to_mesh(tau)


        scale = 1
        #
        plt.figure()
        # plt.plot(mesh_x_flat[:n_points], hifi_tauij[:n_points])
        plt.title(f'{scale}')
        plt.grid()
        plt.plot(mesh_x[:,0], -mesh_hifi_tau[:,2])
        plt.plot(mesh_x[:,0], -scale*mesh_case_tau[:,0])


        # rotate gradients to align with wall
        # first calculate rotation angle required for each point:
        bot_x = mesh_x[:,0]
        bot_y = mesh_y[:,0]

        bot_x = np.insert(bot_x, 0, 0)
        bot_x = np.append(bot_x, 9)

        bot_y = np.insert(bot_y, 0, bot_y[-1])
        bot_y = np.insert(bot_y, -1, bot_y[-1])


        dy = []
        dx = []

        for ii in range(len(mesh_x[:,0])):
            dx.append(bot_x[ii+2] - bot_x[ii])
            dy.append(bot_y[ii+2] - bot_y[ii])

        dx = np.array(dx)
        dy = np.array(dy)

        thetas = np.arctan(dy/dx)

        indices = np.arange(len(data_i['um']))
        mesh_indices = reshape_to_mesh(indices)
        bot_indices = mesh_indices[:,0]

        tauij = data_i['tauij']
        rotated_bot_tau = []

        for ii in bot_indices:
            theta = thetas[ii]
            R = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
            tau_rot = R*tauij[:,:,ii]*R.transpose()
            rotated_bot_tau.append(tau_rot)

        bottom_x = np.insert(np.cumsum(np.sqrt(np.diff(mesh_x[:,0])**2 + np.diff(mesh_y[:,0])**2)), 0, 0)

        aligned_tau = [tau[0,0] for tau in rotated_bot_tau]
        plt.figure()
        plt.plot(bottom_x, aligned_tau)


         # 2.0 * k_int * (- nut() / k_ * omega_ * S + bijDelta_ + 1 / 3 * I);

        figsize = (26, 9)
        cm = 1 / 2.54  # centimeters in inches
        plt.figure(figsize=tuple([val * cm for val in list(figsize)]))

        plt.plot(mesh_x_flat[:n_points], mesh_y_flat[:n_points], c='Black')
        plt.plot(mesh_x_flat[-n_points:], mesh_y_flat[-n_points:], c='Black')
        plt.plot([mesh_x_flat[0], mesh_x_flat[-n_points]], [mesh_y_flat[0], mesh_y_flat[-n_points]], c='Black')
        plt.plot([mesh_x_flat[n_points - 1], mesh_x_flat[-1]], [mesh_y_flat[n_points - 1], mesh_y_flat[-1]], c='Black')

        ax = plt.gca()

        if set_aspect:
            ax.set_aspect('equal')
        ax.set_xticks(np.arange(8), minor=True)
        ax.xaxis.grid(True, which='both', linestyle=':')
        plt.xlim(xlim)
        plt.ylim(ylim)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        first_label = True

        kOmegaSSTlabel = r'$k-\omega$ SST'

        linewidth = 2
        tau_scale = 20
        for x in np.unique(np.round(dsr_x_target)):
            plot_bool = (dsr_x_target > x - 0.1) & (dsr_x_target < x + 0.1)
            plt.plot(x + tau_scale*hifi_tauxy_lines[plot_bool], dsr_y_target[plot_bool], c='Black', marker='o', markevery=7, label=hifi_label)
            plt.plot(x + tau_scale*sparta_tauxy_lines[plot_bool], dsr_y_target[plot_bool], color='C1', linestyle='--', label=sparta_label, linewidth=linewidth)
            plt.plot(x + tau_scale*dsr_tauxy_lines[plot_bool], dsr_y_target[plot_bool], color='C0', linestyle=(0, (3, 1, 1, 1)), label=dsr_label, linewidth=linewidth)
            plt.plot(x + tau_scale*komg_tauxy_lines[plot_bool], dsr_y_target[plot_bool], color='C2', linestyle=':', label=kOmegaSSTlabel, linewidth=linewidth)
            if first_label:
                hifi_label = sparta_label = dsr_label = kOmegaSSTlabel = None


        plt.xlabel( f'{tau_scale}' + r'$\tau_{xy} + x/H$')
        plt.ylabel(r'$y/H$')

        order = [2, 1, 3, 0]
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles= [handles[idx] for idx in order], labels=[labels[idx] for idx in order],
            ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9})

        # plt.legend(ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9})

        plt.savefig(f'../logs_completed/aa_plots/tauxy_{case}.eps', format='eps', bbox_inches='tight')


def calc_and_plot_k(dsr_PH, dsr_CD, dsr_CBFS):

    for ii in range(3):
        if ii == 0:
            case = 'PH'
            dsr_model_dir = dsr_PH
            sparta_model_dir = 'sparta_model1'
            frozen = pickle.load(open('turbulence/frozen_data/PH10595_frozen_var.p', 'rb'))
            set_aspect = True
            sparta_label = r'$M^{(1)}_{SpaRTA}$'
            hifi_label = 'LES'
            xlim = ylim = [None, None]
            dsr_label = r'$M^{(1)}_{dsr}$'
            k_scale = 10
        if ii == 1:
            case = 'CD'
            dsr_model_dir = dsr_CD
            sparta_model_dir = 'sparta_model1'
            frozen = pickle.load(open('turbulence/frozen_data/CD12600_frozen_var.p', 'rb'))
            set_aspect = False

            sparta_label = r'$M^{(3)}_{SpaRTA}$'
            hifi_label = 'DNS'
            xlim = [5.5, 12.5]
            ylim = [0, 1]
            dsr_label = r'$M^{(2)}_{dsr}$'
            k_scale = 12

        if ii == 2:
            case = 'CBFS'
            dsr_model_dir = dsr_CBFS
            sparta_model_dir = 'sparta_model3'
            frozen = pickle.load(open('turbulence/frozen_data/CBFS13700_frozen_var.p', 'rb'))
            set_aspect = False

            hifi_label = 'LES'
            sparta_label = r'$M^{(1)}_{SpaRTA}$'
            xlim = [0, 7]
            ylim = [0, 1.2]
            dsr_label = r'$M^{(3)}_{dsr}$'
            k_scale = 20

        case_dir = f'/home/jasper/OpenFOAM/jasper-7/run/dsr_models/{dsr_model_dir}/{case}'

        mesh_x_flat, mesh_y_flat, mesh_z_flat = fluidfoam.readof.readmesh(case_dir)

        case_result, final_iteration = read_case_results(case_dir)
        x_list = []
        y_list = []
        for line in case_result['pp']:
            if line == 'residuals':
                continue

            x_list.append(case_result['pp'][line]['line_U'][:, 0])
            y_list.append(case_result['pp'][line]['line_U'][:, 1])

        x_target = np.concatenate(x_list)
        y_target = np.concatenate(y_list)

        x_target[(x_target < 0.1) & (x_target > -0.01)] = 0.05

        sparta_dir = f'/home/jasper/OpenFOAM/jasper-7/run/dsr_models/{sparta_model_dir}/{case}'
        sparta_result, final_iteration = read_case_results(sparta_dir)

        komg_dir = f'/home/jasper/OpenFOAM/jasper-7/run/dsr_models/kOmegaSST/{case}'
        komg_result, final_iteration = read_case_results(komg_dir)

        data_i = frozen['data_i']
        hifi_k = data_i['k']

        hifi_k_lines = interp.griddata((mesh_x_flat, mesh_y_flat),
                                        hifi_k, (x_target, y_target), method='linear')

        mesh_x = reshape_to_mesh(mesh_x_flat)

        n_points = mesh_x.shape[0]

        figsize = (26, 9)
        cm = 1 / 2.54  # centimeters in inches
        plt.figure(figsize=tuple([val * cm for val in list(figsize)]))

        plt.plot(mesh_x_flat[:n_points], mesh_y_flat[:n_points], c='Black')
        plt.plot(mesh_x_flat[-n_points:], mesh_y_flat[-n_points:], c='Black')
        plt.plot([mesh_x_flat[0], mesh_x_flat[-n_points]], [mesh_y_flat[0], mesh_y_flat[-n_points]], c='Black')
        plt.plot([mesh_x_flat[n_points - 1], mesh_x_flat[-1]], [mesh_y_flat[n_points - 1], mesh_y_flat[-1]], c='Black')

        ax = plt.gca()

        if set_aspect:
            ax.set_aspect('equal')
        ax.set_xticks(np.arange(8), minor=True)
        ax.xaxis.grid(True, which='both', linestyle=':')
        plt.xlim(xlim)
        plt.ylim(ylim)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        first_label = True

        linewidth = 2
        for x in np.unique(np.round(x_target)):
            plot_bool = (x_target > x - 0.1) & (x_target < x + 0.1)
            plt.plot(x + k_scale * hifi_k_lines[plot_bool], y_target[plot_bool], c='Black', marker='o', markevery=7, label=hifi_label)
            # plt.plot(x + tau_scale * sparta_tauxy_lines[plot_bool], dsr_y_target[plot_bool], color='C1', linestyle='--',
            #          label=sparta_label, linewidth=linewidth)
            # plt.plot(x + tau_scale * dsr_tauxy_lines[plot_bool], dsr_y_target[plot_bool], color='C0',
            #          linestyle=(0, (3, 1, 1, 1)), label=dsr_label, linewidth=linewidth)
            # plt.plot(x + tau_scale * komg_tauxy_lines[plot_bool], dsr_y_target[plot_bool], color='C2', linestyle=':',
            #          label=kOmegaSSTlabel, linewidth=linewidth)
            if first_label:
                hifi_label = None
        #
        add_k_profile(sparta_result['pp'], 'C1', '--', sparta_label, k_scale, linewidth)
        add_k_profile(case_result['pp'], 'C0', (0, (3, 1, 1, 1)), dsr_label, k_scale, linewidth)
        add_k_profile(komg_result['pp'], 'C2', ':', r'$k-\omega$ SST', k_scale, linewidth)

        order = [2, 1, 3, 0]

        plt.xlabel(f'{k_scale}' + r'$k/U_b^2 + x/H$')
        plt.ylabel(r'$y/H$')

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles=[handles[idx] for idx in order], labels=[labels[idx] for idx in order],
                   ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9})

        # plt.legend(ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9})

        plt.savefig(f'../logs_completed/aa_plots/k_{case}.eps', format='eps', bbox_inches='tight')

        # plt.figure()
        # # plt.plot(mesh_x_flat[:n_points], hifi_tauij[:n_points])
        # plt.plot(mesh_x_flat[:n_points], case_result['tauij'][:n_points, 1])
        # mesh_hifi_tauij = reshape_to_mesh(hifi_tauij)

def plot_reward_analysis():
    df_kdef_PH = pd.read_csv('../logs_completed/kDef_PH/kDef_PH_selected_models_CFD_results.csv')
    df_kdef_PH['sum'] = df_kdef_PH['PH_nmse'] + df_kdef_PH['CD_nmse'] + df_kdef_PH['CBFS_nmse']

    df_kdef_CD = pd.read_csv('../logs_completed/kDef_CD/kDef_CD_selected_models_CFD_results.csv')
    df_kdef_CD['sum'] = df_kdef_CD['PH_nmse'] + df_kdef_CD['CD_nmse'] + df_kdef_CD['CBFS_nmse']

    df_kdef_CBFS = pd.read_csv('../logs_completed/kDef_CBFS/kDef_CBFS_selected_models_CFD_results.csv')
    df_kdef_CBFS['sum'] = df_kdef_CBFS['PH_nmse'] + df_kdef_CBFS['CD_nmse'] + df_kdef_CBFS['CBFS_nmse']

    markersize = 25
    lw = 2
    width = 10
    figsize = (width, 3*width/4)
    cm = 1 / 2.54  # centimeters in inches

    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    models = [df_kdef_PH, df_kdef_CD, df_kdef_CBFS]
    cases = ['PH', 'CD', 'CBFS']
    markers = ['d', '^', 'v']
    labels = [r'$PH_{10595}$', r'$CD_{12600}$', r'$CBFS_{13700}$']
    colors = ['C0', 'C1', 'C2']

    for ii in range(len(cases)):
        df_models = models[ii]
        case = cases[ii]

        df_converged = df_models[df_models['sum'] < 1000]
        plt.scatter(df_converged[f'r_max_{case}'], df_converged[f'{case}_nmse'], color=colors[ii], label=labels[ii], marker=markers[ii])
        plt.ylim([0, 1])
        plt.xlim([0.5, 0.7])

    plt.grid(linestyle=':')
    plt.xlabel(r'$r_{max}$')
    plt.ylabel(r'$\varepsilon (U) / \varepsilon(U_0)$')
    plt.legend()

    plt.savefig(f'../logs_completed/aa_plots/r_max_CFDerr_correlation.eps', format='eps', bbox_inches='tight')

    kDef_conv_percs = []
    kDef_first_n = []
    for ii in range(len(cases)):
        df_models = models[ii]

        if len(df_models['training_case'].unique()) == 1:
            training_case = df_models['training_case'].unique()[0]

        if training_case == 'PH':
            sort_by_CFD = ['PH_nmse', 'CD_nmse', 'CBFS_nmse']
            sort_by_r_max = ['r_max_PH', 'r_max_CD', 'r_max_CBFS']
        elif training_case == 'CD':
            sort_by_CFD = ['CD_nmse', 'PH_nmse', 'CBFS_nmse']
            sort_by_r_max = ['r_max_CD', 'r_max_PH', 'r_max_CBFS']
        else:
            sort_by_CFD = ['CBFS_nmse', 'PH_nmse', 'CD_nmse']
            sort_by_r_max = ['r_max_CBFS', 'r_max_PH', 'r_max_CD']

        for col in ['PH_nmse', 'CBFS_nmse', 'CD_nmse']:
            df_models.loc[df_models[col] == 'Diverged', col] = 1000
            df_models.loc[df_models[col] > 1000, col] = 1000

        df_sorted = df_models.sort_values(sort_by_r_max,
                                          ascending=[False, False, False], ignore_index=True)

        conv_perc = []
        first_n = []
        for ii in np.arange(0, len(df_models), 10):
            first_n.append(ii)
            df_slice = df_sorted.iloc[ii:ii+10, :]
            df_converged = df_slice[df_slice['sum'] < 900]
            conv_perc.append(100 * len(df_converged) / (len(df_slice)))

        kDef_conv_percs.append(conv_perc)
        kDef_first_n.append(first_n)

    df_bDel_PH = pd.read_csv('../logs_completed/bDel_PH/bDel_PH_selected_models_CFD_results_full_bDelta.csv')
    df_bDel_PH['sum'] = df_bDel_PH['PH_nmse'] + df_bDel_PH['CD_nmse'] + df_bDel_PH['CBFS_nmse']

    df_bDel_CD = pd.read_csv('../logs_completed/bDel_CD/bDel_CD_selected_models_CFD_results_full_bDelta.csv')
    df_bDel_CD['sum'] = df_bDel_CD['PH_nmse'] + df_bDel_CD['CD_nmse'] + df_bDel_CD['CBFS_nmse']

    df_bDel_CBFS = pd.read_csv('../logs_completed/bDel_CBFS/bDel_CBFS_selected_models_CFD_results_full_bDelta.csv')
    df_bDel_CBFS['sum'] = df_bDel_CBFS['PH_nmse'] + df_bDel_CBFS['CD_nmse'] + df_bDel_CBFS['CBFS_nmse']

    models = [df_bDel_PH, df_bDel_CD, df_bDel_CBFS]

    bDel_conv_percs = []
    bDel_first_n = []
    for ii in range(len(cases)):
        df_models = models[ii]

        if len(df_models['training_case'].unique()) == 1:
            training_case = df_models['training_case'].unique()[0]

        if training_case == 'PH':
            sort_by_CFD = ['PH_nmse', 'CD_nmse', 'CBFS_nmse']
            sort_by_r_max = ['r_max_PH', 'r_max_CD', 'r_max_CBFS']
        elif training_case == 'CD':
            sort_by_CFD = ['CD_nmse', 'PH_nmse', 'CBFS_nmse']
            sort_by_r_max = ['r_max_CD', 'r_max_PH', 'r_max_CBFS']
        else:
            sort_by_CFD = ['CBFS_nmse', 'PH_nmse', 'CD_nmse']
            sort_by_r_max = ['r_max_CBFS', 'r_max_PH', 'r_max_CD']

        for col in ['PH_nmse', 'CBFS_nmse', 'CD_nmse']:
            df_models.loc[df_models[col] == 'Diverged', col] = 1000
            df_models.loc[df_models[col] > 1000, col] = 1000

        df_sorted = df_models.sort_values(sort_by_r_max,
                                          ascending=[False, False, False], ignore_index=True)

        conv_perc = []
        first_n = []
        for ii in np.arange(0, len(df_models), 10):
            first_n.append(ii)
            df_slice = df_sorted.iloc[ii:ii + 10, :]
            df_converged = df_slice[df_slice['sum'] < 900]
            conv_perc.append(100 * len(df_converged) / (len(df_slice)))

        bDel_conv_percs.append(conv_perc)
        bDel_first_n.append(first_n)


    mean_kDef_first_n = []
    mean_kDef_conv_perc = []

    mean_bDel_first_n = []
    mean_bDel_conv_perc = []

    for ii in range(9):
        mean_kDef_first_n.append(bDel_first_n[0][ii])
        mean_kDef_conv_perc.append(np.mean([lst[ii] for lst in kDef_conv_percs]))

        mean_bDel_first_n.append(bDel_first_n[0][ii])
        mean_bDel_conv_perc.append(np.mean([lst[ii] for lst in bDel_conv_percs]))


    plt.figure(figsize=tuple([val * cm for val in list(figsize)]))

    plt.plot(mean_kDef_first_n, mean_kDef_conv_perc, label=r'$\mathcal{P}_{k}^\Delta$', c='C0', linestyle='--', linewidth=lw, marker='^')
    plt.plot(mean_bDel_first_n, mean_bDel_conv_perc, label=r'$b_{ij}^\Delta$', c='C1', linestyle=':', linewidth=lw, marker='v')

    plt.xlabel(r"Model ensemble")
    plt.ylabel(r'% converging runs')
    plt.grid(linestyle=':')

    plt.legend(loc='upper left', ncol=2)

    plt.savefig(f'../logs_completed/aa_plots/r_max_comvergence.eps', format='eps', bbox_inches='tight')




    print('here')

if __name__ == '__main__':

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    # read_and_plot_PH()
    matplotlib.use('tkagg')
    #

    # plot_reward_analysis()

    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/CBFS'
    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/PH'
    #
    #
    # # # ################### to plot tauij
    # calc_and_plot_shear_stress('dsr_138', 'dsr_118', 'combined_282_662')
    # #
    # # ################### to plot Cf   316 and 726
    # plot_Cf('dsr_138', 'dsr_118', 'combined_282_662')
    # #
    # # ################### to plot k
    # calc_and_plot_k('dsr_138', 'dsr_118', 'combined_282_662')
    # # # #
    # # # ################### to plot RE37000 cases
    # plot_experimental()
    # # # #
    # # # # ################### to plot Ux profiles
    plot_selection(['sparta_model1', 'sparta_model2', 'sparta_model3', 'dsr_138', 'dsr_118', 'combined_282_662', 'kOmegaSST'],
                   ['PH', 'CD', 'CBFS'])

    #
    # ####################### lines below used to add CFD results to selected_models file
    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/kDef_PH/kDef_PH_selected_models.csv'
    # process_OF_results(selected_model_file)
    #
    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/kDef_CD/kDef_CD_selected_models.csv'
    # process_OF_results(selected_model_file)
    #
    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/kDef_CBFS/kDef_CBFS_selected_models.csv'
    # process_OF_results(selected_model_file)

    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/bDel_PH/bDel_PH_selected_models.csv'
    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/bDel_PH/bDel_PH_selected_models_TMP.csv'
    # process_OF_results(selected_model_file)

    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/bDel_CD/bDel_CD_selected_models.csv'
    # process_OF_results(selected_model_file)
    #


    # process_combined_models()
    # combined_models_scatter()
    # bDelta_scatter()

    #
    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/bDel_CBFS/bDel_CBFS_selected_models.csv'
    # process_OF_results(selected_model_file)

    # #################### lines below used to make scatter plots of error in CFD and training rewards.
    # selected_model_file = '../logs_completed/kDef_PH/kDef_PH_selected_models_CFD_results.csv'
    # results_scatter(selected_model_file)
    # # #
    # selected_model_file = '../logs_completed/kDef_CD/kDef_CD_selected_models_CFD_results.csv'
    # results_scatter(selected_model_file)
    #
    # selected_model_file = '../logs_completed/kDef_CBFS/kDef_CBFS_selected_models_CFD_results.csv'
    # results_scatter(selected_model_file)
    # selected_model_file = '../logs_completed/bDel_PH/bDel_PH_selected_models_CFD_results_full_bDelta.csv'
    # # selected_model_file = '../logs_completed/bDel_CBFS/bDel_CBFS_selected_models_CFD_results_20percent_bDelta_weight.csv'
    # results_scatter(selected_model_file)
    # # # #
    # selected_model_file = '../logs_completed/bDel_CD/bDel_CD_selected_models_CFD_results_full_bDelta.csv'
    # results_scatter(selected_model_file)
    # #
    # selected_model_file = '../logs_completed/bDel_CBFS/bDel_CBFS_selected_models_CFD_results_full_bDelta.csv'
    # results_scatter(selected_model_file)

    #
    # df_combined = pd.read_csv('../logs_completed/aa_plots/combined_models_CFD_results_full_bDelta.csv')
    # df_combined['sum'] = df_combined['PH_nmse'] + df_combined['CD_nmse'] + df_combined['CBFS_nmse']
    # df_combined[(df_combined['CD_nmse'] < 0.2271) & (df_combined['PH_nmse'] < 0.2546) & (df_combined['CBFS_nmse'] < 0.3804)]  # sparta model 2 scores
    #


    #
    # ################ DO NOT DELETE, this is used for a plot!!!!
    # df_kdef_PH = pd.read_csv('../logs_completed/kDef_PH/kDef_PH_selected_models_CFD_results.csv')
    # df_kdef_PH['sum'] = df_kdef_PH['PH_nmse'] + df_kdef_PH['CD_nmse'] + df_kdef_PH['CBFS_nmse']
    # df_kdef_PH.reset_index()
    # #
    # #
    # #
    # df_kdef_CD = pd.read_csv('../logs_completed/kDef_CD/kDef_CD_selected_models_CFD_results.csv')
    # df_kdef_CD['sum'] = df_kdef_CD['PH_nmse'] + df_kdef_CD['CD_nmse'] + df_kdef_CD['CBFS_nmse']
    # # df_kdef_CD[(df_kdef_CD['CD_nmse'] < 0.2271) & (df_kdef_CD['PH_nmse'] < 0.2546) & (df_kdef_CD['CBFS_nmse'] < 0.3804)]  # sparta model 2 scores
    # #
    # #
    # df_kdef_CBFS = pd.read_csv('../logs_completed/kDef_CBFS/kDef_CBFS_selected_models_CFD_results.csv')
    # df_kdef_CBFS['sum'] = df_kdef_CBFS['PH_nmse'] + df_kdef_CBFS['CD_nmse'] + df_kdef_CBFS['CBFS_nmse']
    # # df_kdef_CBFS[(df_kdef_CBFS['CD_nmse'] < 0.2082) & (df_kdef_CBFS['PH_nmse'] < 0.2033) & (df_kdef_CBFS['CBFS_nmse'] < 0.5793)]  # sparta model 3 scores
    #
    #
    # df_7tok =  pd.DataFrame()
    # df_10tok =  pd.DataFrame()
    # df_12tok = pd.DataFrame()
    # df_20tok = pd.DataFrame()
    #
    #
    #
    # all_conv_percs = []
    # all_first_n    = []
    # all_mean_PH = []
    # all_mean_CD = []
    # all_mean_CBFS = []
    # all_mean_sum = []
    #
    # for df_models in [df_kdef_PH, df_kdef_CD, df_kdef_CBFS]:
    #
    #     dfappend_7tok = df_models[df_models['name'].str.contains('7tokens')]
    #     dfappend_10tok = df_models[df_models['name'].str.contains('10tokens')]
    #     dfappend_12tok = df_models[df_models['name'].str.contains('12tokens')]
    #     dfappend_20tok = df_models[df_models['name'].str.contains('20tokens')]
    #
    #     df_7tok =  pd.concat([df_7tok, dfappend_7tok], axis=0, ignore_index=True)
    #     df_10tok = pd.concat([df_10tok, dfappend_10tok], axis=0, ignore_index=True)
    #     df_12tok = pd.concat([df_12tok, dfappend_12tok], axis=0, ignore_index=True)
    #     df_20tok = pd.concat([df_20tok, dfappend_20tok], axis=0, ignore_index=True)
    #
    #     if len(df_models['training_case'].unique()) == 1:
    #         training_case = df_models['training_case'].unique()[0]
    #
    #     if training_case == 'PH':
    #         sort_by_CFD = ['PH_nmse', 'CD_nmse', 'CBFS_nmse']
    #         sort_by_r_max = ['r_max_PH', 'r_max_CD', 'r_max_CBFS']
    #     elif training_case == 'CD':
    #         sort_by_CFD = ['CD_nmse', 'PH_nmse', 'CBFS_nmse']
    #         sort_by_r_max = ['r_max_CD', 'r_max_PH', 'r_max_CBFS']
    #     else:
    #         sort_by_CFD = ['CBFS_nmse', 'PH_nmse', 'CD_nmse']
    #         sort_by_r_max = ['r_max_CBFS', 'r_max_PH', 'r_max_CD']
    #
    #     for col in ['PH_nmse', 'CBFS_nmse', 'CD_nmse']:
    #         df_models.loc[df_models[col] == 'Diverged', col] = 1000
    #         df_models.loc[df_models[col] > 1000, col] = 1000
    #
    #     df_sorted = df_models.sort_values(sort_by_r_max,
    #                                       ascending=[False, False, False], ignore_index=True)
    #
    #     mean_PH = []
    #     mean_CD = []
    #     mean_CBFS = []
    #     mean_sum = []
    #
    #     conv_perc = []
    #     first_n = []
    #     for ii in np.arange(0, len(df_models), 10):
    #         first_n.append(ii)
    #         df_slice = df_sorted.iloc[ii:ii+10, :]
    #         df_converged = df_slice[df_slice['sum'] < 900]
    #         conv_perc.append(100 * len(df_converged) / (len(df_slice)))
    #
    #         mean_PH.append(df_converged['PH_nmse'].min())
    #         mean_CD.append(df_converged['CD_nmse'].min())
    #         mean_CBFS.append(df_converged['CBFS_nmse'].min())
    #         mean_sum.append(df_converged['sum'].min())
    #
    #     all_mean_PH.append(mean_PH)
    #     all_mean_CD.append(mean_CD)
    #     all_mean_CBFS.append(mean_CBFS)
    #     all_mean_sum.append(mean_sum)
    #
    #     all_first_n.append(first_n)
    #     all_conv_percs.append(conv_perc)
    #
    # print(f'7Tok converged {100* sum(df_7tok["sum"] < 100) / len(df_7tok)} %')
    # print(f'10Tok converged {100* sum(df_10tok["sum"] < 100) / len(df_10tok)} %')
    # print(f'12Tok converged {100* sum(df_12tok["sum"] < 100) / len(df_12tok)} %')
    # print(f'20Tok converged {100* sum(df_20tok["sum"] < 100) / len(df_20tok)} %')
    #
    # labels = ['PH', 'CD', 'CBFS']
    # plt.figure()
    # plt.title('convergence_percentage')
    # for ii in range(len(all_first_n)):
    #     plt.plot(all_first_n[ii], all_conv_percs[ii], label=labels[ii])
    # plt.legend()
    #
    # labels = ['PH', 'CD', 'CBFS']
    # plt.figure()
    # plt.title('mean PH error')
    # for ii in range(len(all_first_n)):
    #     plt.plot(all_first_n[ii], all_mean_PH[ii], label=labels[ii])
    # plt.legend()
    #
    # labels = ['PH', 'CD', 'CBFS']
    # plt.figure()
    # plt.title('mean CD error')
    #
    # for ii in range(len(all_first_n)):
    #     plt.plot(all_first_n[ii], all_mean_CD[ii], label=labels[ii])
    # plt.legend()
    #
    # labels = ['PH', 'CD', 'CBFS']
    # plt.figure()
    # plt.title('mean CBFS error')
    # for ii in range(len(all_first_n)):
    #     plt.plot(all_first_n[ii], all_mean_CBFS[ii], label=labels[ii])
    # plt.legend()
    #
    # labels = ['PH', 'CD', 'CBFS']
    # plt.figure()
    # plt.title('mean sum error')
    # for ii in range(len(all_first_n)):
    #     plt.plot(all_first_n[ii], all_mean_sum[ii], label=labels[ii])
    # plt.legend()
    #
    #
    #
    #
    # mean_first_n = []
    # mean_conv_perc = []
    #
    # for ii in range(9):
    #     mean_first_n.append(all_first_n[0][ii])
    #     mean_conv_perc.append(np.mean([lst[ii] for lst in all_conv_percs]))
    #
    # #
    # df_bDel_PH = pd.read_csv('../logs_completed/bDel_PH/bDel_PH_selected_models_CFD_results_full_bDelta.csv')
    # df_bDel_PH['sum'] = df_bDel_PH['PH_nmse'] + df_bDel_PH['CD_nmse'] + df_bDel_PH['CBFS_nmse']
    #
    # df_bDel_CD = pd.read_csv('../logs_completed/bDel_CD/bDel_CD_selected_models_CFD_results_full_bDelta.csv')
    # df_bDel_CD['sum'] = df_bDel_CD['PH_nmse'] + df_bDel_CD['CD_nmse'] + df_bDel_CD['CBFS_nmse']
    #
    # df_bDel_CBFS = pd.read_csv('../logs_completed/bDel_CBFS/bDel_CBFS_selected_models_CFD_results_full_bDelta.csv')
    # df_bDel_CBFS['sum'] = df_bDel_CBFS['PH_nmse'] + df_bDel_CBFS['CD_nmse'] + df_bDel_CBFS['CBFS_nmse']
    # # df_bDel_CBFS[(df_bDel_CBFS['CD_nmse'] < 0.2082) & (df_bDel_CBFS['PH_nmse'] < 0.2033) & (
    # #             df_bDel_CBFS['CBFS_nmse'] < 0.5793)]  # sparta model 3 scores
    #
    #
    # df_3tok =  pd.DataFrame()
    # df_5tok =  pd.DataFrame()
    # df_10tok = pd.DataFrame()
    #
    #
    #
    # all_conv_percs = []
    # all_first_n    = []
    #
    #
    # for df_models in [df_bDel_PH, df_bDel_CD, df_bDel_CBFS]:
    #
    #     dfappend_3tok = df_models[df_models['name'].str.contains('3tokens')]
    #     dfappend_5tok = df_models[df_models['name'].str.contains('5tokens')]
    #     dfappend_10tok = df_models[df_models['name'].str.contains('10tokens')]
    #
    #     df_3tok =  pd.concat([df_3tok, dfappend_3tok], axis=0, ignore_index=True)
    #     df_5tok =  pd.concat([df_5tok, dfappend_5tok], axis=0, ignore_index=True)
    #     df_10tok = pd.concat([df_10tok, dfappend_10tok], axis=0, ignore_index=True)
    #
    #
    #     if len(df_models['training_case'].unique()) == 1:
    #         training_case = df_models['training_case'].unique()[0]
    #
    #     if training_case == 'PH':
    #         sort_by_CFD = ['PH_nmse', 'CD_nmse', 'CBFS_nmse']
    #         sort_by_r_max = ['r_max_PH', 'r_max_CD', 'r_max_CBFS']
    #     elif training_case == 'CD':
    #         sort_by_CFD = ['CD_nmse', 'PH_nmse', 'CBFS_nmse']
    #         sort_by_r_max = ['r_max_CD', 'r_max_PH', 'r_max_CBFS']
    #     else:
    #         sort_by_CFD = ['CBFS_nmse', 'PH_nmse', 'CD_nmse']
    #         sort_by_r_max = ['r_max_CBFS', 'r_max_PH', 'r_max_CD']
    #
    #     for col in ['PH_nmse', 'CBFS_nmse', 'CD_nmse']:
    #         df_models.loc[df_models[col] == 'Diverged', col] = 1000
    #         df_models.loc[df_models[col] > 1000, col] = 1000
    #
    #     df_sorted = df_models.sort_values(sort_by_r_max,
    #                                       ascending=[False, False, False], ignore_index=True)
    #
    #     conv_perc = []
    #     first_n = []
    #     for ii in np.arange(0, len(df_models), 10):
    #         first_n.append(ii)
    #         df_slice = df_sorted.iloc[ii:ii+10, :]
    #         conv_perc.append(100 * sum(df_slice['sum'] < 100) / (len(df_slice)))
    #
    #     all_first_n.append(first_n)
    #     all_conv_percs.append(conv_perc)
    #
    # labels = ['PH', 'CD', 'CBFS']
    # plt.figure()
    # for ii in range(len(all_first_n)):
    #     plt.plot(all_first_n[ii], all_conv_percs[ii], label=labels[ii])
    # plt.legend()
    #
    # mean_first_n = []
    # mean_conv_perc = []
    #
    # for ii in range(9):
    #     mean_first_n.append(all_first_n[0][ii])
    #     mean_conv_perc.append(np.mean([lst[ii] for lst in all_conv_percs]))
    #
    # plt.figure()
    # plt.plot(mean_first_n, mean_conv_perc)
    #
    #
    #
    #
    #
    # print(f'3Tok converged {100* sum(df_3tok["sum"] < 100) / len(df_3tok)} %')
    # print(f'5Tok converged {100* sum(df_5tok["sum"] < 100) / len(df_5tok)} %')
    # print(f'10Tok converged {100* sum(df_10tok["sum"] < 100) / len(df_10tok)} %')
    #
    # # df_results = pd.concat([df_results, df_row], axis=0, ignore_index=True)
    #
    #
    # bDel_ntokens = [3, 5, 10]
    # bDel_convergence = [30.008, 13.043, 8.108]
    #
    #
    # kDef_ntokens = [7, 10, 12, 20]
    # kDef_convergence = [100, 64.23, 81.70, 50]
    #
    # markersize = 25
    # lw = 2
    # width = 10
    # figsize = (width, 3*width/4)
    # cm = 1 / 2.54  # centimeters in inches
    #
    # plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    # plt.xlabel(r"$n_{tokens}$")
    # plt.ylabel(r'% converging runs')
    # plt.xticks(np.arange(0,25,2))
    # # plt.yticks(np.arange(0,1,0.1))
    # plt.yticks(np.arange(0,110,20))
    # ax = plt.gca()
    # ax.set_axisbelow(True)
    # plt.grid('both', linestyle=':')
    # plt.plot(kDef_ntokens, kDef_convergence, label=r'$\mathcal{P}_{k}^\Delta$', c='C0', linestyle='--', linewidth=lw, marker='^')
    # plt.plot(bDel_ntokens, bDel_convergence, label=r'$b_{ij}^\Delta$', c='C1', linestyle=':', linewidth=lw, marker='v')
    #
    # # order = [2, 0, 1]
    # # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(prop={'size': 12}) # ,ncol=4, loc='center', bbox_to_anchor=(0.5, 1.1), prop={'size': 9}
    #
    # plt.savefig(f'../logs_completed/aa_plots/ntokens_convergence.eps', format='eps', bbox_inches='tight')
    #
    # # errtarget = df_kdef_CD[df_kdef_CD['model_nr'] == 282]['CBFS_nmse'].values[0]
    # #
    # df_kdef_CD[df_kdef_CD['CBFS_nmse'] < errtarget]

    ################################################ do not delete until here






    print('end')
    print('end')
