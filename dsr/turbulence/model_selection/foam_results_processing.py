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

def read_case_results(case_dir):
    dirlist_case = os.listdir(case_dir)

    # get last written solution:
    sol_dirs = [int(sol) for sol in dirlist_case if sol.isnumeric()]
    last_sol_dir = f'{max(sol_dirs)}'

    if last_sol_dir == '0':
        return 'Diverged', last_sol_dir

    sol_files = [f for f in os.listdir(os.path.join(case_dir,last_sol_dir)) if os.path.isfile(os.path.join(case_dir,last_sol_dir, f))]

    vectors = ['U', 'U_LES']
    scalars = ['k', 'k_LES', 'omega', 'nut', 'p', 'phi']

    results = {}
    for file in sol_files:
        if file in scalars:
            results[file] = fluidfoam.readscalar(case_dir, last_sol_dir, file)
        if file in vectors:
            results[file] = fluidfoam.readvector(case_dir, last_sol_dir, file)

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
    else:
        model_type = model_info['name'].split('_')[0]
        name = f'{model_info["training_case"]}_{model_type}_{model_info["model_nr"]}'


    return name, model_info

def process_OF_results(selected_model_file=False):

    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/'
    dirlist = os.listdir(base_dir)
    dirlist.pop(dirlist.index('common'))
    dirlist.pop(dirlist.index('base_dir_no_dimension_check'))
    dirlist.pop(dirlist.index('base_dir'))

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
    save_lists = {
        'model_order': [],
        'PH_mse': [],
        'PH_iter': [],
        'CD_mse': [],
        'CD_iter': [],
        'CBFS_mse': [],
        'CBFS_iter': []
    }

    # load selected models file:
    if selected_model_file:
        df_models = pd.read_csv(selected_model_file)

    results = {}
    for dir in dirlist:

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

def add_scatter(x, df, plot_col, color, markersize, lw, label_once):

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
        plt.scatter(x[ii], df[plot_col].values[ii], c=face_col, edgecolors=color, s=markersize, linewidth=lw, label=label)


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

    best_sparta = {'PH': {'CD': 0.246319164597, 'PH': 0.16591760527490615, 'CBFS': 0.46520543797084507},
                   'CD': {'CD': 0.246319164597, 'PH': 0.16591760527490615, 'CBFS': 0.46520543797084507},
                   'CBFS': {'CD': 0.2081585409088, 'PH': 0.20329225923, 'CBFS': 0.499579335406}
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


    markersize = 30
    lw = 1.5
    figsize = (24, 6)
    cm = 1 / 2.54  # centimeters in inches

    # plot CFD errors:
    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    add_scatter(x, df_sorted, 'PH_nmse', 'C0', markersize, lw, r'$PH_{10595}$')
    add_scatter(x, df_sorted, 'CD_nmse', 'C1', markersize, lw, r'$CD_{12600}$')
    add_scatter(x, df_sorted, 'CBFS_nmse', 'C2', markersize, lw, r'$CBFS_{13700}$')

    plt.ylabel(r'$\varepsilon (U) / \varepsilon(U_0)$')
    plt.xlabel('Models')
    plt.ylim([0,1])
    plt.xticks(np.arange(0,100,10))
    ax = plt.gca()
    ax.xaxis.grid(linestyle=':')
    plt.scatter(10, 10, c='none', edgecolors='grey', s=markersize, linewidth=lw,
                label='Incorrect dimensionality')
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
        'CD': 0.4489642308683687,
        'PH': 0.5462239021080454,
        'CBFS': 0.5369412002533871
    }}

    # plot inv_NRMSE errors:
    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    add_scatter(x, df_sorted, 'r_max_PH', 'C0', markersize, lw, r'$PH_{10595}$')
    add_scatter(x, df_sorted, 'r_max_CD', 'C1', markersize, lw, r'$CD_{12600}$')
    add_scatter(x, df_sorted, 'r_max_CBFS', 'C2', markersize, lw, r'$CBFS_{13700}$')

    plt.ylabel(r'$r_{max}$')
    plt.xlabel('Models')
    plt.ylim([0.4,0.9])
    plt.xticks(np.arange(0,100,10))
    ax = plt.gca()
    ax.xaxis.grid(linestyle=':')
    plt.scatter(10, 10, c='none', edgecolors='grey', s=markersize, linewidth=lw,
                label='Incorrect dimensionality')
    plt.axhline(y=best_sparta[model_type]['PH'], color='C0', linestyle=(0, (5, 1)), label=r'SpaRTA $PH_{10595}$', linewidth = lw) # densely dashed
    plt.axhline(y=best_sparta[model_type]['CD'], color='C1', linestyle=(0, (1, 1)), label=r'SpaRTA $CD_{12600}$', linewidth = lw)
    plt.axhline(y=best_sparta[model_type]['CBFS'], color='C2', linestyle=(0, (3, 1, 1, 1, 1, 1)), label=r'SpaRTA $CBFS_{13700}$', linewidth = lw)
    plt.legend(prop={'size': 8}, loc='center right', bbox_to_anchor=(1.3, 0.5))
    # plt.legend(prop={'size': 8}, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=7)
    plt.savefig(f'../logs_completed/aa_plots/{training_case}_{model_type}_r_max.eps', format='eps', bbox_inches='tight')

def plot_selection(plot_list):
    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/'
    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/CBFS'
    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/PH'

    # best_sparta = {'PH': }


    results, hifi_data = process_OF_results(base_dir)


    for plot_case in plot_list:
        case = plot_case[1]
        model = plot_case[0]

        mesh_x_flat, mesh_y_flat = hifi_data[case]['field'][:, 0],  hifi_data[case]['field'][:, 1]
        mesh_x = reshape_to_mesh(mesh_x_flat)
        # mesh_y = reshape_to_mesh(mesh_y_flat)
        n_points = mesh_x.shape[0]

        # # # check if number of lines in interpolated data matches the openFoam pp folder
        # ppkeys = list(results[list(results.keys())[0]]['pp'].keys())
        # ppkeys.pop(ppkeys.index('residuals'))
        # if len(ppkeys) != n_lines:
        #     raise ValueError('The interpolated high fidelity data contains a different number of lines than the OF results')

        plt.figure()
        plt.plot(mesh_x_flat[:n_points], mesh_y_flat[:n_points], c='Black')
        plt.plot(mesh_x_flat[-n_points:], mesh_y_flat[-n_points:], c='Black')
        plt.plot([mesh_x_flat[0], mesh_x_flat[-n_points]], [mesh_y_flat[0], mesh_y_flat[-n_points]], c='Black')
        plt.plot([mesh_x_flat[n_points - 1], mesh_x_flat[-1]], [mesh_y_flat[n_points - 1], mesh_y_flat[-1]], c='Black')

        ax = plt.gca()
        ax.set_aspect('equal')
        # add LES results:
        u_scale = 1

        for x in np.unique(hifi_data[case]['lines'][:, 0].round()):
            plot_bool = (lines['mesh_x'] > x - 0.1) & (lines['mesh_x'] < x + 0.1)
            plt.plot(x + lines['u'][plot_bool],
                     lines['mesh_y'][plot_bool], c='Black', marker='o', markevery=5, label=label)
            if label:
                label = None

        label = r'$k-\omega$ SST'
        # add baseline kOmegaSST results:
        for line in results['kOmegaSST']['pp']:
            if 'single' in line:
                data = results['kOmegaSST']['pp'][line]['line_U']
                plt.plot(data[:, 0] + u_scale*data[:, 3], data[:, 1], c='C0', linestyle=':', markevery=5, label=label)
                if label:
                    label = None

        # add spaRTA results:
        label = 'SpaRTA'
        for line in results[best_sparta]['pp']:
            if 'single' in line:
                data = results[best_sparta]['pp'][line]['line_U']
                plt.plot(data[:, 0] + u_scale*data[:, 3], data[:, 1], c='C1', linestyle='--', markevery=5, label=label)
                if label:
                    label = None

        # add DSR results:
        label = 'DSR'
        for line in results[best_dsr]['pp']:
            if 'single' in line:
                data = results[best_dsr]['pp'][line]['line_U']
                plt.plot(data[:, 0] + u_scale*data[:, 3], data[:, 1], c='C2', linestyle='-', markevery=5, label=label)
                if label:
                    label = None

        plt.legend()
    #
    # exclude = 6
    # plt.scatter(mesh_x_flat[:exclude*n_points], mesh_y_flat[:exclude*n_points], c='Black')
    # plt.scatter(mesh_x_flat[-exclude*n_points:], mesh_y_flat[-exclude*n_points:], c='Black')


if __name__ == '__main__':

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    # read_and_plot_PH()
    matplotlib.use('tkagg')
    #
    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/CBFS'
    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/PH'


    plot_selection([])




    ####################### lines below used to add CFD results to selected_models file
    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/all_PH/kDef_PH_selected_models.csv'
    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/all_CD/kDef_CD_selected_models.csv'
    # selected_model_file = '/home/jasper/Documents/afstuderen/python/dsr_turbulence/logs_completed/all_CBFS/kDef_CBFS_selected_models.csv'
    # process_OF_results(selected_model_file)


    #################### lines below used to make scatter plots of error in CFD and training rewards.
    # selected_model_file = '../logs_completed/kDef_PH/kDef_PH_selected_models_CFD_results.csv'
    # results_scatter(selected_model_file)
    #
    # selected_model_file = '../logs_completed/kDef_CD/kDef_CD_selected_models_CFD_results.csv'
    # results_scatter(selected_model_file)
    #
    # selected_model_file = '../logs_completed/kDef_CBFS/kDef_CBFS_selected_models_CFD_results.csv'
    # results_scatter(selected_model_file)


    print('end')
    print('end')
