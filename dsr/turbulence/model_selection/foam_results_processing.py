import os
import sys

import matplotlib
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

def process_OF_results(base_dir):
    # change the working directory to dir with the openFOAM results

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

    results = {}
    for dir in dirlist:

        name, model_info = find_model_info(os.path.join(base_dir, dir))

        results[name] = {'model_info': model_info}

        for case in cases:

            case_result, final_iteration = read_case_results(os.path.join(base_dir, dir, case))
            if case_result == 'Diverged':
                results[name][case] = {'norm_mse': case_result,
                                       'final_iteration': final_iteration}
            else:
                results[name][case] = {'norm_mse': mse(hifi_data[case]['U'],
                                                   case_result['U'][:2, hifi_data[case]['keep']]),
                                       'final_iteration': final_iteration}
                results[name][case]['norm_mse'] = results[name][case]['norm_mse']/hifi_data[case]['mse_kOmegaSST']

    # best_cases = {}
    # for case in cases:
    #     best_cases[case] = {
    #         'dsr': {
    #             'name': [],
    #             'norm_mse': []
    #         },
    #         'sparta': {
    #             'name': [],
    #             'norm_mse': []
    #         }
    #     }
    #
    # for model in results:
    #     results[model]['norm_mse'] = {}
    #     for case in cases:
    #         if results[model][case] == 'Diverged':
    #             pass
    #         else:
    #             U_case = results[model][case]['U']
    #             if U_case.shape == (3, 1):
    #                 results[model]['norm_mse'][case] = 'Diverged'
    #             else:
    #                 results[model]['norm_mse'][case] = mse(hifi_data[case]['U'],
    #                                                        U_case[:2, hifi_data[case]['keep']])
    #
    #                 results[model]['norm_mse'][case] = results[model]['norm_mse'][case]/hifi_data[case]['mse_kOmegaSST']
    #                 if 'sparta' in model:
    #                     model_type = 'sparta'
    #                 else:
    #                     model_type = 'dsr'
    #
    #                 best_cases[case][model_type]['norm_mse'].append(results[model]['norm_mse'][case])
    #                 best_cases[case][model_type]['name'].append(model)
    #
    # results['best'] = best_cases

    return results, hifi_data

def results_scatter(base_dir):
    results, hifi_data = process_OF_results(base_dir)

    cases = ['CD', 'PH', 'CBFS']

    # scatter all points based on what training case, separated by whether the dimensions are correct and number of tokens

    print('end')


def plot_selection(plot_list):
    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/'
    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/CBFS'
    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/PH'
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

    # read_and_plot_PH()
    matplotlib.use('tkagg')
    #
    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/'
    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/CBFS'
    # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/PH'
    # process_OF_results(base_dir)

    results_scatter(base_dir)

    # plot_selection([['PH_kDef_7', 'CBFS'],
    #                 ['PH_kDef_20', 'PH']])


    #
    # interpolate_PH()
    # interpolate_CBFS()
    # interpolate_CD()

    print('end')
    print('end')
