import os
import platform
import numpy as np
import json

from dsr.turbulence.resultprocessing import fetch_iteration_metrics, compare_dicts, report_plot

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

def create_plots_for_increasing_n_iterations():
    # WIP
    logdir = '../logs_completed/compare_baselines'

    dirlist = os.listdir(logdir)

    iterations = np.arange(10, 110, 10)

    metrics = {}
    all_rewards = []
    for dir in dirlist:
        results = load_iterations(os.path.join(logdir, dir))

        basename = '_'.join(list(results.keys())[0].split('.')[0].split('_')[:-1])

        metrics[dir] = {}
        metrics[dir]['mean'] = []
        metrics[dir]['std'] = []
        metrics[dir]['max'] = []

        rewards_sorted = []
        for ii in range(len(results)):
            rewards_sorted.append(results[f'{basename}_{ii+1}.csv']['base_r_best'].values[-1])
            all_rewards.append(results[f'{basename}_{ii+1}.csv']['base_r_best'].values[-1])

        for nit in iterations:
            metrics[dir]['mean'].append(np.mean(rewards_sorted[:nit]))
            metrics[dir]['std'].append(np.std(rewards_sorted[:nit]))
            metrics[dir]['max'].append(np.max(rewards_sorted[:nit]))

    from scipy.stats import norm
    mu, std = norm.fit(all_rewards)


    plt.figure()
    plt.hist(all_rewards, bins=20)

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    plt.plot(x, p, 'k', linewidth=2)

    plt.axvline(x=np.mean(all_rewards), color='C1')
    plt.axvline(x=np.mean(all_rewards) + np.std(all_rewards), color='C2')
    plt.axvline(x=np.mean(all_rewards) + 2* np.std(all_rewards), color='C3')
    plt.axvline(x=np.mean(all_rewards) - np.std(all_rewards), color='C2')
    plt.axvline(x=np.mean(all_rewards) - 2* np.std(all_rewards), color='C3')
    # plt.axvline(x=np.mean(all_rewards))

    plt.show()


    plt.figure()
    for key in metrics:
        plt.plot(iterations, metrics[key]['mean'], label=key)

    plt.legend()

    plt.figure()
    for key in metrics:
        plt.plot(iterations, metrics[key]['std'], label=key)

    plt.legend()

    plt.figure()
    for key in metrics:
        plt.plot(iterations, metrics[key]['max'], label=key)

    plt.legend()

    plt.show()






def contourplot_vars():

    import pickle

    frozen = pickle.load(open(f'turbulence/frozen_data/PH10595_frozen_var.p', 'rb'))
    data_i = frozen['data_i']

    mesh_x = data_i['meshRANS'][0, :, :]
    mesh_y = data_i['meshRANS'][1, :, :]
    mesh_x_flat = mesh_x.flatten(order='F').T
    mesh_y_flat = mesh_y.flatten(order='F').T

    k = np.reshape(data_i['k'], mesh_x.shape, order='F')
    omega_frozen = np.reshape(data_i['omega_frozen'], mesh_x.shape, order='F')
    nut_frozen = np.reshape(data_i['nut_frozen'], mesh_x.shape, order='F')
    kDeficit = np.reshape(data_i['kDeficit'], mesh_x.shape, order='F')


    plt.figure()
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, k, levels=30, cmap='Reds')
    # plt.contourf(mesh_x, mesh_y, , levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    plt.colorbar()

    plt.figure()
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, omega_frozen, vmin=np.min(omega_frozen), vmax=100, levels=3000, cmap='Reds')
    # plt.contourf(mesh_x, mesh_y, , levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    plt.colorbar()

    plt.figure()
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, nut_frozen, levels=30, cmap='Reds')
    # plt.contourf(mesh_x, mesh_y, , levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    plt.colorbar()

    plt.figure()
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, kDeficit, levels=30, cmap='Reds')
    # plt.contourf(mesh_x, mesh_y, , levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    plt.colorbar()

def plot_pretty_sensitivity_results(logdir, parameters):
    # first scan dirs, save required dirs
    # depending on plots required, fetch required iterations and plot accordingly

    dirlist = os.listdir(logdir)
    dirlist.remove('config_baseline.json')
    try:
        dirlist.remove('results')
    except ValueError:
        pass

    try:
        dirlist.remove('report_plots')
    except ValueError:
        pass

    try:
        os.mkdir(os.path.join(logdir, 'report_plots'))
    except FileExistsError:
        pass


    with open(os.path.join(logdir, 'config_baseline.json'), encoding='utf-8') as f:
        config_bsl = json.load(f)

    bsl_params = {
        'entropy_weight': config_bsl['controller']['entropy_weight'],
        'learning_rate': config_bsl['controller']['learning_rate'],
        'initializer': config_bsl['controller']['initializer'],
        'num_layers': config_bsl['controller']['num_layers'],
        'num_units': config_bsl['controller']['num_units']
    }

    base_labels_params = {
        'entropy_weight': (lambda x: f'$\lambda_H = {x}$'),
        'learning_rate': (lambda x: f'$\\alpha = {x}$'),
        'initializer': (lambda x: f'Initialiser: {x}'),
        'num_layers': (lambda x: f'$n_{{layers}} = {x}$'),
        'num_units': (lambda x: f'$n_{{units}} = {x}$')
    }

    results = {param: {} for param in parameters}

    for run in dirlist:
        run_dir = os.path.join(logdir, run)
        with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
            config_run = json.load(f)

        diff = compare_dicts(config_bsl, config_run)

        if diff[0] == 'baseline':
            results['baseline'] = {'data': fetch_iteration_metrics(run_dir)}
            print(run_dir)
        else:
            if len(diff) == 1:
                if diff[0][0] in parameters:
                    results[diff[0][0]][run_dir] = {'value': diff[0][1]}
                if diff[0][0] == 'initializer':
                    # also figure out the standard deviation
                    results[diff[0][0]][run_dir] = {'value': f"${config_run['task']['name'].split('_')[-1]}$"}



    for param in parameters:
        results[param]['par_val'] = [bsl_params[param]]

        # write def that return the max given the data/load_iterations result
        tmp_arr = np.array(results['baseline']['data']['base_r_best'])
        results[param]['arrays'] = [np.max(tmp_arr, axis=0)]
        for run in results[param]:
            if 'log' in run:
                results[param]['par_val'].append(results[param][run]['value'])
                data = fetch_iteration_metrics(run)

                tmp_arr = np.array(data['base_r_best'])
                results[param]['arrays'].append(np.max(tmp_arr, axis=0))
                del tmp_arr

        # create requirements for plot:
        # legend needs to be sorted, baseline always is C0.
        x = np.arange(results[param]['arrays'][0].shape[0])
        y = []
        labels = []
        colors = [f'C{ii}' for ii in range(len(results[param]['par_val']))]
        sorted_colors = []
        for ii in np.argsort(results[param]['par_val']):
            y.append(results[param]['arrays'][ii])
            sorted_colors.append(colors[ii])
            labels.append(base_labels_params[param](results[param]['par_val'][ii]))
            if results[param]['par_val'][ii] == bsl_params[param]:
                labels[-1] = labels[-1] + ' (BSL)'

        filename = os.path.join(logdir, 'report_plots', f'{logdir.split("_")[-1]}_{param}.eps')

        ########################## below this only bDelta double plot
        # param = 'num_units' (if the first param is the other one)
        # results[param]['par_val'] = [bsl_params[param]]
        #
        # # write def that return the max given the data/load_iterations result
        # tmp_arr = np.array(results['baseline']['data']['base_r_best'])
        # results[param]['arrays'] = [np.max(tmp_arr, axis=0)]
        # for run in results[param]:
        #     if 'log' in run:
        #         results[param]['par_val'].append(results[param][run]['value'])
        #         data = fetch_iteration_metrics(run)
        #
        #         tmp_arr = np.array(data['base_r_best'])
        #         results[param]['arrays'].append(np.max(tmp_arr, axis=0))
        #         del tmp_arr
        #
        # # create requirements for plot:
        # # legend needs to be sorted, baseline always is C0.
        # x = np.arange(results[param]['arrays'][0].shape[0])
        # # y = []
        # # labels = []
        # colors = [f'C{ii}' for ii in range(len(results[param]['par_val']))]
        # # sorted_colors = []
        # for ii in np.argsort(results[param]['par_val']):
        #     y.append(results[param]['arrays'][ii])
        #     sorted_colors.append(colors[ii])
        #     labels.append(base_labels_params[param](results[param]['par_val'][ii]))
        #     if results[param]['par_val'][ii] == bsl_params[param]:
        #         labels[-1] = labels[-1] + ' (BSL)'
        #
        # sorted_colors[-1] = 'C3'
        #
        # filename = os.path.join(logdir, 'report_plots', 'bDeltaCombined.eps')
        ######################### above only for bDelta double plot

        figsize = (12, 9)
        xlabel = 'Iterations'
        ylabel = r'$r_{max}(\tau)$'

        report_plot(x, y, labels, sorted_colors, xlabel, ylabel, filename, figsize)

    # make the combined bDelta plot:





    print('here')


if __name__ == "__main__":

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files


    # create_plots_for_increasing_n_iterations()
    # logdir = '../logs_completed/compare_iterlim_optimisation'
    # logdir = '../logs_completed/sensitivity_analysis_kDeficit'
    logdir = '../logs_completed/sensitivity_analysis_bDelta'

    plot_pretty_sensitivity_results(logdir, ['learning_rate', 'num_units'])
    # plot_pretty_sensitivity_results(logdir, ['initializer'])

    print('end')
    print('end')