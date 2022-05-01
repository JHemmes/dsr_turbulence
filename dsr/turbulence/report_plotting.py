import os
import platform
import numpy as np
import json
import pandas as pd

from dsr.turbulence.resultprocessing import fetch_iteration_metrics, compare_dicts, report_plot, load_iterations

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

def plot_turbulent_velocity_fluctuations():
    # sample plot of turbulent fluctuations
    np.random.seed(0)

    tsmall = np.linspace(0, 0.5, 200)
    tlarge = np.linspace(0, 0.5, 25)

    mean = 2.7

    noisesmall = np.random.normal(loc=0, scale=0.05, size=tsmall.shape)
    noiselarge = np.random.normal(loc=mean, scale=0.1, size=tlarge.shape)

    # interpolate large to small scale, add small scale noise
    final_data = np.interp(tsmall, tlarge, noiselarge) + noisesmall

    # add mean line
    mean_line = np.mean(final_data) * np.ones(shape=tsmall.shape)

    cm = 1 / 2.54  # centimeters in inches
    plt.figure(figsize=(12 * cm, 9 * cm))
    plt.plot(tsmall, final_data, linewidth=1)
    plt.plot(tsmall, mean_line, '--')
    plt.xlabel(r"$t \; [s]$")
    plt.ylabel(r"$u_x \; \left[\frac{m}{s}\right]$")
    plt.savefig('../turbulent_fluctuations.eps', format='eps', bbox_inches='tight')

def plot_ml_polynomial():
    # for the conversion of figsize inches to cm
    cm = 1 / 2.54

    np.random.seed(4)
    n_points = 70
    lower = -2
    upper = 2
    X = np.random.uniform(lower, upper, n_points)
    y = X ** 3
    y += np.random.normal(0, 0.7, n_points)

    X_true = np.linspace(lower, upper, 100)
    y_true = X_true ** 3

    X_true = np.linspace(lower, upper, 100)
    y_init = X_true ** 3 + X_true ** 2 + X_true + 1

    X_pred = 1.25
    y_pred = X_pred ** 3

    cm = 1 / 2.54

    plt.figure(figsize=(12 * cm, 9 * cm))
    plt.scatter(X, y, label=r"Data point $(x_m, y_m)$")
    plt.plot(X_true, y_init, '--C2', linewidth=1, label=R"Initial model", zorder=1)
    plt.plot(X_true, y_true, 'C1', linewidth=1, label=R"Fitted model", zorder=1)
    plt.scatter(x=X_pred, y=y_pred, color='C3', label=r"Prediction", zorder=1)
    plt.xlabel(r"$X$")
    plt.ylabel(r"$y$")
    plt.legend()
    plt.savefig('../MLexample.eps', format='eps', bbox_inches='tight')



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


def contourplot_vars(case):

    # find better kDeficit limits:
    if 'PH' in case:
        skip_wall = 15
    elif 'CD' in case:
        skip_wall = 2
    elif 'CBFS' in case:
        skip_wall = 6

    import pickle

    frozen = pickle.load(open(f'turbulence/frozen_data/{case}_frozen_var.p', 'rb'))
    data_i = frozen['data_i']

    mesh_x = data_i['meshRANS'][0, :, :]
    mesh_y = data_i['meshRANS'][1, :, :]
    mesh_x_flat = mesh_x.flatten(order='F').T
    mesh_y_flat = mesh_y.flatten(order='F').T

    k = np.reshape(data_i['k'], mesh_x.shape, order='F')
    omega_frozen = np.reshape(data_i['omega_frozen'], mesh_x.shape, order='F')
    nut_frozen = np.reshape(data_i['nut_frozen'], mesh_x.shape, order='F')
    kDeficit = np.reshape(data_i['kDeficit'], mesh_x.shape, order='F')
    bDelta = data_i['bDelta']


    vmin = np.min(k[:, skip_wall:-skip_wall])
    vmax = np.max(k[:, skip_wall:-skip_wall])
    levels = np.linspace(vmin, vmax, 50)
    plt.figure(figsize=(20,10))
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, k, vmin=vmin, vmax=vmax, levels=levels, cmap='Reds')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.colorbar(fraction=0.046, pad=0.04, ax=ax, shrink=0.5)
    plt.savefig(f'../logs_completed/contourplots/{case}_k.png', bbox_inches='tight')

    vmin = np.min(omega_frozen[:, skip_wall:-skip_wall])
    vmax = np.max(omega_frozen[:, skip_wall:-skip_wall])
    levels = np.linspace(vmin, vmax, 50)

    plt.figure(figsize=(20,10))
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, omega_frozen, vmin=vmin, vmax=vmax, levels=levels, cmap='Reds')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.colorbar(fraction=0.046, pad=0.04, ax=ax, shrink=0.5)
    plt.savefig(f'../logs_completed/contourplots/{case}_omega_frozen.png', bbox_inches='tight')


    vmin = np.min(nut_frozen[:,skip_wall:-skip_wall])
    vmax = np.max(nut_frozen[:,skip_wall:-skip_wall])
    #
    levels = np.linspace(vmin, vmax, 30)

    plt.figure(figsize=(20,10))
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, nut_frozen, vmin=vmin, vmax=vmax, levels=levels, cmap='Reds')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.colorbar(fraction=0.046, pad=0.04, ax=ax, shrink=0.5)
    plt.savefig(f'../logs_completed/contourplots/{case}_nut.png', bbox_inches='tight')


    vmin = np.min(kDeficit[:,skip_wall:-skip_wall])
    vmax = np.max(kDeficit[:,skip_wall:-skip_wall])
    #
    levels = np.linspace(vmin, vmax, 30)

    plt.figure(figsize=(20,10))
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, kDeficit, vmin=vmin, vmax=vmax, levels=levels, cmap='Reds')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.colorbar(fraction=0.046, pad=0.04, ax=ax, shrink=0.5)
    plt.savefig(f'../logs_completed/contourplots/{case}_kDeficit.png', bbox_inches='tight')

    components = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    for component in components:
        bDelta_component = np.reshape(bDelta[component[0], component[1], :], mesh_x.shape, order='F')

        vmin = np.min(bDelta_component[:, skip_wall:-skip_wall])
        vmax = np.max(bDelta_component[:, skip_wall:-skip_wall])
        #
        levels = np.linspace(vmin, vmax, 30)

        plt.figure(figsize=(20,10))
        plt.tight_layout()
        # plt.contourf(mesh_x, mesh_y, bDelta_component, levels=30, cmap='Reds')
        plt.contourf(mesh_x, mesh_y, bDelta_component, vmin=vmin, vmax=vmax, levels=levels, cmap='Reds')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.colorbar(fraction=0.046, pad=0.04, ax=ax, shrink=0.5)
        plt.savefig(f'../logs_completed/contourplots/{case}_bDelta_{component}.png', bbox_inches='tight')

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
        linestyles = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), '-.', (0, (3, 1, 1, 1, 1, 1)), ':']
        stylecounter = 0
        y = []
        labels = []
        colors = [f'C{ii+1}' for ii in range(len(results[param]['par_val']))]
        sorted_colors = []
        sorted_linestyles = []
        for ii in np.argsort(results[param]['par_val']):
            y.append(results[param]['arrays'][ii])
            sorted_linestyles.append(linestyles[stylecounter])
            sorted_colors.append(colors[stylecounter])
            stylecounter += 1
            labels.append(base_labels_params[param](results[param]['par_val'][ii]))
            if results[param]['par_val'][ii] == bsl_params[param]:
                labels[-1] = labels[-1] + ' (BSL)'
                sorted_colors[-1] = 'C0'
                sorted_linestyles[-1] = '-'
                stylecounter -= 1

        filename = os.path.join(logdir, 'report_plots', f'{logdir.split("_")[-1]}_{param}.eps')

        ########################## below this only bDelta combined plot
        param = 'num_units' # (if the first param is learning_rate other one)
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
        linestyles = [(0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), '-.', (0, (3, 1, 1, 1, 1, 1)), ':']
        for ii in np.argsort(results[param]['par_val']):
            y.append(results[param]['arrays'][ii])
            sorted_linestyles.append(linestyles[stylecounter])
            sorted_colors.append(colors[stylecounter])
            stylecounter += 1
            labels.append(base_labels_params[param](results[param]['par_val'][ii]))
            if results[param]['par_val'][ii] == bsl_params[param]:
                labels[-1] = labels[-1] + ' (BSL)'
                sorted_colors[-1] = 'C0'
                sorted_linestyles[-1] = '-'
                stylecounter -= 1
        #
        # sorted_colors[-1] = 'C3'
        #
        filename = os.path.join(logdir, 'report_plots', 'bDeltaCombined.eps')
        ######################### above only for bDelta combined plot

        figsize = (12, 9)
        xlabel = 'Iterations'
        ylabel = r'$r_{max}(\tau)$'

        report_plot(x, y, labels, sorted_colors, xlabel, ylabel, filename, figsize,
                    linewidths=False, linestyles=sorted_linestyles)

    print('here')

def plot_optimise_statistics():
    logdir = '../logs_completed/log_2022-03-20-182735_optimize_statistics'

    matplotlib.use('tkagg')

    optim_stats = {}
    for filename in os.listdir(logdir):
        split = filename.split('_')[-1].split('.')
        if split[0] == 'stats' and split[1] == 'csv':
            df_append = pd.read_csv(f'{logdir}/{filename}', header=None)
            optim_stats[filename] = df_append

    # for key in return_dict:

    all_match_bestperformer = []
    first_bestperformer = []
    last_bestperformer = []
    for key in optim_stats:
        for value in optim_stats[key].iloc[:,0].values:
            all_match_bestperformer.append(value)

        for value in optim_stats[key].iloc[:,0].values[:300]:
            first_bestperformer.append(value)

        for value in optim_stats[key].iloc[:,0].values[300:]:
            last_bestperformer.append(value)

    all_match_bestperformer = np.array(all_match_bestperformer)
    print(f"ratio of batches that found best perfromer with less than 100 iters  {np.mean(all_match_bestperformer < 100)}")


    bins = np.arange(0, 2000, 10)
    #
    plt.figure()
    plt.hist(x=all_match_bestperformer, bins=bins) #, density=True)
    plt.xlim([0,200])
    plt.show()
    #

    #
    plt.figure()
    plt.title('first300')
    plt.hist(x=first_bestperformer, bins=bins, density=True)
    plt.xlim([0,200])
    plt.show()
    #

    #
    plt.figure()
    plt.title('last300')

    plt.hist(x=last_bestperformer, bins=bins, density=True)
    plt.xlim([0,200])
    plt.show()
    #
    #

    batches_match = np.zeros((optim_stats[list(optim_stats.keys())[0]].shape[0],
                              optim_stats[list(optim_stats.keys())[0]].shape[1] - 1, len(optim_stats)))

    # all_match_bestperformer = np.array(all_match_bestperformer)
    # np.mean(all_match_bestperformer < 100)

    for ii in range(len(optim_stats)):
        key = list(optim_stats.keys())[ii]
        batches_match[:, :, ii] = optim_stats[key].values[:, 1:]

    mean_batch_match = np.mean(batches_match, axis=-1)


    figsize = (12, 9)
    cm = 1 / 2.54  # centimeters in inches
    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    plot_iters = [0, 100, 300, 500]
    colors = [f'C{ii+1}' for ii in range(len(plot_iters))]
    colors[-1] = 'C6' # the purple is difficult to see over the blue hist
    linestyles = ['-',  (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), '-.', (0, (3, 1, 1, 1, 1, 1)) ]

    counter = 0
    for ii in plot_iters:
        plt.plot(bins, 100*mean_batch_match[ii, :], label=f'Iter: {ii}', color=colors[counter], linestyle=linestyles[counter], linewidth=2)
        counter += 1
    plt.xlabel('Iteration limit')
    plt.xlim([0,200])
    plt.xticks(np.arange(0,220, 20))
    plt.grid('both', linestyle=':')
    plt.legend()

    ax1 = plt.gca()
    ax1.set_ylabel('Match percentage')
    ax2 = ax1.twinx()
    ax1.set_zorder(10)
    ax1.patch.set_visible(False)
    ax2.hist(x=all_match_bestperformer, bins=bins, density=True, zorder=-1, label='All')
    # opacity doesnt work with eps...
    # ax2.hist(x=last_bestperformer, bins=bins, density=True, zorder=-1, alpha = 0.5,color ='red', label='Last 300') # not sure if this should be included
    ax2.set_ylabel('Probability density')
    # plt.legend(loc='right')

    plt.savefig('../logs_completed/aa_plots/iterlim_prob_dens_batch_match.eps', format='eps', bbox_inches='tight')


    plt.show()

    # make plot of duration and max and mean rewards

    logdir = '../logs_completed/compare_iterlim_optimisation/log_2022-01-19-154202_LR01'
    lim_stats = load_iterations(logdir)

    logdir = '../logs_completed/compare_iterlim_optimisation/log_2022-03-22-105724_unconstrained_optimisation'
    unlim_stats = load_iterations(logdir)

    lim_base_r_arr = []
    lim_duration_arr = []
    for run in lim_stats:
        lim_base_r_arr.append(lim_stats[run]['base_r_best'].values)
        lim_duration_arr.append(lim_stats[run]['proc_time'].values)

    lim_base_r_arr = np.array(lim_base_r_arr)
    lim_duration_arr = np.array(lim_duration_arr)

    unlim_base_r_arr = []
    unlim_duration_arr = []
    for run in unlim_stats:
        unlim_base_r_arr.append(unlim_stats[run]['base_r_best'].values)
        unlim_duration_arr.append(unlim_stats[run]['proc_time'].values)

    unlim_base_r_arr = np.array(unlim_base_r_arr)
    unlim_duration_arr = np.array(unlim_duration_arr)

    figsize = (12, 9)
    cm = 1 / 2.54  # centimeters in inches
    plt.figure(figsize=tuple([val*cm for val in list(figsize)]))
    plt.plot(np.max(lim_base_r_arr, axis=0), label=r'$r_{max}$ Constrained', linewidth = 2, linestyle= '-')
    plt.plot(np.max(unlim_base_r_arr, axis=0), label=r'$r_{max}$ Unconstrained', linewidth = 2, linestyle= (0, (5, 1)))
    plt.plot(np.mean(lim_base_r_arr, axis=0), label=r'$r_{mean}$ Constrained', linewidth = 2, linestyle= (0, (1, 1)))
    plt.plot(np.mean(unlim_base_r_arr, axis=0), label=r'$r_{mean}$ Unconstrained', linewidth = 2, linestyle= (0, (3, 1, 1, 1)))
    plt.xlim([0,600])
    plt.legend(loc='lower right')
    plt.grid('both', linestyle = ':')
    plt.xlabel('Iterations')
    plt.ylabel( r'$r\;(\tau)$')
    plt.savefig('../logs_completed/aa_plots/iterlim_rewards.eps', format='eps', bbox_inches='tight')

    clipped_lim_duration_arr = lim_duration_arr[:, :600]
    np.mean(np.sum(clipped_lim_duration_arr, axis=1))
    np.mean(np.sum(unlim_duration_arr, axis=1))

    print(np.mean(np.sum(clipped_lim_duration_arr, axis=1)) / np.mean(np.sum(unlim_duration_arr, axis=1)))

    print('here')


if __name__ == "__main__":

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files


    # create_plots_for_increasing_n_iterations()
    #
    # logdir = '../logs_completed/sensitivity_analysis_kDeficit'
    # plot_pretty_sensitivity_results(logdir, ['entropy_weight', 'learning_rate', 'initializer', 'num_layers' , 'num_units'])

    logdir = '../logs_completed/sensitivity_analysis_bDelta'
    plot_pretty_sensitivity_results(logdir, ['learning_rate', 'num_units'])


    # plot_pretty_sensitivity_results(logdir, ['initializer'])

    # plot_optimise_statistics()



    print('end')
    print('end')