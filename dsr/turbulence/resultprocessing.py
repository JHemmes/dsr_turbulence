# Jasper Hemmes - 2021

import os
import sys
import shutil
import pickle
import numpy as np
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import time
from dsr.turbulence.dataprocessing import load_frozen_RANS_dataset, de_flatten_tensor
from dsr.program import from_str_tokens
from dsr.turbulence.lumley_plots import plot_lumley_comparison
import copy


def plot_results(results, config):
    inputs = config['task']['dataset_info']['input']
    for input in inputs:
        if input in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']:
            tensor = True
            break
        else:
            tensor = False

    if tensor:
        scatter_results_tensor(results, config)
        # contourplot_results_tensor(results, config)
        lumley_plot(results, config)
    else:
        scatter_results_scalar(results, config)
        contourplot_results_scalar(results, config)

def scatter_results_scalar(results, config):

    plot_sparta = True

    X, y = config['task']['dataset']

    logdir = config['training']['logdir']

    if plot_sparta:
        inputs = config['task']['dataset_info']['input']

        # find grad_u_T1
        if 'grad_u_T1' in inputs:
            grad_u_T1 = X[:,inputs.index('grad_u_T1')]
        else:
            dummy_config = config['task']
            dummy_config['dataset']['input'] = ['grad_u_T1']
            grad_u_T1, _ = load_frozen_RANS_dataset(dummy_config)

        # find K
        if 'k' in inputs:
            k = X[:, inputs.index('k')]
        else:
            dummy_config = config['task']
            dummy_config['dataset']['input'] = ['k']
            k, _ = load_frozen_RANS_dataset(dummy_config)
        Rsparta = 2*k*grad_u_T1*1.4

        yhat, _ = results['program'].execute(X)
        NRMSE = np.sqrt(np.mean((y-yhat)**2))/np.std(y)

        reward = results['r']
        expression = results['expression']
        name = results['name']
        seed = results['seed']
        filename = f'dsr_{name}_{seed}'

        fig = plt.figure(figsize=(15,15), dpi=100)
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.scatter(y, yhat, s=2)

        if plot_sparta:
            plt.scatter(y, Rsparta, s=2, zorder=-1)
            plt.legend(['DSR', 'sparta'])

        plt.xlabel('Target (ground truth)')
        plt.ylabel('DSR model result')
        plt.title(f'reward = {reward}, NRMSE = {NRMSE} \n  expression = ' + expression)
        plt.grid('both')
        plt.xlim([10e-6, 1])
        plt.ylim([10e-6, 1])
        plt.savefig(logdir + '/' + filename)


def scatter_results_tensor(results, config):

    plot_sparta = True

    X, y = config['task']['dataset']
    logdir = config['training']['logdir']

    if plot_sparta:

        yhat_sparta = calc_tensor_sparta_yhat(config)

        yhat, _ = results['program'].execute(X)
        NRMSE = np.sqrt(np.mean((y-yhat)**2))/np.std(y)

        # de-flatten tensors:
        y = de_flatten_tensor(y)
        yhat = de_flatten_tensor(yhat)
        yhat_sparta = de_flatten_tensor(yhat_sparta)

        reward = results['r']
        expression = results['expression']
        name = results['name']
        seed = results['seed']
        filename = f'dsr_{name}_{seed}'



        fig, axs = plt.subplots(2, 3, figsize=(22,15), dpi=100)
        axs_flat = np.reshape(axs, (6,))
        axs[0, 0].scatter(y[0, 0, :], yhat[0, 0, :], s=2)
        axs[0, 0].set_title("T[1,1]")
        axs[0, 0].set_ylabel('DSR model result')

        axs[0, 1].scatter(y[0, 1, :], yhat[0, 1, :], s=2)
        axs[0, 1].set_title("T[1,2]")

        axs[0, 2].scatter(y[0, 2, :], yhat[0, 2, :], s=2)
        axs[0, 2].set_title("T[1,3]")

        axs[1, 0].scatter(y[1, 1, :], yhat[1, 1, :], s=2)
        axs[1, 0].set_title("T[2,2]")
        axs[1, 0].set_xlabel('target (ground truth)')
        axs[1, 0].set_ylabel('DSR model result')

        axs[1, 1].scatter(y[1, 2, :], yhat[1, 2, :], s=2)
        axs[1, 1].set_title("T[2,3]")
        axs[1, 1].set_xlabel('target (ground truth)')

        axs[1, 2].scatter(y[2, 2, :], yhat[2, 2, :], s=2)
        axs[1, 2].set_title("T[3,3]")
        axs[1, 2].set_xlabel('target (ground truth)')

        if plot_sparta:
            axs[0, 0].scatter(y[0, 0, :], yhat_sparta[0, 0, :], s=2, zorder=-1)
            axs[0, 0].set_ylim([min(yhat_sparta[0, 0, :]), max(yhat_sparta[0, 0, :])])

            axs[0, 1].scatter(y[0, 1, :], yhat_sparta[0, 1, :], s=2, zorder=-1)
            axs[0, 1].set_ylim([min(yhat_sparta[0, 1, :]), max(yhat_sparta[0, 1, :])])

            axs[0, 2].scatter(y[0, 2, :], yhat_sparta[0, 2, :], s=2, zorder=-1)
            axs[0, 2].set_ylim([min(yhat_sparta[0, 2, :]), max(yhat_sparta[0, 2, :])])

            axs[1, 0].scatter(y[1, 1, :], yhat_sparta[1, 1, :], s=2, zorder=-1)
            axs[1, 0].set_ylim([min(yhat_sparta[1, 1, :]), max(yhat_sparta[1, 1, :])])

            axs[1, 1].scatter(y[1, 2, :], yhat_sparta[1, 2, :], s=2, zorder=-1)
            axs[1, 1].set_ylim([min(yhat_sparta[1, 2, :]), max(yhat_sparta[1, 2, :])])

            axs[1, 2].scatter(y[2, 2, :], yhat_sparta[2, 2, :], s=2, zorder=-1)
            axs[1, 2].set_ylim([min(yhat_sparta[2, 2, :]), max(yhat_sparta[2, 2, :])])

        for ax in axs_flat:
            if plot_sparta:
                ax.legend(['DSR', 'sparta'])

        plt.suptitle(f'reward = {reward}, NRMSE = {NRMSE} \n  expression = ' + expression)
        plt.grid('both')
        plt.savefig(logdir + '/' + filename)
        # log_axes = [axs[0, 0], axs[0, 1], axs[1, 2]]
        # lin_axes = [axs[0, 2], axs[1, 0], axs[1, 1]]
        #
        # for ax in log_axes:
        #     ax.set_xscale('log')
        #     ax.set_yscale('log')
        #     if plot_sparta:
        #         ax.legend(['DSR', 'sparta'])
        #
        # for ax in lin_axes:
        #     if plot_sparta:
        #         ax.legend(['DSR', 'sparta'])


def calc_tensor_sparta_yhat(config):

    # load all required inputs
    dummy_config = copy.deepcopy(config['task'])
    dummy_config.pop('dataset')
    dummy_config['dataset'] = dummy_config.pop('dataset_info')
    dummy_config['dataset']['input'] = ['T1', 'T2', 'T3', 'T4', 'inv1', 'inv2']
    dummy_config['dataset']['output'] = 'bDelta'
    RANSdata, _ = load_frozen_RANS_dataset(dummy_config)
    T1 = RANSdata[:,0]
    T2 = RANSdata[:,1]
    T3 = RANSdata[:,2]
    T4 = RANSdata[:,3]
    inv1 = RANSdata[:,4]
    inv2 = RANSdata[:,5]

    model1 = False
    model2 = False
    model3 = True

    if model1:
        yhat_sparta = (24.94*inv1**2 + 2.65*inv2)*T1 + 2.96*T2 + (2.49*inv2 + 20.05)*T3 + (2.49*inv1 + 14.93)*T4
    if model2:
        yhat_sparta = T1*(0.46*inv1**2 + 11.68*inv2 - 0.30*(inv2**2) + 0.37) + T2*(-12.25*inv1 - 0.63*(inv2**2) + 8.23) + \
                T3*(-1.36*inv2 - 2.44) + T4*(-1.36*inv1 + 0.41*inv2 - 6.52)
    if model3:
        yhat_sparta = T1*(0.11*inv1*inv2 + 0.27*inv1*(inv2**2) - 0.13*inv1*(inv2**3) + 0.07*inv1*(inv2**4) \
                + 17.48*inv1 + 0.01*(inv1**2)*inv2 + 1.251*(inv1**2) + 3.67*inv2 + 7.52*(inv2**2) - 0.3) \
                + T2*(0.17*inv1*(inv2**2) - 0.16*inv1*(inv2**2) - 36.25*inv1 - 2.39*(inv1**2) + 19.22*inv2 + 7.04) \
                + T3*(-0.22*(inv1**2) + 1.8*inv2 + 0.07*(inv2**2)+2.65) + T4*(0.2*(inv1**2) - 5.23*inv2 - 2.93)

    return yhat_sparta


def contourplot_results_tensor(results, config):

    # re-read config file in output directory to find in and outputs
    logdir = config['training']['logdir']
    with open(logdir + '/config.json', encoding='utf-8') as f:
        config = json.load(f)

    name = results['name']
    seed = results['seed']

    cases = ['PH10595', 'CD12600', 'CBFS13700']

    for case in cases:

        config['task']['dataset']['name'] = case
        X, y = load_frozen_RANS_dataset(config['task'])

        yhat, _ = results['program'].execute(X)

        y = de_flatten_tensor(y)
        yhat = de_flatten_tensor(yhat)

        # load mesh data
        frozen = pickle.load(open(f'turbulence/frozen_data/{case}_frozen_var.p', 'rb'))
        data_i = frozen['data_i']

        mesh_x = data_i['meshRANS'][0, :, :]
        mesh_y = data_i['meshRANS'][1, :, :]

        filename = f'{logdir}/dsr_{name}_{seed}_contour_{case}'

        # here set up grid spec, dress 6 axes one by one using dress_contour_axes
        fig = plt.figure(figsize=(22, 15), dpi=250)  # , constrained_layout=False
        outer_grid = fig.add_gridspec(2, 3)  # outer_grid = fig11.add_gridspec(4, 4, wspace=0, hspace=0)
        axes = []

        for row in range(2):
            for col in range(3):

                inner_grid = outer_grid[row, col].subgridspec(2, 1, wspace=0, hspace=0)
                axs = inner_grid.subplots()
                r = row
                c = col

                if r == 1:
                    c = col + 1

                if (r == 1) and (c == 3):
                    r = 2
                    c = 2
                # outer_grid[row, col].suptitle(f'T{r + 1},{c + 1}')

                ymin = np.min(y[r, c, :])
                ymax = np.max(y[r, c, :])

                y_plot = y[r, c, :]
                yhat_plot = yhat[r, c, :]
                y_plot = np.reshape(y_plot, mesh_x.shape, order='F')
                yhat_plot = np.reshape(yhat_plot, mesh_x.shape, order='F')

                ax0 = axs[0].contourf(mesh_x, mesh_y, y_plot, levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
                axs[0].set_title(f'Target {r+1},{c+1}', y=1.0, pad=-14)
                ax1 = axs[1].contourf(mesh_x, mesh_y, yhat_plot, levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
                axs[1].set_title('DSR Model', y=1.0, pad=-14)
                axes.append([ax0, ax1, axs])

        for pair in axes:
            ax0 = pair[0]
            axs = pair[2]
            fig.colorbar(ax0, ax=axs[0])
            fig.colorbar(ax0, ax=axs[1])
        plt.savefig(filename, bbox_inches='tight')


def lumley_plot(results, config):

    # re-read config file in output directory to find in and outputs
    logdir = config['training']['logdir']
    with open(logdir + '/config.json', encoding='utf-8') as f:
        config = json.load(f)

    name = results['name']
    seed = results['seed']

    cases = ['CBFS13700', 'PH10595', 'CD12600']

    for case in cases:

        config['task']['dataset']['name'] = case

        filename = f'{logdir}/dsr_{name}_{seed}_contour_{case}'

        plot_lumley_comparison(results, case, config, filename)

def eval_expression(expression, X):

    vardict = {}
    for ii in range(X.shape[1]):
        vardict[f'x{ii+1}'] = X[:, ii]
    locals().update(vardict)

    expression = expression.replace('exp', 'np.exp')
    expression = expression.replace('log', 'np.log')
    expression = expression.replace('sin', 'np.sin')
    expression = expression.replace('cos', 'np.cos')

    yhat = eval(expression)

    return yhat

def case_contourplots(mesh_x, mesh_y, y, yhat, filename):

    inv_nrmse = 1 / (1 + np.sqrt(np.mean((y-yhat)**2))/np.std(y))

    yhat = np.reshape(yhat, mesh_x.shape, order='F')
    y = np.reshape(y, mesh_x.shape, order='F')

    ymin = np.min(y)
    ymax = np.max(y)

    fig, ax = plt.subplots(2, figsize=(15,10), dpi=250)
    fig.tight_layout()
    ax0 = ax[0].contourf(mesh_x, mesh_y, y, levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    ax[0].set_title('Target', y=1.05, pad=-14)
    ax1 = ax[1].contourf(mesh_x, mesh_y, yhat, levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    ax[1].set_title(f'DSR model, inv_nrmse = {inv_nrmse}', y=1.05, pad=-14)
    ax[0].axison = False
    ax[1].axison = False
    fig.colorbar(ax0, ax=ax[0])
    fig.colorbar(ax0, ax=ax[1])
    plt.savefig(filename)

def case_contourplots_with_sparta(mesh_x, mesh_y, y, yhat, ysparta, filename):

    yhat = np.reshape(yhat, mesh_x.shape, order='F')
    y = np.reshape(y, mesh_x.shape, order='F')
    ysparta = np.reshape(ysparta, mesh_x.shape, order='F')

    ymin = np.min(y)
    ymax = np.max(y)

    fig, ax = plt.subplots(3, figsize=(15,15), dpi=250)
    fig.tight_layout()
    ax0 = ax[0].contourf(mesh_x, mesh_y, y, levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    ax[0].set_title('Target', y=1.0, pad=-14)
    ax1 = ax[1].contourf(mesh_x, mesh_y, yhat, levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    ax[1].set_title('DSR model', y=1.0, pad=-14)
    ax2 = ax[2].contourf(mesh_x, mesh_y, ysparta, levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    ax[2].set_title('sparta', y=1.0, pad=-14)
    fig.colorbar(ax0, ax=ax[0])
    fig.colorbar(ax0, ax=ax[1])
    fig.colorbar(ax0, ax=ax[2])
    plt.savefig(filename+'sparta')

def contourplot_results_scalar(results, config):

    # re-read config file in output directory to find in and outputs
    logdir = config['training']['logdir']
    with open(logdir + '/config.json', encoding='utf-8') as f:
        config = json.load(f)

    name = results['name']
    seed = results['seed']

    cases = ['PH10595', 'CD12600', 'CBFS13700']

    for case in cases:
        config['task']['dataset']['name'] = case
        X, y = load_frozen_RANS_dataset(config['task'])

        frozen = pickle.load(open(f'turbulence/frozen_data/{case}_frozen_var.p', 'rb'))
        data_i = frozen['data_i']

        mesh_x = data_i['meshRANS'][0, :, :]
        mesh_y = data_i['meshRANS'][1, :, :]

        yhat, _ = results['program'].execute(X)

        filename = f'{logdir}/dsr_{name}_{seed}_contour_{case}'

        case_contourplots(mesh_x, mesh_y, y, yhat, filename)

def retrospecitvely_plot_contours(logdir, with_sparta=True):
    with open(logdir + '/config.json', encoding='utf-8') as f:
        config = json.load(f)

    cases = ['PH10595', 'CD12600', 'CBFS13700']

    for case in cases:
        config['task']['dataset']['name'] = case
        X, y = load_frozen_RANS_dataset(config['task'])

        ysparta = 2*1.4*X[:,0]*X[:,1]

        frozen = pickle.load(open(f'turbulence/frozen_data/{case}_frozen_var.p', 'rb'))
        data_i = frozen['data_i']

        mesh_x = data_i['meshRANS'][0, :, :]
        mesh_y = data_i['meshRANS'][1, :, :]

        # filename = f'{logdir}/dsr_{name}_{seed}_contour_{case}'

        # case_contourplots(mesh_x, mesh_y, y, yhat, filename)

        files = os.listdir(logdir)
        for filename in files:
            if filename[-7:] == 'hof.csv':
                df = pd.read_csv(f'{logdir}/{filename}')
                df_row = df.iloc[0]
                name = config['task']['name']
                seed = [int(s) for s in filename.split('_') if s.isdigit()][0]
                filename = f'{logdir}/dsr_{name}_{seed}_contour_{case}'

                yhat = eval_expression(df_row['expression'], X)

                if with_sparta:
                    case_contourplots_with_sparta(mesh_x, mesh_y, y, yhat, ysparta, filename)
                else:
                    case_contourplots(mesh_x, mesh_y, y, yhat, filename)

            plt.close('all')

    #
# def unused_code(inputs):
#
#     # load train_case data to allow to plot aong meshlines ect.
#     um = np.reshape(data_i['um'], mesh_x.shape, order='F')
#     vm = np.reshape(data_i['vm'], mesh_x.shape, order='F')
#     y = np.reshape(y, mesh_x.shape, order='F')
#
#
#     expression = 'x2/(x7 + exp(x2 + 9.408434534904377*x3/x1) - 0.9940896762252548)'
#     yhat_bad = eval_expression(expression, X)
#     yhat_bad = np.reshape(yhat_bad, mesh_x.shape, order='F')
#
#     # yhat_bad, _ = results['program'].execute(X) # normally this would be used, now the best performer is hardcoded in here
#     # yhat_bad = np.reshape(yhat_bad, mesh_x.shape, order='F')
#
#     expression = 'x1*x2*(-x1 + x3)/(x3*(x1 - 577.4204240086917*x7)*(x5 - 13.056646077044142))'
#     yhat_good = eval_expression(expression, X)
#     yhat_good = np.reshape(yhat_good, mesh_x.shape, order='F')
#
#     fig = plt.figure()
#     plt.contourf(mesh_x, mesh_y, yhat_good, levels=30)
#     plt.colorbar()
#     plt.show()
#
#     fig = plt.figure()
#     plt.contourf(mesh_x, mesh_y, yhat_bad, levels=30)
#     plt.colorbar()
#     plt.show()
#
#     fig = plt.figure()
#     plt.contourf(mesh_x[:,:10], mesh_y[:,:10], yhat_good[:,:10], levels=30)
#     plt.colorbar()
#     plt.show()
#
#
#
#     val1 = -7
#     val2 = 10
#
#     mesh_snippet = mesh_y[val1, :val2]
#
#
#     fig =plt.figure()
#     plt.plot(y[val1, :val2], mesh_snippet)
#     plt.plot(yhat_good[val1, :val2], mesh_snippet)
#     plt.plot(yhat_bad[val1, :val2], mesh_snippet)
#     plt.legend(['y target', 'yhat good', 'yhat bad'])
#     plt.show()

def load_hof(path):
    print('Not implemented')


def load_iterations(logdir):

    return_dict = {}
    for filename in os.listdir(logdir):
        split = filename.split('_')[-1].split('.')
        if (split[0].isnumeric()) and split[1] == 'csv':
            df_append = pd.read_csv(f'{logdir}/{filename}')
            return_dict[filename] = df_append

    return return_dict

def fetch_iteration_metrics(logdir, finished=True):

    plot_metrics = ['base_r_best', 'r_max_full', 'r_best_full', 'base_r_max', 'pg_loss', 'ent_loss', 'proc_time', 'wall_time', 'invalid_avg_full',
                    'invalid_avg_sub', 'n_novel_sub', 'l_avg_sub', 'l_avg_full', 'nfev_avg_full',
                    'nfev_avg_sub', 'eq_w_const_full', 'eq_w_const_sub',
                    'n_const_per_eq_full', 'n_const_per_eq_sub', 'a_ent_full', 'a_ent_sub',
                    'base_r_avg_sub']
    # plot_metrics = ['invalid_avg_full', 'n_novel_sub', 'l_avg_sub', 'l_avg_full', 'base_r_best', 'sample_metric']

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

def plot_iterations_metrics(logdir, finished=True):

    # find number of iterations completed:
    plot_dict = fetch_iteration_metrics(logdir, finished)

    # mean_n_novel_sub = np.mean(np.sum(np.array(plot_dict['n_novel_sub']), axis = 1))

    for metric in plot_dict.keys():
        fig = plt.figure()
        for arr in plot_dict[metric]:
            plt.plot(arr, label=None)
        if finished:
            plt.plot(np.mean(np.array(plot_dict[metric]), axis=0), color='red', label='mean')
        plt.xlabel('iterations')
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        plt.savefig(f'{logdir}/iterations_{metric}')

    # To possibly use debugger to compare two runs, create plot_dict2 for other run.
    # for metric in plot_metrics:
    #     fig = plt.figure()
    #     for arr in plot_dict[metric]:
    #         plt.plot(arr, label=None, color='C0')
    #     for arr in plot_dict2[metric]:
    #         plt.plot(arr, label=None, color='C1')
    #     if finished:
    #         plt.plot(np.mean(np.array(plot_dict[metric]), axis=0), color='red', label='mean')
    #     plt.xlabel('iterations')
    #     plt.ylabel(metric)
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig(f'{logdir}/iterations_{metric}')

# def compare_dicts(bsl, run):
#     diff = []
#     for key in bsl.keys():
#         if isinstance(bsl[key], dict):
#             diff.append(compare_dicts(bsl[key], run[key]))
#         elif key in ['name', 'logdir', 'verbose']:
#             pass
#         else:  # compare
#             if not bsl[key] == run[key]:
#                 diff.append(f'{key}_{run[key]}')
#
#     diff = [item for item in diff if item is not 'baseline']
#     if len(diff) == 0:
#         return 'baseline'
#     else:
#         return '_'.join(diff)

def compare_dicts(bsl, run):
    diff = []
    for key in bsl.keys():
        if isinstance(bsl[key], dict):
            diff += compare_dicts(bsl[key], run[key])
        elif key in ['name', 'logdir', 'verbose', 'save_batch', 'save_controller']:
            pass
        else:  # compare
            if not bsl[key] == run[key]:
                diff.append((key, run[key]))

    diff = [item for item in diff if item is not 'baseline']
    if len(diff) == 0:
        return ['baseline']
    else:
        return diff

def plot_sensitivity_results(logdir):
    try:
        shutil.rmtree(os.path.join(logdir, 'results'))
    except FileNotFoundError:
        pass

    dirlist = os.listdir(logdir)

    os.mkdir(os.path.join(logdir, 'results'))

    with open(os.path.join(logdir, 'config_baseline.json'), encoding='utf-8') as f:
        config_bsl = json.load(f)

    dirlist.remove('config_baseline.json')
    # ratios used to scale duration
    machine_dur_ratios = {'OW': 1,
                          'M15': 0.47895466499411693,
                          'M18': 0.6902561859941414,
                          'M3': 0.8175768873161131}

    first_write = True
    baseline = None


    try:  # remove existing results file since new will be created.
        os.remove(os.path.join(logdir, 'results', 'results.csv'))
    except FileNotFoundError:
        pass
    parameters = []

    all_results = {}
    for run in dirlist:
        run_dir = os.path.join(logdir, run)
        with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
            config_run = json.load(f)

        machine_name = config_run['task']['name'].split('_')[0]
        diff = compare_dicts(config_bsl, config_run)
        run_name = machine_name
        if diff[0] == 'baseline':
            run_name += '_baseline'
            baseline = run_name
            parameters.append('baseline')
        else:
            for item in diff:
                run_name += f'_{item[0]}_{item[1]}'
                if item[0] not in parameters:
                    parameters.append(item[0])


        run_dict = fetch_iteration_metrics(os.path.join(logdir, run), finished=False)

        result_col = ['run_name']
        result_val = [run_name]

        tmp_arr = np.array(run_dict['proc_time'])
        result_col.append('adjusted_avg_proc_duration')
        result_val.append(round(np.mean(np.sum(tmp_arr, axis=1)) * machine_dur_ratios[machine_name]))

        save_dict = {}

        for key in run_dict:
            tmp_arr = np.array(run_dict[key])
            save_dict[key] = {}
            save_dict[key]['mean'] = np.mean(tmp_arr, axis=0)
            save_dict[key]['std'] = np.std(tmp_arr, axis=0)
            save_dict[key]['max'] = np.max(tmp_arr, axis=0)
            tmp_arr = np.sort(tmp_arr, axis=0)
            save_dict[key]['5max'] = np.mean(tmp_arr[-5:, :], axis=0)
            result_col.extend(['_'.join([mode, key]) for mode in save_dict[key].keys()])
            result_val.extend([save_dict[key][mode][-1] for mode in save_dict[key].keys()])

        # write results to csv:
        df_append = pd.DataFrame([result_val], columns=result_col)
        df_append.to_csv(os.path.join(logdir, 'results', 'results.csv'), mode='a', header=first_write, index=False)

        if first_write:  # used to only write header once
            first_write = False

        all_results[run_name] = save_dict

    for key in all_results:
        all_results[key]['varied'] = []
        for parameter in parameters:
            if parameter in key:
                all_results[key]['varied'].append(parameter)

    plot_dict = {key: [baseline] if baseline else [] for key in parameters}
    plot_dict['baseline'] = []
    plot_dict['all'] = all_results.keys()

    # if logdir.split('_')[-1] == 'kDeficit':
    #     plot_dict['compare'] = ['OW_baseline',
    #                             'M18_initializer_uniform_learning_rate_0.01',
    #                             'OW_initializer_normal_learning_rate_0.01',
    #                             'OW_learning_rate_0.01',
    #                             'M18_num_units_128_initializer_normal_learning_rate_0.01',
    #                             'OW_num_units_256_initializer_normal_learning_rate_0.01',
    #                             'OW_num_units_256',
    #                             'OW_entropy_weight_0.0025',
    #                             'M3_initializer_normal']
    # else:
    #     plot_dict['compare'] = ['OW_baseline',
    #                             'M15_learning_rate_0.01',
    #                             'M18_learning_rate_0.01',
    #                             'OW_initializer_normal_learning_rate_0.01',
    #                             'M3_num_units_64_initializer_normal_learning_rate_0.01',
    #                             'OW_num_units_256']

    for parameter in parameters:
        for run in all_results:
            if parameter in all_results[run]['varied']: # and len(all_results[run]['varied']) == 1:
                plot_dict[parameter].append(run)

    for key in plot_dict:
        plot_dir = os.path.join(logdir, 'results', key)
        os.makedirs(plot_dir, exist_ok=True)
        create_plots(all_results, plotmode='mean', plotlist=plot_dict[key], plot_dir=plot_dir)
        create_plots(all_results, plotmode='max', plotlist=plot_dict[key], plot_dir=plot_dir)
        create_plots(all_results, plotmode='5max', plotlist=plot_dict[key], plot_dir=plot_dir)

def create_plots(all_results, plotmode, plotlist, plot_dir):

    for metric in all_results[list(all_results.keys())[0]]:
        if metric == 'varied':
            pass
        else:
            plt.figure(figsize=(12,10))
            for run in plotlist:
                plt.plot(all_results[run][metric][plotmode])
            plt.xlabel('iterations')
            plt.ylabel(' '.join([plotmode, metric]))
            plt.legend(plotlist)
            plt.grid('both')
            plt.savefig(f'{plot_dir}/{metric}_{plotmode}.png')
            plt.close('all')


if __name__ == "__main__":
    dsrpath = os.path.abspath(__file__)
    os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    ############################################################################
    # #use function below to plot the contours when the logs are already written
    # retrospecitvely_plot_contours('../logs_completed/log_2021-04-28-152005_kdeficit_10msamples')
    # retrospecitvely_plot_contours('./log/log_2021-08-25-170231', False)
    #
    # print('end')

    # logdir = '../logs_completed/log_2021-06-04-130021_2M_bDelta'
    # logdir = '../logs_completed/log_comparison_of_metrics/reg_mspe'
    # logdir = '../logs_completed/log_2021-07-14-163737_10M_run'
    # logdir = './log/log_2021-11-24-153425'
    # logdir = './log/log_2021-08-25-170231'

    # plot_iterations_metrics(logdir, finished=True)


    # logdir = '../logs_completed/sensitivity_analysis_kDeficit'
    # logdir = '../logs_completed/sensitivity_analysis_bDelta'
    logdir = '../logs_completed/sensitivity_compare'
    plot_sensitivity_results(logdir)

    print('end')




