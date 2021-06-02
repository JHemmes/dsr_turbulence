# Jasper Hemmes - 2021

import os
import sys
import pickle
import numpy as np
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from dsr.turbulence.dataprocessing import load_frozen_RANS_dataset
from dsr.program import from_str_tokens


def plot_results(results, config):
    inputs = config['task']['dataset_info']['input']
    for input in inputs:
        if input in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']:
            tensor = True
            break
        else:
            tensor = False

    if tensor:
        print('do nothing (yet)')
        # scatter_results_tensor(results, config)
        # contourplot_results_tensor(results, config)
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

        yhat = results['program'].cython_execute(X)
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

        sparta1 = '(24.94*inv1**2 + 2.65*inv2)*T1 + 2.96*T2 + (2.49*inv2 + 20.05)*T3 + (2.49inv1 + 14.93)*T4'
        sparta2 = 'T1*(0.46*inv1**2 + 11.68*inv2 -0.30inv2**2+0.37) + T2*(-12.25*inv1 - 0.63inv2**2 + 8.23)' \
                  'T3*(-1.36inv2 - 2.44) + T4*(-1.36inv1 + 0.41*inv2 - 6.52)'
        sparta3 = 'T1*(0.11*inv1*inv2 + 0.27*inv1*(inv2**2) -0.13inv1*(inv2**3) + 0.07*inv1*(inv2**4) + 17.48*inv1 ' \
                  '+ 0.01*(inv1**2)*inv2 + 1.251*(inv1**2) + 3.67*inv2 + 7.52*(inv2**2) -0.3)' \
                  '+ T2*(0.17*inv1*(inv2**2) - 0.16*inv1*(inv2**2) - 36.25*inv1 - 2.39*(inv1**2) +19.22*inv2 +7.04)' \
                  '+ T3*(-0.22*(inv1**2) - 5.23*inv2 - 2.93)'


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

        yhat = results['program'].cython_execute(X)
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

    yhat = np.reshape(yhat, mesh_x.shape, order='F')
    y = np.reshape(y, mesh_x.shape, order='F')

    ymin = np.min(y)
    ymax = np.max(y)

    fig, ax = plt.subplots(2, figsize=(15,10), dpi=250)
    fig.tight_layout()
    ax0 = ax[0].contourf(mesh_x, mesh_y, y, levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    ax[0].set_title('Target', y=1.0, pad=-14)
    ax1 = ax[1].contourf(mesh_x, mesh_y, yhat, levels=30, vmin=ymin, vmax=ymax, cmap='Reds')
    ax[1].set_title('DSR model', y=1.0, pad=-14)
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

        yhat = results['program'].cython_execute(X)

        filename = f'{logdir}/dsr_{name}_{seed}_contour_{case}'

        case_contourplots(mesh_x, mesh_y, y, yhat, filename)

def retrospecitvely_plot_contours(logdir):
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

                # case_contourplots(mesh_x, mesh_y, y, yhat, filename)
                case_contourplots_with_sparta(mesh_x, mesh_y, y, yhat, ysparta, filename)
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
#     # yhat_bad = results['program'].cython_execute(X) # normally this would be used, now the best performer is hardcoded in here
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



if __name__ == "__main__":
    dsrpath = os.path.abspath(__file__)
    os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    ############################################################################
    # #use function below to plot the contours when the logs are already written
    # retrospecitvely_plot_contours('../logs_completed/log_2021-04-28-152005_kdeficit_10msamples')
    # # retrospecitvely_plot_contours('../logs_completed/log_2021-04-28-152005_kdeficit_10msamples')
    #
    # print('end')




    expression = 'x2*(-0.7289507583079632*x1 + 0.076926414459202349*x1/(x3 + x5) - 0.095542880884756653)'



    with open('config_kDeficit.json', encoding='utf-8') as f:
        config = json.load(f)


    X, y = load_frozen_RANS_dataset(config['task'])

    y_hat = eval_expression(expression, X)

    n_rep = 1


    print(f'{n_rep} evaluations of reward for different error metrics')

    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = -np.mean((y - y_hat)**2)

    print(f'neg_mse took: {round(time.time() - starttime,2)}')
    print(ans)

    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = -np.sqrt(np.mean((y - y_hat)**2))
    print(f'neg_rmse took: {round(time.time() - starttime,2)}')
    print(ans)



    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        var_y = np.std(y)
        ans = -np.mean((y - y_hat)**2)/var_y
    print(f'neg_nmse took: {round(time.time() - starttime,2)}')
    print(ans)


    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        var_y = np.std(y)
        ans = -np.sqrt(np.mean((y - y_hat)**2)/var_y)
    print(f'neg_nrmse took: {round(time.time() - starttime,2)}')


    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = -np.log(1 + np.mean((y - y_hat)**2))
    print(f'neglog_mse took: {round(time.time() - starttime,2)}')
    print(ans)


    args = [1.]
    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = 1/(1 + args[0]*np.mean((y - y_hat)**2))
    print(f'inv_mse took: {round(time.time() - starttime,2)}')


    args = [1.]
    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        var_y = np.std(y)
        ans = 1/(1 + args[0]*np.mean((y - y_hat)**2)/var_y)
    print(f'inv_nmse took: {round(time.time() - starttime,2)}')
    print(ans)

    args = [1.]
    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        var_y = np.std(y)
        ans = 1/(1 + args[0]*np.sqrt(np.mean((y - y_hat)**2)/var_y))
    print(f'inv_nrmse took: {round(time.time() - starttime,2)}, with var_y IN the loop')
    print(ans)

    args = [1.]
    starttime = time.time()
    var_y = np.std(y)
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = 1/(1 + args[0]*np.sqrt(np.mean((y - y_hat)**2)/var_y))
    print(f'inv_nrmse took: {round(time.time() - starttime,2)}, with var_y OUT the loop')
    print(ans)


    starttime = time.time()
    for _ in range(n_rep):
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        ans = -np.mean((y-y_hat)**2 /np.sqrt(0.001**2 + y**2))
    print(f'reg_mspe: {round(time.time() - starttime,2)}')
    print(ans)

