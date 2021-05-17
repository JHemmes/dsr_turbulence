# Jasper Hemmes - 2021

import os
import sys
import pickle
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import time


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


def plot_results(results, config):

    # set input and output variables
    X, y = config['task']['dataset']

    # re-read config file in output directory to find case name
    logdir = config['training']['logdir']
    with open(logdir + '/config.json', encoding='utf-8') as f:
        config = json.load(f)

    case = config['task']['dataset']['name']

    frozen = pickle.load(open(f'turbulence/frozen_data/{case}_frozen_var.p', 'rb'))

    data_i = frozen['data_i']

    # load case data to allow to plot aong meshlines ect.

    mesh_x = data_i['meshRANS'][0,:,:]
    mesh_y = data_i['meshRANS'][1,:,:]

    um = np.reshape(data_i['um'], mesh_x.shape, order='F')
    vm = np.reshape(data_i['vm'], mesh_x.shape, order='F')
    y = np.reshape(y, mesh_x.shape, order='F')


    expression = 'x2/(x7 + exp(x2 + 9.408434534904377*x3/x1) - 0.9940896762252548)'
    yhat_bad = eval_expression(expression, X)
    yhat_bad = np.reshape(yhat_bad, mesh_x.shape, order='F')

    # yhat_bad = results['program'].cython_execute(X) # normally this would be used, now the best performer is hardcoded in here
    # yhat_bad = np.reshape(yhat_bad, mesh_x.shape, order='F')

    expression = 'x1*x2*(-x1 + x3)/(x3*(x1 - 577.4204240086917*x7)*(x5 - 13.056646077044142))'
    yhat_good = eval_expression(expression, X)
    yhat_good = np.reshape(yhat_good, mesh_x.shape, order='F')

    fig = plt.figure()
    plt.contourf(mesh_x, mesh_y, yhat_good, levels=30)
    plt.colorbar()
    plt.show()

    fig = plt.figure()
    plt.contourf(mesh_x, mesh_y, yhat_bad, levels=30)
    plt.colorbar()
    plt.show()

    fig = plt.figure()
    plt.contourf(mesh_x[:,:10], mesh_y[:,:10], yhat_good[:,:10], levels=30)
    plt.colorbar()
    plt.show()



    val1 = -7
    val2 = 10

    mesh_snippet = mesh_y[val1, :val2]


    fig =plt.figure()
    plt.plot(y[val1, :val2], mesh_snippet)
    plt.plot(yhat_good[val1, :val2], mesh_snippet)
    plt.plot(yhat_bad[val1, :val2], mesh_snippet)
    plt.legend(['y target', 'yhat good', 'yhat bad'])
    plt.show()







    print('end')

if __name__ == "__main__":





    print('end')