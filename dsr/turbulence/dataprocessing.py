# Jasper Hemmes - 2021

import os
import sys
import pickle
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import time
from dsr.utils import load_pickle, save_pickle

def load_benchmark_dataset(config_task):
    if config_task['dataset']['output'] == 'bDelta':
        n_points = 50
        np.random.seed(0)
        X = np.zeros((n_points, 11))
        X[:, 0] = np.random.uniform(0, 2, n_points)    # use as tensors when enforcing sum
        X[:, 1] = np.random.uniform(0, 0.1, n_points)  # use as tensors when enforcing sum
        X[:, 2] = np.random.uniform(0, 1, n_points)    # use as tensors when enforcing sum
        X[:, 3] = np.random.uniform(-1, 0, n_points)   # use as tensors when enforcing sum
        X[:, 4] = np.random.uniform(-0.0002, 0.0003, n_points)    # use to scale "tensors"
        X[:, 5] = np.random.uniform(2, 5, n_points)               # use to scale "tensors"
        X[:, 6] = np.random.uniform(-1, 1, n_points)              # use to scale "tensors"
        X[:, 7] = np.random.uniform(0, 2, n_points)               # use to scale "tensors"
        X[:, 8] = np.random.uniform(0, 0.1, n_points)             # use to scale "tensors"
        X[:, 9] = np.random.uniform(-1, 0, n_points)
        X[:, 10] = np.random.uniform(1, 7, n_points)

        term0 = X[:, 0] * (0.1 * (X[:, 5]+X[:, 6]) + 1.7 * np.log(X[:, 7]) + 0.23)
        term1 = X[:, 1] * (X[:, 7] * X[:, 8] * 3 + 0.7*np.exp(X[:, 8]) + 15)
        term2 = X[:, 2] * (np.exp(X[:, 6]) + 3/(np.log(X[:, 5])))
        term3 = X[:, 3] * (X[:, 4] * 10 + X[:, 7])

        noise = np.random.normal(0, 0.15, (n_points,))

        y = term0 + term1 + term2 + term3 + noise

        # check expression fitness without noise
        var_y = np.var(y)
        y_hat = term0 + term1 + term2 + term3
        inv_nrmse = 1 / (1 + np.sqrt(np.mean((y - y_hat) ** 2) / var_y))
        print(f'If inv_nrmse is above {round(inv_nrmse, 4)}, the benchmark expression is overfitted on the noise')


        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(y)
        # plt.plot(term0)
        # plt.plot(term1)
        # plt.plot(term2)
        # plt.plot(term3)
        # plt.legend(['y', 'term0', 'term1', 'term2', 'term3'])

    else:
        np.random.seed(0)
        X = np.zeros((30, 7))
        X[:, 0] = np.random.uniform(0, 7, 30)   # use this
        X[:, 1] = np.random.uniform(0, 0.1, 30) # use this
        X[:, 2] = np.random.uniform(0, 1, 30)   # use this
        X[:, 3] = np.random.uniform(-1, 0, 30)  # use this
        X[:, 4] = np.random.uniform(-0.0002, 0.0003, 30)  # use this
        X[:, 5] = np.random.uniform(0, 2, 30)
        X[:, 6] = np.random.uniform(-1, 1, 30)
        noise = np.random.normal(0, 0.15, (30,))

        y = 4.5 * np.exp(X[:, 2]) * X[:, 1] + X[:, 3] ** 2 + 0.5 * np.log(1.7 * X[:, 1] + 1 / X[:, 0]) + noise

        # check expression fitness without noise
        var_y = np.var(y)
        y_hat = 4.5 * np.exp(X[:, 2]) * X[:, 1] + X[:, 3] ** 2 + 0.5 * np.log(1.7 * X[:, 1] + 1 / X[:, 0])
        inv_nrmse = 1 / (1 + np.sqrt(np.mean((y - y_hat) ** 2) / var_y))
        print(f'If inv_nrmse is above {round(inv_nrmse, 4)}, the benchmark expression is overfitted on the noise')


    # Quick visual check to see possible correlation between vars and noise
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('tkagg')
    #
    # for ii in range(X.shape[-1]):
    #     plt.figure()
    #     plt.scatter(noise, X[:,ii])


    return X, y


def calc_sij_rij(grad_u, omega, normalize=True):
    """ Calculates the strain rate and rotation rate tensors.  Normalizes by omega:
    Sij = k/eps * 0.5* (grad_u  + grad_u^T) = 1/omega * 0.5* (grad_u  + grad_u^T)
    Rij = k/eps * 0.5* (grad_u  - grad_u^T) = 1/omega * 0.5* (grad_u  - grad_u^T)
    :param grad_u: velocity gradient (3, 3, num_of_points)
    :return: Sij, Rij (3, 3, num_of_points)
    """
    omega = np.maximum(omega, 1e-8)  # make sure omega is not zero

    # this is some form of limiter is the k-omega-sst model
    tmp = 0.5*(grad_u + np.transpose(grad_u, (1,0,2)))
    omega_lim = np.zeros(omega.shape)
    for ii in range(omega.shape[0]):
        omega_lim[ii] = 1. / max(np.sqrt(2 * np.tensordot(tmp[:,:,ii], tmp[:,:,ii])) / 0.31, omega[ii])

    if normalize:
        Sij = omega_lim*0.5*(grad_u + np.transpose(grad_u, (1,0,2)))
        Rij = omega_lim*0.5*(grad_u - np.transpose(grad_u, (1,0,2)))
    else:
        Sij = 0.5*(grad_u + np.transpose(grad_u, (1,0,2)))
        Rij = 0.5*(grad_u - np.transpose(grad_u, (1,0,2)))

    # ensure Sij is traceless, should be the case but machine precision could result in a very small magnitude trace
    Sij = Sij - np.repeat(np.eye(3), omega.shape[0], axis=1).reshape(3, 3, omega.shape[0]) * np.trace(Sij)/3

    return Sij, Rij


def calc_tensor_basis(Sij, Rij):
    """ Calculates the integrity basis of the base tensor expansion.

    :param Sij: Mean strain-rate tensor (3, 3, num_of_points)
    :param Rij: Mean rotation-rate tensor (3, 3, num_of_points)
    :return: T: Base tensor series (10, 3, 3, num_of_points)
    """
    num_of_cells = Sij.shape[2]
    T = np.ones([10, 3, 3, num_of_cells]) * np.nan
    for i in range(num_of_cells):
        sij = Sij[:, :, i]
        rij = Rij[:, :, i]
        T[0, :, :, i] = sij
        T[1, :, :, i] = np.dot(sij, rij) - np.dot(rij, sij)
        T[2, :, :, i] = np.dot(sij, sij) - 1. / 3. * np.eye(3) * np.trace(np.dot(sij, sij))
        T[3, :, :, i] = np.dot(rij, rij) - 1. / 3. * np.eye(3) * np.trace(np.dot(rij, rij))
        T[4, :, :, i] = np.dot(rij, np.dot(sij, sij)) - np.dot(np.dot(sij, sij), rij)
        T[5, :, :, i] = np.dot(rij, np.dot(rij, sij)) \
                        + np.dot(sij, np.dot(rij, rij)) \
                        - 2. / 3. * np.eye(3) * np.trace(np.dot(sij, np.dot(rij, rij)))
        T[6, :, :, i] = np.dot(np.dot(rij, sij), np.dot(rij, rij)) - np.dot(np.dot(rij, rij), np.dot(sij, rij))
        T[7, :, :, i] = np.dot(np.dot(sij, rij), np.dot(sij, sij)) - np.dot(np.dot(sij, sij), np.dot(rij, sij))
        T[8, :, :, i] = np.dot(np.dot(rij, rij), np.dot(sij, sij)) + np.dot(np.dot(sij, sij), np.dot(rij, rij)) \
                        - 2. / 3. * np.eye(3) * np.trace(np.dot(np.dot(sij, sij), np.dot(rij, rij)))
        T[9, :, :, i] = np.dot(np.dot(rij, np.dot(sij, sij)), np.dot(rij, rij)) \
                        - np.dot(np.dot(rij, np.dot(rij, sij)), np.dot(sij, rij))

        # Enforce zero trace for anisotropy
        for j in range(10):
            T[j, :, :, i] = T[j, :, :, i] - 1. / 3. * np.eye(3) * np.trace(T[j, :, :, i])

    return T


def calc_invariants(Sij, Rij):
    """ Given the non-dimensionalized mean strain rate and mean rotation rate tensors Sij and Rij,
    this returns a set of normalized scalar invariants.

    :param Sij: k/eps * 0.5 * (du_i/dx_j + du_j/dx_i)
    :param Rij: k/eps * 0.5 * (du_i/dx_j - du_j/dx_i)
    :return: invariants: The num_points X num_scalar_invariants numpy matrix of scalar invariants
    """
    num_of_invariants = 5
    num_of_cells = Sij.shape[2]
    invariants = np.zeros((num_of_invariants, num_of_cells))

    for ii in range(num_of_cells):
        invariants[0, ii] = np.trace(np.dot(Sij[:, :, ii], Sij[:, :, ii]))
        invariants[1, ii] = np.trace(np.dot(Rij[:, :, ii], Rij[:, :, ii]))
        invariants[2, ii] = np.trace(np.dot(Sij[:, :, ii], np.dot(Sij[:, :, ii], Sij[:, :, ii])))
        invariants[3, ii] = np.trace(np.dot(Rij[:, :, ii], np.dot(Rij[:, :, ii], Sij[:, :, ii])))
        invariants[4, ii] = np.trace(np.dot(np.dot(Rij[:, :, ii], Rij[:, :, ii]), np.dot(Sij[:, :, ii], Sij[:, :, ii])))

    return invariants

def flatten_tensor(tensor):
    """ Flattens symmetric tensor.

    :param tensor: Given tensor (3,3,num_of_points)
    :return: tensor_flatten: Flatted tensor (6*num_of_points)
    """
    num_of_points = tensor.shape[2]
    tensor_flatten = np.zeros([num_of_points, 6])
    tensor_flatten[:, 0] = tensor[0, 0, :]
    tensor_flatten[:, 1] = tensor[0, 1, :]
    tensor_flatten[:, 2] = tensor[0, 2, :]
    tensor_flatten[:, 3] = tensor[1, 1, :]
    tensor_flatten[:, 4] = tensor[1, 2, :]
    tensor_flatten[:, 5] = tensor[2, 2, :]
    return tensor_flatten.flatten('A')


def de_flatten_tensor(tensor_flat):
    """De-Flattens symmetric tensor.

    :param tensor: Given Flattened tensor (6*num_of_points)
    :return: tensor: De-Flattened tensor (3,3,num_of_points)
    """
    num_of_points = int(tensor_flat.shape[0]/6)
    tensor = np.zeros((3, 3, num_of_points))

    tensor[0, 0, :] = tensor_flat[::6]
    tensor[0, 1, :] = tensor_flat[1::6]
    tensor[1, 0, :] = tensor_flat[1::6]
    tensor[0, 2, :] = tensor_flat[2::6]
    tensor[2, 0, :] = tensor_flat[2::6]
    tensor[1, 1, :] = tensor_flat[3::6]
    tensor[1, 2, :] = tensor_flat[4::6]
    tensor[2, 1, :] = tensor_flat[4::6]
    tensor[2, 2, :] = tensor_flat[5::6]

    return tensor

def broadcast_scalar(scalar):
    """ broadcasts scalar array to use with flattened tensors.

    :param scalar: Given scalar (num_of_points,)
    :return: broadcast_scalar: expanded scalar (6*num_of_points)
    """
    return np.repeat(scalar, 6)

def broadcast(scalar, flat_bool):
    """ Returns broadcasted scalar or not, depending on flat_bool """

    if flat_bool:
        return broadcast_scalar(scalar)
    else:
        return scalar


def load_frozen_RANS_dataset(config_task):

    case = config_task['name']
    output = config_task['output']
    inputs = config_task['input']
    skip_wall = False
    if config_task['skip_wall']:
        # set skip wall to right value depending on dataset:
        if 'PH' in case:
            skip_wall = 15
        elif 'CD' in case:
            skip_wall = 2
        elif 'CBFS' in case:
            skip_wall = 6

    pickle_path = f'turbulence/frozen_data/pickled_' + '_'.join([case] + [f'NW_{skip_wall}'] + [output] + inputs) + '.p'

    try:
        X, y = load_pickle(pickle_path)
        # X, y = load_pickle('thisfiledoesntexists.p')
    except FileNotFoundError:

        frozen = pickle.load(open(f'turbulence/frozen_data/{case}_frozen_var.p', 'rb'))
        data_i = frozen['data_i']

        if skip_wall:
            # shorten all variables in frozen dict to only keep relevant data
            mesh_x = data_i['meshRANS'][0, :, :]
            n_points_x = mesh_x.shape[0]
            bool_arr = np.ones(data_i['k'].shape) == 1

            bool_arr[:skip_wall*n_points_x] = False
            bool_arr[-skip_wall*n_points_x:] = False

            # this should possibly happen outside skip_wall, but for now it remains to ensure plotting receives all data
            bool_arr[data_i['k'] < 0.00001] = False

            for key in data_i:
                if isinstance(data_i[key], np.ndarray):
                    if data_i[key].shape[0] == bool_arr.shape[0]:
                        data_i[key] = data_i[key][bool_arr]
                    elif data_i[key].shape[-1] == bool_arr.shape[0]:
                        data_i[key] = data_i[key][:, :, bool_arr]


        # Check what inputs need to be calculated:
        grad_tens = False
        invar = False
        tens = False
        flatten = False

        for input in inputs:
            if input in ['grad_u_T1', 'grad_u_T2', 'grad_u_T3', 'grad_u_T4', 'grad_u_T5', 'grad_u_T5', 'grad_u_T7', 'grad_u_T8', 'grad_u_T9', 'grad_u_T10']:
                grad_tens = True
            if input in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']:
                tens = True
                flatten = True
            if input in ['inv1', 'inv2', 'inv3', 'inv4', 'inv5']:
                invar = True

        # calculate correct values

        if tens:
            Sij, Rij = calc_sij_rij(data_i['grad_u'], data_i['omega_frozen'])
            Tij = calc_tensor_basis(Sij, Rij)

        if grad_tens:
            Sij, Rij = calc_sij_rij(data_i['grad_u'], data_i['omega_frozen'])
            Tij = calc_tensor_basis(Sij, Rij)
            grad_u = data_i['grad_u']

        if invar:
            if 'Sij' not in locals():  # check whether Sij and Rij have been calculated already
                Sij, Rij = calc_sij_rij(data_i['grad_u'], data_i['omega_frozen'])

            invariants = calc_invariants(Sij, Rij)


        y = data_i[output]

        n_inputs = len(inputs)

        if flatten:
            # initialise for flattened tensors
            X = np.zeros((max(y.shape)*6, n_inputs))
            y = flatten_tensor(y)
        else:
            # initialise for scalars
            X = np.zeros((max(y.shape), n_inputs))

        for index, value in enumerate(inputs):
            if value in data_i.keys():
                # input present in frozen data
                input = data_i[value]
                if len(input.shape) == 1:
                    # input is a scalar, might need to be broadcasted
                    X[:, index] = broadcast(input, flatten)
                else:
                    # input is a tensor, needs flattening
                    X[:,index] = flatten_tensor(input)

            elif value in ['grad_u_T1', 'grad_u_T2', 'grad_u_T3', 'grad_u_T4', 'grad_u_T5',
                         'grad_u_T5', 'grad_u_T7', 'grad_u_T8', 'grad_u_T9', 'grad_u_T10']:
                # input is a product of base tensors and grad_u
                tensor_idx = int(value[value.find('_u_T') + 4:]) - 1
                input = np.zeros(grad_u.shape[-1])
                for ii in range(grad_u.shape[-1]):
                    input[ii] = np.tensordot(grad_u[:, :, ii], Tij[tensor_idx, :, :, ii])
                X[:,index] = broadcast(input, flatten)
            elif value in ['inv1', 'inv2', 'inv3', 'inv4', 'inv5']:
                # input is on of pope's base tensor invariants
                inv_idx = int(value[-1])-1
                X[:,index] = broadcast(invariants[inv_idx, :], flatten)
            elif value in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']:
                # input is one of the base tensors
                tensor_idx = int(value[1:]) - 1
                tensor = Tij[tensor_idx, :, :, :]
                X[:,index] = flatten_tensor(tensor)
            else:
                print(f'{value} as input is currently not supported, might be a typo?')
                data_i[value] # errors on this statement, is supposed to stop the program

        # bool_test required for scatters, and needs to be created before X is clipped
        # bool_test = ~(X == 0).any(axis=1)

        if skip_wall: # delete all points where input vars are zero, only relevant for CBFS case
            # this should probably not be dependent on skip wall value, but is now used to maintain possibility to return all points

            # base tensors have some zero components, avoid removing those points:
            cols = np.ones(len(inputs)) == 1
            for ii in range(len(inputs)):
                if inputs[ii] in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']:
                    cols[ii] = False

            y = y[~(X[:,cols] == 0).any(axis=1)]
            X = X[~(X[:,cols] == 0).any(axis=1), :]

        save_pickle(pickle_path, (X,y))

        # # can be used to plot regions of zero pressure gradient:
        # mesh_x = data_i['meshRANS'][0, :, :]
        # mesh_y = data_i['meshRANS'][1, :, :]
        # mesh_x_flat = mesh_x.flatten(order='F').T
        # mesh_y_flat = mesh_y.flatten(order='F').T
        # #
        # k = np.reshape(data_i['k'], mesh_x.shape, order='F')
        # omega_frozen = np.reshape(data_i['omega_frozen'], mesh_x.shape, order='F')
        # nut_frozen = np.reshape(data_i['nut_frozen'], mesh_x.shape, order='F')
        #
        # import matplotlib
        # matplotlib.use('tkagg')
        #
        # mesh_x_flat_keep = mesh_x_flat[bool_arr]
        # mesh_y_flat_keep = mesh_y_flat[bool_arr]
        # mesh_x_flat_keep = mesh_x_flat_keep[bool_test]
        # mesh_y_flat_keep = mesh_y_flat_keep[bool_test]
        #
        # for ii in range(len(inputs)):
        #     var = inputs[ii]
        #     mesh_x_short = mesh_x_flat_keep[X[:, ii] == 0]
        #     mesh_y_short = mesh_y_flat_keep[X[:, ii] == 0]
        #     plt.figure()
        #     plt.scatter(mesh_x_short, mesh_y_short)
        #     plt.title(var)
        #
        # for ii in range(len(inputs)):
        #     var = inputs[ii]
        #     # mesh_x_short = mesh_x_flat_keep[X[:, ii] == 0]
        #     # mesh_y_short = mesh_y_flat_keep[X[:, ii] == 0]
        #     plt.figure()
        #     plt.scatter(mesh_x_flat_keep, mesh_y_flat_keep)
        #     plt.title(var)
        #
        # plt.show()
        #
        #
        # plt.figure()
        # plt.scatter(mesh_x_flat_keep, mesh_y_flat_keep)
        # plt.show()


    return (X, y)


def scatter_results_directory(logdir, X, y, plot_sparta=True):

    vardict = {}
    for ii in range(X.shape[1]):
        vardict[f'x{ii+1}'] = X[:, ii]
    locals().update(vardict)

    if plot_sparta:
        with open(logdir + '/' + 'config.json', encoding='utf-8') as f:
            config = json.load(f)
        inputs = config['task']['dataset']['input']
        # find grad_u_T1
        if 'grad_u_T1' in inputs:
            grad_u_T1 = X[:,inputs.index('grad_u_T1')]
        else:
            dummy_config = config['task']
            dummy_config['dataset']['input'] = ['grad_u_T1']
            grad_u_T1, tmp = load_frozen_RANS_dataset(dummy_config)

        # find K
        if 'k' in inputs:
            k = X[:, inputs.index('k')]
        else:
            dummy_config = config['task']
            dummy_config['dataset']['input'] = ['k']
            k, tmp = load_frozen_RANS_dataset(dummy_config)
        Rsparta = 2*k*grad_u_T1*1.4

    for filename in os.listdir(logdir):
        if filename.endswith('hof.csv'):
            data = pd.read_csv(logdir + '/' + filename)
            expression = data['expression'][0]
            reward = data['r'][0]
            # ?? this can probably happen a lot neater in a separate function
            expression = expression.replace('exp', 'np.exp')
            expression = expression.replace('log', 'np.log')
            expression = expression.replace('sin', 'np.sin')
            expression = expression.replace('cos', 'np.cos')
            yhat = eval(expression)
            NRMSE = np.sqrt(np.mean((y-yhat)**2))/np.std(y)

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
            plt.savefig(logdir + '/' + filename[:-4])


if __name__ == '__main__':
    dsrpath = os.path.abspath(__file__)
    os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    logdir = '../logs_completed/log_2021-04-28-152005_kdeficit_10msamples'

    # fig = plt.figure(figsize=(15, 15), dpi=100)
    fig = plt.figure()
    tidy = []
    files = os.listdir(logdir)
    for filename in files:
        print(filename)
        if (filename[:3] == 'dsr') and (filename[-7:-4] != 'hof'):
            data = pd.read_csv(logdir + '/' + filename)
            plt.plot(data['base_r_best'], color='C0')
            tidy.append(data['base_r_best'].values)

    plt.show()

    tidylong = [array for array in tidy if len(array) > 5000]

    tidy_mean = np.mean(tidylong, axis=0)

    fig = plt.figure()
    plt.plot(tidy_mean)
    plt.show()

    #
    #
    #
    # # Load the config file
    # with open('config_bDelta.json', encoding='utf-8') as f:
    #     config = json.load(f)
    #
    # config_task = config["task"]      # Task specification parameters
    # config_training = config["training"]    # Training hyperparameters
    #
    # X, y = load_frozen_RANS_dataset(config_task)

    # ##########  Code below is used to make the scatterplot of the results if that has not happened in the loop.
    #
    # logdir = '../logs_completed/log_2021-04-28-152005_kdeficit_10msamples'
    #
    # # Load the config file
    # with open(logdir + '/config.json', encoding='utf-8') as f:
    #     config = json.load(f)
    #
    # # set required configs
    # config_task = config["task"]      # Task specification parameters
    # config_training = config["training"]    # Training hyperparameters
    #
    # X, y = load_frozen_RANS_dataset(config_task)
    #
    # scatter_results(logdir, X, y)
    #
    # print('end')




######### code below is used to create plots of reward-vs iterations


    logdir = 'log/Completed logs/log_2021-04-22-135835_tidycache_400'

    fig = plt.figure(figsize=(15, 15), dpi=100)
    tidy = []
    files = os.listdir(logdir)
    for filename in files:
        print(filename)
        if (filename[:3] == 'dsr') and (filename[-7:-4] != 'hof'):
            data = pd.read_csv(logdir + '/' + filename)
            plt.plot(data['base_r_best'], color='C0')
            tidy.append(data['base_r_best'].values)

    logdir = 'log/Completed logs/log_2021-04-21-214053_notidycache_400'
    notidy = []
    files = os.listdir(logdir)
    for filename in files:
        print(filename)
        if (filename[:3] == 'dsr') and (filename[-7:-4] != 'hof'):
            data = pd.read_csv(logdir + '/' + filename)
            plt.plot(data['base_r_best'], color='C1')
            notidy.append(data['base_r_best'].values)

    tidy = np.array(tidy)
    notidy = np.array(notidy)

    tidy_mean = np.mean(tidy, axis=0)
    notidy_mean = np.mean(notidy, axis=0)

    fig = plt.figure()
    plt.plot(tidy_mean)
    plt.plot(notidy_mean)
    plt.legend(['tidy', 'notidy'])

    print('end')






    # # these are the models from SPARTA
    # R = 2*k*grad_uTij1
    #
    # R1 = R*0.4
    # R2 = R*1.4
    # R3 = R*0.93
    #
    # inv_NRMSE1 = 1/(1+np.sqrt(sum((kdefecit - R1) ** 2)/kdefecit.shape[0])/np.std(kdefecit) )
    # inv_NRMSE2 = 1/(1+np.sqrt(sum((kdefecit - R2) ** 2)/kdefecit.shape[0])/np.std(kdefecit) )
    # inv_NRMSE3 = 1/(1+np.sqrt(sum((kdefecit - R3) ** 2)/kdefecit.shape[0])/np.std(kdefecit) )
    #

    #
    # # Best models from dsr log_2021-03-24-165929
    #
    #
    #
    #
    #
    #
    # R = 0.077443009021649077*grad_uTij1
    #
    # fig = plt.figure()
    # plt.scatter(kdefecit, R)
    # plt.xlabel('kDefecit')
    # plt.ylabel('DSR best model')
    # plt.grid('both')
    # plt.show
    #
    #
    # #plt.close('all')
