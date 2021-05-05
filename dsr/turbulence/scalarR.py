# Jasper Hemmes - 2021

import os
import sys
import pickle
import numpy as np


def calc_sij_rij(grad_u, omega):
    """ Calculates the strain rate and rotation rate tensors.  Normalizes by omega:
    Sij = k/eps * 0.5* (grad_u  + grad_u^T) = 1/omega * 0.5* (grad_u  + grad_u^T)
    Rij = k/eps * 0.5* (grad_u  - grad_u^T) = 1/omega * 0.5* (grad_u  - grad_u^T)
    :param grad_u: velocity gradient (3, 3, num_of_points)
    :return: Sij, Rij (3, 3, num_of_points)
    """
    omega = np.maximum(omega, 1e-8)

    Sij = 0.5*(grad_u + np.transpose(grad_u, (1,0,2)))/omega
    Rij = 0.5*(grad_u - np.transpose(grad_u, (1,0,2)))/omega

    # ?? do i need this?
    # Sij = Sij - np.repeat(np.eye(3), omega.shape[0], axis=1).reshape(3, 3, omega.shape[0]) * np.trace(Sij)/3

    return Sij, Rij

def calc_tensor_basis(Sij, Rij):
    """ Calculates the integrity basis of the base tensor expansion.

    :param Sij: Mean strain-rate tensor (3, 3, num_of_points)
    :param Rij: Mean rotation-rate tensor (3, 3, num_of_points)
    :return: T: Base tensor series (10, 3, 3, num_of_points)
    """
    # ?? looking to replace this for loop. However it only happens once so does it matter that much?
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


if __name__ == '__main__':
    dsrpath = os.path.abspath(__file__)
    os.chdir(dsrpath[:dsrpath.find('/dsr/dsr/')+8]) # change the working directory to main dsr dir with the config files
    sys.path.append(dsrpath[:dsrpath.find('/dsr/dsr/')+4]) # add dsr directory to path to make sure imports work.

    from dsr import DeepSymbolicRegressor

    # Generate some data
    np.random.seed(0)
    X = np.random.random((10, 6))
    y = np.sin(X[:, 1]) + 3*X[:, 2] ** 2
    # y = np.sin(X[:, 0]) + 0.756*X[:, 1] ** 2 + 2*X[:, 2] + X[:, 3] + np.exp(X[:, 4]) + X[:, 5]*X[:, 6]  + np.cos(X[:, 7]+X[:, 8])

    dataset = np.zeros([X.shape[0], X.shape[1]+1])
    dataset[:,:-1] = X
    dataset[:,-1] = y
    np.savetxt('turbulence/dsr_datasets/test.csv', dataset, delimiter=',')

    # Create the model
    model = DeepSymbolicRegressor("custom_config.json")

    # Fit the model
    model.fit(X, y)  # Should solve in ~10 seconds

    # View the best expression
    print(model.program_.pretty())

    # Make predictions
    model.predict(2 * X)



    ############################# Load and prepare frozen rans data ####################################

    case = 'PH10595'

    frozen = pickle.load(open(f'turbulence/frozen_data/{case}_frozen_var.p', 'rb'))
    data_i = frozen['data_i']

    # calculate strain rate tensors:

    Sij, Rij = calc_sij_rij(data_i['grad_u'], data_i['omega_frozen'])
    Tij = calc_tensor_basis(Sij, Rij)

    k = data_i['k']
    grad_u = data_i['grad_u']

    n_cells = k.shape[0]
    grad_uTij1  = np.zeros(n_cells) # tensorproduct of Tij1 and grad_u
    # grad_uTij2  = np.zeros(num_of_cells) # tensorproduct of Tij1 and grad_u
    # grad_uTij3  = np.zeros(num_of_cells) # tensorproduct of Tij1 and grad_u
    # grad_uTij4  = np.zeros(num_of_cells) # tensorproduct of Tij1 and grad_u

    for ii in range(n_cells):
        grad_uTij1[ii] = np.tensordot(grad_u[:,:,ii], Tij[0,:,:,ii]) # Tij1 = Sij

    #2nd dimension is the amount of variables to be passed to dsr
    X = np.zeros([n_cells, 2])

    X[:, 0] = k
    X[:, 1] = grad_uTij1

    y = data_i['kDeficit']



    ################################## Initialise the model ############################################

    model = DeepSymbolicRegressor("custom_config.json")

    # Fit the model
    model.fit(X, y)  # Should solve in ~10 seconds

    # View the best expression
    print(model.program_.pretty())

    # Make predictions
    model.predict(2 * X)


