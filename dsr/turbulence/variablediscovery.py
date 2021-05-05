# Jasper Hemmes - 2021

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    # dsrpath = os.path.abspath( __file__)  # these two lines are to add the dsr dir to path to run it without installing dsr package
    # sys.path.append(dsrpath[:dsrpath.rfind('/')])







    case = 'PH10595'
    # case = 'CD12600'
    # case = 'CBFS13700'


    frozen = pickle.load(open(f'frozen_data/{case}_frozen_var.p', 'rb'))
    data_i = frozen['data_i']

    # calculate strain rate tensors:

    Sij, Rij = calc_sij_rij(data_i['grad_u'], data_i['omega_frozen'])


    # 'um',                  - (15600,)     -
    # 'vm',                  - (15600,)     -
    # 'wm',                  - (15600,)     -
    # 'pm',                  - (15600,)     -
    # 'uu',                  - (15600,)     -
    # 'vv',                  - (15600,)     -
    # 'ww',                  - (15600,)     -
    # 'uv',                  - (15600,)     -
    # 'meshRANS',            -              -
    # 'hifi-data',           -              -
    # 'uw',                  - (15600,)     -
    # 'vw',                  - (15600,)     -
    # 'k',                   - (15600,)     - This vector contains slightly more decimals than k_val, values are the same
    # 'tauij',               - (3,3,15600)  -
    # 'aij',                 - (3,3,15600)  -
    # 'bij',                 - (3,3,15600)  - Is this the ground truth bij from LES?
    # 'omega_frozen',        - (15600,)     -
    # 'nut_frozen',          - (15600,)     -
    # 'k_val',               - (15600,)     - This vector contains slightly less decimals than k_val, values are the same
    # 'grad_u',              - (3,3,15600)  -
    # 'aDelta',              - (3,3,15600)  - ?? Where does this originate? aDelta*2*k does not result in bDelta
    # 'aBoussinesq',         - (3,3,15600)  - ?? Where does this originate? you would think: bij_boussinesq = -nut_frozen/k*Sij, and then aBoussinesq = bij_boussinesq/(2*k). However that is not the case.
    # 'tauij_valid',         - (3,3,15600)  - tauij with less decimals
    # 'bDelta',              - (3,3,15600)  - ?? this is not the same as aDelta*2*k. However: bij =-nut_frozen/k*Sij + bDelta is very close, not exactly the same, but close.
    # 'Pk',                  - (15600,)     - Pk = aij*grad_u (check minus signs)
    # 'Pk_Boussinesq',       - (15600,)     - Pk_boussinesq = aBoussinesq * grad_u (MINUS SIGN NOT RIGHT)
    # 'Pk_delta',            - (15600,)     - Pk_delta = aDelta*grad_u
    # 'kDeficit',            - (15600,)     - I expect this is the R term.
    # 'timescale_limited',   - (15600,)     -
    # 'd'])                  - (15600,)     -


    aDelta = data_i['aDelta']
    bDelta = data_i['bDelta']
    k = data_i['k']
    Pk_delta = data_i['Pk_delta']
    aij = data_i['aij']
    bij = data_i['bij']
    grad_u = data_i['grad_u']
    tauij = data_i['tauij']
    nut_frozen = data_i['nut_frozen']
    kdefecit = data_i['kDeficit']
    tauij_valid = data_i['tauij_valid']
    Pk = data_i['Pk']
    Pk_delta = data_i['Pk_delta']
    Pk_Boussinesq = data_i['Pk_Boussinesq']
    aBoussinesq = data_i['aBoussinesq']




    bij_boussinesq = -nut_frozen/k*Sij
    aBoussinesqtest = bij_boussinesq/(2*k)

    pktest = np.zeros(k.shape[0])
    pkdeltatest = np.zeros(k.shape[0])
    pkbousstest = np.zeros(k.shape[0])
    grad_uTij1  = np.zeros(k.shape[0])
    bDeltatest = aDelta*2*k

    for ii in range(k.shape[0]):
        # pktest[ii] = np.tensordot(aij[:,:,ii], grad_u[:,:,ii])
        # pkdeltatest[ii] = np.tensordot(aDelta[:,:,ii], grad_u[:,:,ii])
        # pkbousstest[ii] = np.tensordot(aBoussinesq[:,:,ii], grad_u[:,:,ii])
        grad_uTij1[ii] = np.tensordot(grad_u[:,:,ii], Sij[:,:,ii]) # Tij1 = Sij




    # these are the models from SPARTA
    R = 2*k*grad_uTij1

    R1 = R*0.4
    R2 = R*1.4
    R3 = R*0.93


    NRMSE1 = np.sqrt(sum((kdefecit - R1) ** 2)/kdefecit.shape[0])/np.std(kdefecit)
    NRMSE2 = np.sqrt(sum((kdefecit - R2) ** 2)/kdefecit.shape[0])/np.std(kdefecit)
    NRMSE3 = np.sqrt(sum((kdefecit - R3) ** 2)/kdefecit.shape[0])/np.std(kdefecit)

    inv_NRMSE1 = 1/(1+NRMSE1)
    inv_NRMSE2 = 1/(1+NRMSE2)
    inv_NRMSE3 = 1/(1+NRMSE3)


    fig = plt.figure()
    plt.scatter(kdefecit, R2)
    plt.xlabel('kDefecit')
    plt.ylabel('Sparta R model 2')
    plt.grid('both')
    plt.show

    # Best models from dsr log_2021-03-24-165929

    logdir =  'log_2021-03-24-165929'




    R = 0.077443009021649077*grad_uTij1

    NRMSE = np.sqrt(sum((kdefecit - R) ** 2) / kdefecit.shape[0]) / np.std(kdefecit)
    inv_NRMSE = 1 / (1 + NRMSE)

    fig = plt.figure()
    plt.scatter(kdefecit, R)
    plt.xlabel('kDefecit')
    plt.ylabel('DSR best model')
    plt.grid('both')
    plt.show


    #plt.close('all')




