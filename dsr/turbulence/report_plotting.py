import os
import platform
import numpy as np

from resultprocessing import load_iterations

import matplotlib
matplotlib.use('tkagg')
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



if __name__ == "__main__":

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files


    # create_plots_for_increasing_n_iterations()


    print('end')
    print('end')