import os
import platform
import numpy as np

from resultprocessing import load_iterations

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt





if __name__ == "__main__":

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    logdir = '../logs_completed/report_plotting'

    dirlist = os.listdir(logdir)


    iterations = np.arange(10, 110, 10)

    metrics = {}

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


        for nit in iterations:
            metrics[dir]['mean'].append(np.mean(rewards_sorted[:nit]))
            metrics[dir]['std'].append(np.std(rewards_sorted[:nit]))
            metrics[dir]['max'].append(np.max(rewards_sorted[:nit]))

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
    print('end')
    print('end')