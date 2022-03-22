# Jasper Hemmes - 2021

import os
import sys
import shutil
import pickle
import numpy as np
import json
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import time
# from dsr.turbulence.dataprocessing import load_frozen_RANS_dataset, de_flatten_tensor
# from dsr.program import from_str_tokens
# from dsr.turbulence.lumley_plots import plot_lumley_comparison
# import copy





def load_iterations(logdir):

    return_dict = {}
    for filename in os.listdir(logdir):
        split = filename.split('_')[-1].split('.')
        if split[0] == 'stats' and split[1] == 'csv':
            df_append = pd.read_csv(f'{logdir}/{filename}', header=None)
            return_dict[filename] = df_append

    return return_dict

if __name__ == "__main__":
    dsrpath = os.path.abspath(__file__)
    os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    logdir = '../logs_completed/log_2022-03-20-182735_optimize_statistics'
    optim_stats = load_iterations(logdir)


    batches_match = np.zeros((optim_stats[list(optim_stats.keys())[0]].shape[0], optim_stats[list(optim_stats.keys())[0]].shape[1] - 1, len(optim_stats)))

    #
    # all_match_bestperformer = []
    # for key in optim_stats:
    #     for value in optim_stats[key].iloc[:,0].values:
    #         all_match_bestperformer.append(value)
    #
    #
    bins = np.arange(0, 2000, 10)
    #
    # plt.figure()
    # plt.hist(x=all_match_bestperformer, bins=bins, density=True)
    # plt.xlim([0,200])
    # plt.show()
    #
    #
    # all_match_bestperformer = np.array(all_match_bestperformer)
    # np.mean(all_match_bestperformer < 100)


    for ii in range(len(optim_stats)):
        key = list(optim_stats.keys())[ii]
        batches_match[:,:,ii] = optim_stats[key].values[:,1:]

    mean_batch_match = np.mean(batches_match, axis=-1)

    plt.figure()
    for ii in np.arange(0, mean_batch_match.shape[0], 50):
        plt.plot(bins, mean_batch_match[ii, :], label=f'{ii}')
    plt.xlim([0, 200])
    plt.legend()
    plt.grid('both')


    print('end')




