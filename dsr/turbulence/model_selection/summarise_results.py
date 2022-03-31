
import os
import platform
import json
import numpy as np

from dsr.turbulence.resultprocessing import load_iterations


def fetch_iteration_metrics(logdir, finished=True):

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



def summarise_results(logdir):

    dirlist = os.listdir(logdir)


    dirlist.remove('config_baseline.json')
    # ratios used to scale duration
    #
    #
    # try:  # remove existing results file since new will be created.
    #     os.remove(os.path.join(logdir, 'results', 'results.csv'))
    # except FileNotFoundError:
    #     pass
    # parameters = []

    all_results = {}
    for run in dirlist:
        run_dir = os.path.join(logdir, run)

        # with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
        #     config_run = json.load(f)
        #
        # machine_name = config_run['task']['name'].split('_')[0]
        # diff = compare_dicts(config_bsl, config_run)
        # run_name = machine_name
        # if diff[0] == 'baseline':
        #     run_name += '_baseline'
        #     baseline = run_name
        #     parameters.append('baseline')
        # else:
        #     for item in diff:
        #         run_name += f'_{item[0]}_{item[1]}'
        #         if item[0] not in parameters:
        #             parameters.append(item[0])
#
#
#         run_dict = fetch_iteration_metrics(os.path.join(logdir, run), finished=False)
#
#         result_col = ['run_name']
#         result_val = [run_name]
#
#         tmp_arr = np.array(run_dict['proc_time'])
#         result_col.append('adjusted_avg_proc_duration')
#         result_val.append(round(np.mean(np.sum(tmp_arr, axis=1)) * machine_dur_ratios[machine_name]))
#
#         save_dict = {}
#
#         for key in run_dict:
#             tmp_arr = np.array(run_dict[key])
#             save_dict[key] = {}
#             save_dict[key]['mean'] = np.mean(tmp_arr, axis=0)
#             save_dict[key]['std'] = np.std(tmp_arr, axis=0)
#             save_dict[key]['max'] = np.max(tmp_arr, axis=0)
#             tmp_arr = np.sort(tmp_arr, axis=0)
#             save_dict[key]['5max'] = np.mean(tmp_arr[-5:, :], axis=0)
#             result_col.extend(['_'.join([mode, key]) for mode in save_dict[key].keys()])
#             result_val.extend([save_dict[key][mode][-1] for mode in save_dict[key].keys()])
#
#         # write results to csv:
#         df_append = pd.DataFrame([result_val], columns=result_col)
#         df_append.to_csv(os.path.join(logdir, 'results', 'results.csv'), mode='a', header=first_write, index=False)
#
#         if first_write:  # used to only write header once
#             first_write = False
#
#         all_results[run_name] = save_dict
#
#     for key in all_results:
#         all_results[key]['varied'] = []
#         for parameter in parameters:
#             if parameter in key:
#                 all_results[key]['varied'].append(parameter)
#
#     plot_dict = {key: [baseline] if baseline else [] for key in parameters}
#     plot_dict['baseline'] = []
#     plot_dict['all'] = all_results.keys()
#
#     # if logdir.split('_')[-1] == 'kDeficit':
#     #     plot_dict['compare'] = ['OW_baseline',
#     #                             'M18_initializer_uniform_learning_rate_0.01',
#     #                             'OW_initializer_normal_learning_rate_0.01',
#     #                             'OW_learning_rate_0.01',
#     #                             'M18_num_units_128_initializer_normal_learning_rate_0.01',
#     #                             'OW_num_units_256_initializer_normal_learning_rate_0.01',
#     #                             'OW_num_units_256',
#     #                             'OW_entropy_weight_0.0025',
#     #                             'M3_initializer_normal']
#     # else:
#     #     plot_dict['compare'] = ['OW_baseline',
#     #                             'M15_learning_rate_0.01',
#     #                             'M18_learning_rate_0.01',
#     #                             'OW_initializer_normal_learning_rate_0.01',
#     #                             'M3_num_units_64_initializer_normal_learning_rate_0.01',
#     #                             'OW_num_units_256']
#
#     for parameter in parameters:
#         for run in all_results:
#             if parameter in all_results[run]['varied']: # and len(all_results[run]['varied']) == 1:
#                 plot_dict[parameter].append(run)
#
#     for key in plot_dict:
#         plot_dir = os.path.join(logdir, 'results', key)
#         os.makedirs(plot_dir, exist_ok=True)
#         create_plots(all_results, plotmode='mean', plotlist=plot_dict[key], plot_dir=plot_dir)
#         create_plots(all_results, plotmode='max', plotlist=plot_dict[key], plot_dir=plot_dir)
#         create_plots(all_results, plotmode='5max', plotlist=plot_dict[key], plot_dir=plot_dir)
#
# def create_plots(all_results, plotmode, plotlist, plot_dir):
#
#     for metric in all_results[list(all_results.keys())[0]]:
#         if metric == 'varied':
#             pass
#         else:
#             plt.figure(figsize=(12,10))
#             for run in plotlist:
#                 plt.plot(all_results[run][metric][plotmode])
#             plt.xlabel('iterations')
#             plt.ylabel(' '.join([plotmode, metric]))
#             plt.legend(plotlist)
#             plt.grid('both')
#             plt.savefig(f'{plot_dir}/{metric}_{plotmode}.png')
#             plt.close('all')
#

if __name__ == "__main__":

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
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
    logdir = '../logs_completed/compare_iterlim_optimisation'
    summarise_results(logdir)

    print('end')




