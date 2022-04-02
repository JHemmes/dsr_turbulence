import os
import sys

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import fluidfoam




def read_case_results(case_dir):
    dirlist_case = os.listdir(case_dir)

    # try to find name
    if 'name' in dirlist_case:
        with open(os.path.join(case_dir, 'name')) as f:
            name = f.readline()
        name = name[:-1]
    else:
        name = case_dir

    # get last written solution:
    sol_dirs = [int(sol) for sol in dirlist_case if sol.isnumeric()]
    last_sol_dir = f'{max(sol_dirs)}'

    sol_files = [f for f in os.listdir(os.path.join(case_dir,last_sol_dir)) if os.path.isfile(os.path.join(case_dir,last_sol_dir, f))]

    vectors = ['U', 'U_LES']
    scalars = ['k', 'k_LES', 'omega', 'nut', 'p', 'phi']

    results = {}
    for file in sol_files:
        if file in scalars:
            results[file] = fluidfoam.readscalar(case_dir, last_sol_dir, file)
        if file in vectors:
            results[file] = fluidfoam.readvector(case_dir, last_sol_dir, file)

    # get line_results from postprocessing dir:
    pp_dir = os.path.join(case_dir, 'postProcessing')
    pp_dirlist = os.listdir(pp_dir)

    results['pp'] = {}
    for d in pp_dirlist:
        if d == 'residuals':
            try:
                data = np.genfromtxt(os.path.join(pp_dir, d, '0', 'residuals_1.dat'))
            except OSError:
                data = np.genfromtxt(os.path.join(pp_dir, d, '0', 'residuals.dat'))
            results['pp'][d] = data
        else:
            results['pp'][d] = {}
            pp_files = os.listdir(os.path.join(pp_dir, d, last_sol_dir))
            for file in pp_files:
                data = np.genfromtxt(os.path.join(pp_dir, d, last_sol_dir, file))
                results['pp'][d][file.split('.')[0]] = data

    return results, name

def reshape_to_mesh(array):
    if len(array) == 120*130:
        return np.reshape(array, (120, 130), order='F')
    elif len(array) == 140*100:
        return np.reshape(array, (140, 100), order='F')
    elif len(array) == 140*150:
        return np.reshape(array, (140, 150), order='F')
    else:
        print('Unknown mesh')

def read_and_plot_PH():
    # change the working directory to dir with the openFOAM results
    os.chdir('/home/jasper/OpenFOAM/jasper-7/PeriodicHill-SpaRTA')

    dirlist = os.listdir('.')
    dirlist = [d for d in dirlist if '00Baseline' in d]

    dirlist.insert(0, '01Frozen')

    results = {}
    for case in dirlist:
        result, name = read_case_results(case)
        results[name] = result

    # mse?
    # calc error of k_omega and LES
    u_les = results['frozen_w_LES']['U_LES'][:-1, :]
    u_komg = results['baseline_kOmegaSST']['U'][:-1, :]
    mse_komg = sum(sum((u_les - u_komg)**2))/(u_les.shape[0]*u_les.shape[1])

    # for each DSR run calc error between case and LES and divide by error of standard kOmegaSST
    dsr_runs = list(results.keys())
    dsr_runs.pop(dsr_runs.index('frozen_w_LES'))
    dsr_runs.pop(dsr_runs.index('baseline_kOmegaSST'))

    best_mse_DSR = 1000
    best_mse_sparta = 1000

    for run in dsr_runs:
        u = results[run]['U'][:-1, :]
        mse = sum(sum((u_les - u) ** 2)) / (u_les.shape[0] * u_les.shape[1])
        norm_mse = mse/mse_komg
        results[run]['norm_mse'] = norm_mse
        if norm_mse < best_mse_DSR:
            if 'DSR' in run:
                best_dsr = run
                best_mse_DSR = norm_mse
        if norm_mse < best_mse_sparta:
            if 'Sparta' in run:
                best_sparta = run
                best_mse_sparta = norm_mse

    print(f'Best DSR run with normalised mse {best_mse_DSR} is {best_dsr}')
    print(f'Best sparta run with normalised mse {best_mse_sparta} is {best_sparta}')

    mesh_x_flat, mesh_y_flat, mesh_z_flat = fluidfoam.readof.readmesh('00Baseline_kOmegaSST')
    mesh_x = reshape_to_mesh(mesh_x_flat)
    n_points = mesh_x.shape[0]

    plt.figure()
    plt.plot(mesh_x_flat[:n_points], mesh_y_flat[:n_points], c='Black')
    plt.plot(mesh_x_flat[-n_points:], mesh_y_flat[-n_points:], c='Black')
    plt.plot([mesh_x_flat[0], mesh_x_flat[-n_points]], [mesh_y_flat[0], mesh_y_flat[-n_points]], c='Black')
    plt.plot([mesh_x_flat[n_points - 1], mesh_x_flat[-1]], [mesh_y_flat[n_points - 1], mesh_y_flat[-1]], c='Black')

    ax = plt.gca()
    ax.set_aspect('equal')

    # add LES results:
    u_scale = 1

    label = 'LES'
    for line in results['frozen_w_LES']['pp']:
        if 'single' in line:
            data = results['frozen_w_LES']['pp'][line]['line_U_LES']
            plt.plot(data[:, 0] + u_scale*data[:, 3], data[:, 1], c='Black', marker='o', markevery=5, label=label)
            if label:
                label = None

    label = r'$k-\omega$ SST'
    # add baseline kOmegaSST results:
    for line in results['baseline_kOmegaSST']['pp']:
        if 'single' in line:
            data = results['baseline_kOmegaSST']['pp'][line]['line_U']
            plt.plot(data[:, 0] + u_scale*data[:, 3], data[:, 1], c='C0', linestyle=':', markevery=5, label=label)
            if label:
                label = None

    # add spaRTA results:
    label = 'SpaRTA'
    for line in results[best_sparta]['pp']:
        if 'single' in line:
            data = results[best_sparta]['pp'][line]['line_U']
            plt.plot(data[:, 0] + u_scale*data[:, 3], data[:, 1], c='C1', linestyle='--', markevery=5, label=label)
            if label:
                label = None
    # add DSR results:

    label = 'DSR'
    for line in results[best_dsr]['pp']:
        if 'single' in line:
            data = results[best_dsr]['pp'][line]['line_U']
            plt.plot(data[:, 0] + u_scale*data[:, 3], data[:, 1], c='C2', linestyle='-', markevery=5, label=label)
            if label:
                label = None

    plt.legend()
    # # add DSR results:
    # for line in results['DSR:0.07654184023911788*grad_u_T1*k/inv1']['pp']:
    #     if 'single' in line:
    #         data = results['DSR:0.07654184023911788*grad_u_T1*k/inv1']['pp'][line]['line_U']
    #         plt.plot(data[:, 0] + u_scale*data[:, 3], data[:, 1], c='C2', linestyle='-', markevery=5)
    #
    # exclude = 15
    # plt.scatter(mesh_x_flat[:exclude*n_points], mesh_y_flat[:exclude*n_points], c='Black')
    # plt.scatter(mesh_x_flat[-exclude*n_points:], mesh_y_flat[-exclude*n_points:], c='Black')


if __name__ == '__main__':

    read_and_plot_PH()



    print('end')
    print('end')
