import os
import sys

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import fluidfoam
import scipy.interpolate as interp

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

def interpolate_CBFS():
    # needs work to avoid interpolating outside flow domain.
    # create line from lower and upper boundaries (extract the one row of points)
    # then interpolate the x station of lines to that lower and upper edge,
    # Then create linspace using those values as limits.

    # ["x/H", "y/H", "p", "u/U_in", "v/U_in", "w/U_in", "uu/U_in^2", "vv/U_in^2", "ww/U_in^2", "uv/U_in^2", "uw/U_in^2", "vw/U_in^2", "k/U_in^2"]
    with open('/home/jasper/Documents/afstuderen/python/inversion/DATA/CBFS-Bentaleb/data/curvedbackstep_vel_stress.dat') as f:
        data = f.readlines()

    data_numbers = []
    for line in data:
        line = line[:-2]
        line = line.split(' ')
        line.pop(0)
        if len(line) > 2:
            try:
                data_numbers.append([eval(val) for val in line])
            except:
                pass

    data = np.array(data_numbers)

    mesh_x = np.reshape(data[:, 0], (768, 160), order='F')
    mesh_y = np.reshape(data[:, 1], (768, 160), order='F')

    top_x = mesh_x[:, -1]
    top_y = mesh_y[:, -1]
    ftop = interp.interp1d(top_x, top_y)

    bot_x = mesh_x[:, 0]
    bot_y = mesh_y[:, 0]
    fbot = interp.interp1d(bot_x, bot_y)

    x_stations = np.arange(0, 11)
    x_stations[0] = 1e-6  # this is also the case in the openFoam postprocessing

    y_top = ftop(x_stations)
    y_bot = fbot(x_stations)

    n_points = 150
    mesh_x_target = []
    mesh_y_target = []

    for ii in range(len(x_stations)):
        mesh_x_target.append(x_stations[ii] * np.ones(n_points))
        beta = np.linspace(0, np.pi, n_points)
        mesh_y_target.append(y_bot[ii] + (0.5*(1-np.cos(beta))) * (y_top[ii] - y_bot[ii]))
        # mesh_y_target.append(np.linspace(y_bot[ii], y_top[ii], n_points))

    mesh_x_target = np.moveaxis(np.array(mesh_x_target), -1, 0)
    mesh_y_target = np.moveaxis(np.array(mesh_y_target), -1, 0)

    x_flat = mesh_x_target.flatten()
    y_flat = mesh_y_target.flatten()

    U = interp.griddata((data[:, 0], data[:, 1]),
                         data[:, 3], (x_flat, y_flat), method='linear')

    uinterpolated = np.reshape(U, mesh_x_target.shape, order='A')

    V = interp.griddata((data[:, 0], data[:, 1]),
                         data[:, 4], (x_flat, y_flat), method='linear')

    vinterpolated = np.reshape(V, mesh_x_target.shape, order='A')

    # plt.figure()
    # plt.contourf(mesh_x_target, mesh_y_target, uinterpolated, levels=30, cmap='Reds')
    # plt.scatter(mesh_x_target, mesh_y_target)

    to_save = np.zeros((x_flat.shape[0], 4))
    to_save[:,0] = x_flat
    to_save[:,1] = y_flat
    to_save[:,2] = U
    to_save[:,3] = V

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/CBFS/common/LES_interpolated_lines.csv', to_save, delimiter=',')

    # also interpolate to full mesh:
    mesh_x_flat, mesh_y_flat, mesh_z_flat = fluidfoam.readof.readmesh('/home/jasper/OpenFOAM/jasper-7/run/CBFS/CBFS_simpleFoam_kOmegaSST/')

    u_full_field = interp.griddata((data[:, 0], data[:, 1]),
                                    data[:, 3], (mesh_x_flat, mesh_y_flat), method='linear')

    v_full_field = interp.griddata((data[:, 0], data[:, 1]),
                                    data[:, 4], (mesh_x_flat, mesh_y_flat), method='linear')

    to_save = np.zeros((mesh_x_flat.shape[0], 4))
    to_save[:,0] = mesh_x_flat
    to_save[:,1] = mesh_y_flat
    to_save[:,2] = u_full_field
    to_save[:,3] = v_full_field

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/CBFS/common/LES_interpolated_field.csv', to_save, delimiter=',')

    # mesh_x_test = reshape_to_mesh(mesh_x_flat)
    # mesh_y_test = reshape_to_mesh(mesh_y_flat)
    # u_test = reshape_to_mesh(u_full_field)
    #
    # plt.contourf(mesh_x_test, mesh_y_test, u_test, levels=30, cmap='Reds')

def interpolate_PH():

    #
    # # # no interpolation for full field, just rewrite frozen data into right format!
    # #
    # frozen_path = '/home/jasper/OpenFOAM/jasper-7/run/PH/common/01Frozen'
    # data, name = read_case_results(frozen_path)
    # #
    # mesh_x_flat, mesh_y_flat, mesh_z_flat = fluidfoam.readof.readmesh(frozen_path)
    # #
    # mesh_x = reshape_to_mesh(mesh_x_flat)
    # mesh_y = reshape_to_mesh(mesh_y_flat)
    # #
    # u_les = data['U_LES'][0, :]
    # v_les = data['U_LES'][1, :]
    # #
    # to_save = np.zeros((mesh_x_flat.shape[0], 4))
    # to_save[:,0] = mesh_x_flat
    # to_save[:,1] = mesh_y_flat
    # to_save[:,2] = u_les
    # to_save[:,3] = v_les
    # #
    # np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/PH/common/LES_interpolated_field.csv', to_save, delimiter=',')
    #
    # list_x = []
    # list_y = []
    # list_u = []
    # list_v = []
    #
    #
    # for ii in range(9):
    #     key = f'singleGraph_x{int(ii)}'
    #     list_x.append(data['pp'][key]['line_U_LES'][:, 0])
    #     list_y.append(data['pp'][key]['line_U_LES'][:, 1])
    #     list_u.append(data['pp'][key]['line_U_LES'][:, 3])
    #     list_v.append(data['pp'][key]['line_U_LES'][:, 4])
    #
    # x_arr = np.concatenate(list_x)
    # y_arr = np.concatenate(list_y)
    # u_arr = np.concatenate(list_u)
    # v_arr = np.concatenate(list_v)
    #
    #
    # to_save = np.zeros((x_arr.shape[0], 4))
    # to_save[:, 0] = x_arr
    # to_save[:, 1] = y_arr
    # to_save[:, 2] = u_arr
    # to_save[:, 3] = v_arr
    # np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/PH/common/LES_interpolated_lines.csv', to_save, delimiter=',')
    #
    # new interpolated data
    path = '/home/jasper/Documents/afstuderen/python/inversion/DATA/PH-Breuer/data/Re_10595/Hill_Breuer.csv'
    data = np.genfromtxt(path, delimiter=',')
    data = data[1:, :-1]
    shape = (281, 234)
    shape = (234, 281)
    mesh_x = np.reshape(data[:, 0], shape, order='A').T
    mesh_y = np.reshape(data[:, 1], shape, order='A').T
    # mesh_u = np.reshape(data[:, 2], shape, order='A').T
    # mesh_v = np.reshape(data[:, 3], shape, order='A').T

    bot_x = mesh_x[:, 0]
    bot_y = mesh_y[:, 0]

    top_x = mesh_x[:, -1]
    top_y = np.max(mesh_y, axis=1) # hack because the mesh is in strange shape

    ftop = interp.interp1d(top_x, top_y)
    fbot = interp.interp1d(bot_x, bot_y)

    x_stations = np.arange(0, 9)
    x_stations[0] = 1e-6  # this is also the case in the openFoam postprocessing

    y_top = ftop(x_stations)
    y_bot = fbot(x_stations)

    n_points = 150
    mesh_x_target = []
    mesh_y_target = []

    for ii in range(len(x_stations)):
        mesh_x_target.append(x_stations[ii] * np.ones(n_points))
        # mesh_y_target.append(np.linspace(y_bot[ii], y_top[ii], n_points))
        beta = np.linspace(0, np.pi, n_points)
        mesh_y_target.append(y_bot[ii] + (0.5*(1-np.cos(beta))) * (y_top[ii] - y_bot[ii]))



    mesh_x_target = np.moveaxis(np.array(mesh_x_target), -1, 0)
    mesh_y_target = np.moveaxis(np.array(mesh_y_target), -1, 0)

    x_flat = mesh_x_target.flatten()
    y_flat = mesh_y_target.flatten()

    u_lines = interp.griddata((data[:, 0], data[:, 1]), data[:, 2], (x_flat, y_flat), method='linear')

    v_lines = interp.griddata((data[:, 0], data[:, 1]), data[:, 3], (x_flat, y_flat), method='linear')

    to_save = np.zeros((x_flat.shape[0], 4))
    to_save[:, 0] = x_flat
    to_save[:, 1] = y_flat
    to_save[:, 2] = u_lines
    to_save[:, 3] = v_lines

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/PH/common/LES_interpolated_lines.csv', to_save, delimiter=',')

    # also interpolate to full mesh:
    mesh_x_flat, mesh_y_flat, mesh_z_flat = fluidfoam.readof.readmesh('/home/jasper/OpenFOAM/jasper-7/run/PH/PH_simplefoam_kOmegaSST/')

    u_field = interp.griddata((data[:, 0], data[:, 1]), data[:, 2], (mesh_x_flat, mesh_y_flat), method='linear')

    v_field = interp.griddata((data[:, 0], data[:, 1]), data[:, 3], (mesh_x_flat, mesh_y_flat), method='linear')

    to_save = np.zeros((mesh_x_flat.shape[0], 4))
    to_save[:, 0] = mesh_x_flat
    to_save[:, 1] = mesh_y_flat
    to_save[:, 2] = u_field
    to_save[:, 3] = v_field

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/PH/common/LES_interpolated_field.csv', to_save, delimiter=',')

def interpolate_CD():
    # needs work to avoid interpolating outside flow domain.
    # create line from lower and upper boundaries (extract the one row of points)
    # then interpolate the x station of lines to that lower and upper edge,
    # Then create linspace using those values as limits.

    with open('/home/jasper/Documents/afstuderen/python/inversion/DATA/CDC-Laval/data/conv-div-mean.dat' ) as f:
        data = f.readlines()

    cols = ["X", "Y", "mean_u", "mean_v", "mean_w", "dx_mean_u", "dx_mean_v", "dx_mean_w", "dy_mean_u", "dy_mean_v",
            "dy_mean_w", "dz_mean_u", "dz_mean_v", "dz_mean_w", "reynolds_stress_uu", "reynolds_stress_uv",
            "reynolds_stress_uw", "reynolds_stress_vv", "reynolds_stress_vw", "reynolds_stress_ww"]

    data_numbers = []
    for line in data:
        line = line[:-2]
        line = line.split(' ')
        line.pop(0)
        if len(line) > 2:
            try:
                data_numbers.append([eval(val) for val in line])
            except:
                pass

    data = np.array(data_numbers)
    shape = (2304, 385)
    mesh_x = np.reshape(data[:, 0], shape, order='A')
    mesh_y = np.reshape(data[:, 1], shape, order='A')

    top_x = mesh_x[:, -1]
    top_y = mesh_y[:, -1]
    ftop = interp.interp1d(top_x, top_y)

    bot_x = mesh_x[:, 0]
    bot_y = mesh_y[:, 0]
    fbot = interp.interp1d(bot_x, bot_y)

    x_stations = np.arange(0, 13)
    x_stations[0] = 1e-6  # this is also the case in the openFoam postprocessing

    y_top = ftop(x_stations)
    y_bot = fbot(x_stations)

    n_points = 150
    mesh_x_target = []
    mesh_y_target = []

    for ii in range(len(x_stations)):
        mesh_x_target.append(x_stations[ii] * np.ones(n_points))
        beta = np.linspace(0, np.pi, n_points)
        mesh_y_target.append(y_bot[ii] + (0.5*(1-np.cos(beta))) * (y_top[ii] - y_bot[ii]))
        # mesh_y_target.append(np.linspace(y_bot[ii], y_top[ii], n_points))

    mesh_x_target = np.moveaxis(np.array(mesh_x_target), -1, 0)
    mesh_y_target = np.moveaxis(np.array(mesh_y_target), -1, 0)

    # plt.scatter(mesh_x_target, mesh_y_target)

    uu = np.reshape(data[:, 2], shape, order='A')
    plt.figure()
    plt.title('U')
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, uu, levels=30, cmap='Reds')
    plt.colorbar()

    uu = np.reshape(data[:, 3], shape, order='A')
    plt.figure()
    plt.title('V')
    plt.tight_layout()
    plt.contourf(mesh_x, mesh_y, uu, levels=30, cmap='Reds')
    plt.colorbar()

    x_flat = mesh_x_target.flatten()
    y_flat = mesh_y_target.flatten()

    U = interp.griddata((data[:, 0], data[:, 1]),
                         data[:, 2], (x_flat, y_flat), method='linear')

    uinterpolated = np.reshape(U, mesh_x_target.shape, order='A')

    V = interp.griddata((data[:, 0], data[:, 1]),
                         data[:, 3], (x_flat, y_flat), method='linear')

    # vinterpolated = np.reshape(V, mesh_x_target.shape, order='A')

    # plt.figure()
    # plt.contourf(mesh_x_target, mesh_y_target, uinterpolated, levels=30, cmap='Reds')
    # plt.scatter(mesh_x_target, mesh_y_target)

    to_save = np.zeros((x_flat.shape[0], 4))
    to_save[:,0] = x_flat
    to_save[:,1] = y_flat
    to_save[:,2] = U
    to_save[:,3] = V

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/CD/common/DNS_interpolated_lines.csv', to_save, delimiter=',')

    # also interpolate to full mesh:
    mesh_x_flat, mesh_y_flat, mesh_z_flat = fluidfoam.readof.readmesh('/home/jasper/OpenFOAM/jasper-7/run/CD/CD_simplefoam_kOmegaSST/')

    u_full_field = interp.griddata((data[:, 0], data[:, 1]),
                                    data[:, 2], (mesh_x_flat, mesh_y_flat), method='linear')

    v_full_field = interp.griddata((data[:, 0], data[:, 1]),
                                    data[:, 3], (mesh_x_flat, mesh_y_flat), method='linear')


    to_save = np.zeros((mesh_x_flat.shape[0], 4))
    to_save[:,0] = mesh_x_flat
    to_save[:,1] = mesh_y_flat
    to_save[:,2] = u_full_field
    to_save[:,3] = v_full_field

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/CD/common/DNS_interpolated_field.csv', to_save, delimiter=',')


def read_and_plot_cases(base_dir):
    # change the working directory to dir with the openFOAM results

    dirlist = os.listdir(base_dir)
    dirlist.pop(dirlist.index('common'))

    results = {}
    for case in dirlist:
        result, name = read_case_results(os.path.join(base_dir, case))
        results[name] = result

    # load high fidelity_data
    commonlist = os.listdir(os.path.join(base_dir, 'common'))
    for file in commonlist:
        if 'LES' in file or 'DNS' in file:
            if 'LES' in file:
                label = 'LES'
            if 'DNS' in file:
                label = 'DNS'
            if 'line' in file:
                hifi_data_lines = np.genfromtxt(os.path.join(base_dir, 'common', file), delimiter=',')
            if 'field' in file:
                hifi_data_field = np.genfromtxt(os.path.join(base_dir, 'common', file), delimiter=',')

    # reshape interpolated data.
    n_lines = np.unique(hifi_data_lines[:, 0]).shape[0]
    n_points = int(hifi_data_lines.shape[0]/n_lines)
    #
    # lines = {'mesh_x': np.reshape(hifi_data_lines[:, 0], (n_points, n_lines), order='A'),
    #          'mesh_y': np.reshape(hifi_data_lines[:, 1], (n_points, n_lines), order='A'),
    #          'u': np.reshape(hifi_data_lines[:, 2], (n_points, n_lines), order='A'),
    #          'v': np.reshape(hifi_data_lines[:, 3], (n_points, n_lines), order='A')}

    lines = {'mesh_x': hifi_data_lines[:, 0],
             'mesh_y': hifi_data_lines[:, 1],
             'u': hifi_data_lines[:, 2],
             'v': hifi_data_lines[:, 3]}

    # plt.contourf(lines['mesh_x'], lines['mesh_y'], lines['v'], levels=30, cmap='Reds')

    if 'CBFS' in base_dir:
        # clip domain for MSE caluculation:
        xmin = 0
        xmax = 9
        ymin = 0
        ymax = 3
        x = hifi_data_field[:, 0]
        y = hifi_data_field[:, 1]

        keep_points = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
    else:
        keep_points = np.ones(hifi_data_field[:, 0].shape) == 1
        # new_x = x[bools]
        # new_y = y[bools]
        #
        #
        # x_mesh = reshape_to_mesh(x)
        # y_mesh = reshape_to_mesh(y)
        # ukomg_mesh = reshape_to_mesh(u_kOmegaSST[0, :])
        # vkomg_mesh = reshape_to_mesh(u_kOmegaSST[1, :])
        # y_mesh = reshape_to_mesh(y)
        # uhifi_mesh = reshape_to_mesh(u_hifi[0, :])
        # vhifi_mesh = reshape_to_mesh(u_hifi[1, :])
        # # plt.scatter(x_mesh[:, 0], y_mesh[:, 0])
        # plt.contourf(x_mesh, y_mesh, ukomg_mesh, levels=30, cmap='Reds')
        # plt.figure()
        # plt.contourf(x_mesh, y_mesh, uhifi_mesh, levels=30, cmap='Reds')
        #
        # ukomg_mesh = reshape_to_mesh(u_kOmegaSST[0, bools])
        # vkomg_mesh = reshape_to_mesh(u_kOmegaSST[1, bools])
        # y_mesh = reshape_to_mesh(y)
        # uhifi_mesh = reshape_to_mesh(u_hifi[0, bools])
        # vhifi_mesh = reshape_to_mesh(u_hifi[1, bools])
        # # plt.scatter(x_mesh[:, 0], y_mesh[:, 0])
        # plt.contourf(x_mesh, y_mesh, ukomg_mesh, levels=30, cmap='Reds')
        # plt.figure()
        # plt.contourf(x_mesh, y_mesh, uhifi_mesh, levels=30, cmap='Reds')
        #


    # calc error of k_omega and LES
    u_kOmegaSST = results['kOmegaSST']['U'][:2, keep_points]
    u_hifi = np.moveaxis(hifi_data_field[keep_points, 2:], 0, -1)

    mse_komg = sum(sum((u_hifi - u_kOmegaSST)**2))/(u_hifi.shape[0]*u_hifi.shape[1])

    # for each DSR run calc error between case and LES and divide by error of standard kOmegaSST
    runs = list(results.keys())
    runs.pop(runs.index('kOmegaSST'))

    best_mse_DSR = 1000
    best_mse_sparta = 1000
    best_dsr = None
    best_sparta = None

    for run in runs:
        u = results[run]['U'][:2, keep_points]
        mse = sum(sum((u_hifi - u) ** 2)) / (u_hifi.shape[0] * u_hifi.shape[1])
        norm_mse = mse/mse_komg
        results[run]['norm_mse'] = norm_mse
        if norm_mse < best_mse_DSR:
            if 'DSR' in run or 'SW' in run:
                best_dsr = run
                best_mse_DSR = norm_mse
        if norm_mse < best_mse_sparta:
            if 'Sparta' in run:
                best_sparta = run
                best_mse_sparta = norm_mse

    print(f'Best DSR run with normalised mse {best_mse_DSR} is {best_dsr}')
    print(f'Best sparta run with normalised mse {best_mse_sparta} is {best_sparta}')

    mesh_x_flat, mesh_y_flat = hifi_data_field[:, 0], hifi_data_field[:, 1]
    mesh_x = reshape_to_mesh(mesh_x_flat)
    # mesh_y = reshape_to_mesh(mesh_y_flat)
    n_points = mesh_x.shape[0]

    # # check if number of lines in interpolated data matches the openFoam pp folder
    ppkeys = list(results[list(results.keys())[0]]['pp'].keys())
    ppkeys.pop(ppkeys.index('residuals'))
    if len(ppkeys) != n_lines:
        raise ValueError('The interpolated high fidelity data contains a different number of lines than the OF results')

    plt.figure()
    plt.plot(mesh_x_flat[:n_points], mesh_y_flat[:n_points], c='Black')
    plt.plot(mesh_x_flat[-n_points:], mesh_y_flat[-n_points:], c='Black')
    plt.plot([mesh_x_flat[0], mesh_x_flat[-n_points]], [mesh_y_flat[0], mesh_y_flat[-n_points]], c='Black')
    plt.plot([mesh_x_flat[n_points - 1], mesh_x_flat[-1]], [mesh_y_flat[n_points - 1], mesh_y_flat[-1]], c='Black')

    ax = plt.gca()
    ax.set_aspect('equal')
    # add LES results:
    u_scale = 1

    for x in np.unique(hifi_data_lines[:, 0].round()):
        plot_bool = (lines['mesh_x'] > x - 0.1) & (lines['mesh_x'] < x + 0.1)
        plt.plot(x + lines['u'][plot_bool],
                 lines['mesh_y'][plot_bool], c='Black', marker='o', markevery=5, label=label)
        if label:
            label = None

    label = r'$k-\omega$ SST'
    # add baseline kOmegaSST results:
    for line in results['kOmegaSST']['pp']:
        if 'single' in line:
            data = results['kOmegaSST']['pp'][line]['line_U']
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
    #
    # exclude = 6
    # plt.scatter(mesh_x_flat[:exclude*n_points], mesh_y_flat[:exclude*n_points], c='Black')
    # plt.scatter(mesh_x_flat[-exclude*n_points:], mesh_y_flat[-exclude*n_points:], c='Black')


if __name__ == '__main__':

    # read_and_plot_PH()
    matplotlib.use('tkagg')
    #
    base_dir = '/home/jasper/OpenFOAM/jasper-7/run/CD'
    # # # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/CBFS'
    # # # base_dir = '/home/jasper/OpenFOAM/jasper-7/run/PH'
    read_and_plot_cases(base_dir)

    # interpolate_PH()
    # interpolate_CBFS()
    # interpolate_CD()

    print('end')
    print('end')
