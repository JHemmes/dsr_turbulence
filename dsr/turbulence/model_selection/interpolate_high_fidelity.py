import os

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import fluidfoam
import scipy.interpolate as interp



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

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/common/CBFS_lines.csv', to_save, delimiter=',')

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

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/common/CBFS_field.csv', to_save, delimiter=',')

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
    # shape = (281, 234)
    shape = (234, 281)
    mesh_x = np.reshape(data[:, 0], shape, order='A').T


    ########  tauij investigation
    # mesh_y = np.reshape(data[:, 1], shape, order='A').T
    # # mesh_u = np.reshape(data[:, 2], shape, order='A').T
    # # mesh_v = np.reshape(data[:, 3], shape, order='A').T
    #
    # col = 9
    #
    # reshaped = np.reshape(data[:, col], shape, order='A').T
    # plt.figure()
    # plt.title(f'{col}')
    # plt.plot(mesh_x[:, 0], reshaped[:, 2])
    # plt.scatter(mesh_x[:, 1], mesh_y[:, 1])




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

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/common/PH_lines.csv', to_save, delimiter=',')

    # also interpolate to full mesh:
    mesh_x_flat, mesh_y_flat, mesh_z_flat = fluidfoam.readof.readmesh('/home/jasper/OpenFOAM/jasper-7/run/PH/PH_simplefoam_kOmegaSST/')

    u_field = interp.griddata((data[:, 0], data[:, 1]), data[:, 2], (mesh_x_flat, mesh_y_flat), method='linear')

    v_field = interp.griddata((data[:, 0], data[:, 1]), data[:, 3], (mesh_x_flat, mesh_y_flat), method='linear')

    to_save = np.zeros((mesh_x_flat.shape[0], 4))
    to_save[:, 0] = mesh_x_flat
    to_save[:, 1] = mesh_y_flat
    to_save[:, 2] = u_field
    to_save[:, 3] = v_field

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/common/PH_field.csv', to_save, delimiter=',')

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

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/common/CD_lines.csv', to_save, delimiter=',')

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

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/common/CD_field.csv', to_save, delimiter=',')


def parse_piv_file(file):
    with open(file) as f:
        lines = f.readlines()

    numbers = []
    for line in lines:
        if 'x/h' in line:
            line = line.strip('\n')
            x = float(line.split('=')[-1])
        if '#' in line:
            continue

        line = line.strip('\n')
        linesplit = line.split(',')
        numbers.append([float(val) for val in linesplit])

    data = np.array(numbers)

    return data, x


def fetch_save_experimental_data():
    logdir = "/home/jasper/Documents/afstuderen/python/inversion/DATA/PH-Rapp/data"

    dirlist = os.listdir(logdir)

    x_stations = []
    y_list = []
    u_list = []
    v_list = []

    for file in dirlist:
        data, x = parse_piv_file(os.path.join(logdir, file))
        x_stations.append(x)
        y_list.append(data[:, 0])
        u_list.append(data[:, 1])
        v_list.append(data[:, 2])

    # for file in dirlist:
    x_stations_sorted = sorted(x_stations)

    x_list_sorted = []
    y_list_sorted = []
    u_list_sorted = []
    v_list_sorted = []

    for x in x_stations_sorted:
        index = x_stations.index(x)
        x_list_sorted.append(x*np.ones(y_list[index][u_list[index] != 0].shape))
        y_list_sorted.append(y_list[index][u_list[index] != 0])
        u_list_sorted.append(u_list[index][u_list[index] != 0])
        v_list_sorted.append(v_list[index][u_list[index] != 0])

    # plt.figure()
    # for ii in range(len(x_stations_sorted)):
    #     # plt.scatter([x_stations[ii] for val in y_list[ii]], y_list[ii])
    #     plt.plot(x_list_sorted[ii] + u_list_sorted[ii], y_list_sorted[ii] )

    x_arr = np.concatenate(x_list_sorted)
    y_arr = np.concatenate(y_list_sorted)
    u_arr = np.concatenate(u_list_sorted)
    v_arr = np.concatenate(v_list_sorted)

    to_save = np.moveaxis(np.vstack([x_arr, y_arr, u_arr, v_arr]), -1, 0)

    np.savetxt('/home/jasper/OpenFOAM/jasper-7/run/common/PH_exp.csv', to_save, delimiter=',')



if __name__ == '__main__':


    # fetch_save_experimental_data()


    interpolate_PH()
    interpolate_CBFS()
    interpolate_CD()
