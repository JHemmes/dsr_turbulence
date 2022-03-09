import pickle
import os
import platform
from dsr.turbulence.dataprocessing import load_frozen_RANS_dataset, de_flatten_tensor, calc_sij_rij
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib
import json

iX, iY = 0, 1

def colors_to_cmap(colors):
    """
    Yields a matplotlib colormap object
    that,  reproduces the colors in the given array when passed a
    list of N evenly spaced numbers between 0 and 1 (inclusive), where N is the
    first dimension of ``colors``.

    Args:
      colors (ndarray (N,[3|4])): RGBa_array
    Return:
      cmap (matplotlib colormap object): Colormap reproducing input colors,
                                         cmap[i/(N-1)] == colors[i].

    Example:
      cmap = colors_to_cmap(colors)
      zs = np.linspace(0,1,range(len(colors)))
    """
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        colors = np.hstack((colors, np.ones((len(colors), 1))))
    steps = (0.5 + np.asarray(range(len(colors) - 1), dtype=float)) / (len(colors) - 1)
    return matplotlib.colors.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: ([(0, col[0], col[0])] +
                   [(step, c0, c1) for (step, c0, c1) in zip(steps, col[:-1], col[1:])] +
                   [(1, col[-1], col[-1])])
         for (clridx, clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:, clridx]]},
        N=len(colors))

class BarycentricColormap:

    def __init__(self):
        # Vertices of output triangle
        self.xv = np.array([[0, 0],
                            [1, 0],
                            [.5, np.sqrt(3) / 2]])
        xv = self.xv
        self.Tinv = np.linalg.inv(
            np.array([[xv[0, iX] - xv[2, iX], xv[1, iX] - xv[2, iX]],
                      [xv[0, iY] - xv[2, iY], xv[1, iY] - xv[2, iY]]]))

    def bary2cartesian(self, lam):
        """
        Convert barycentric coordinates (normalized) ``lam`` (ndarray (N,3)), to
        Cartesian coordiates ``x`` (ndarray (N,2)).
        """
        return np.einsum('ij,jk', lam, self.xv)

    def cartesian2bary(self, x):
        """
        Convert Cartesian coordiates ``x`` (ndarray (N,2)), to barycentric
        coordinates (normalized) ``lam`` (ndarray (N,3)).
        """
        lam = np.zeros((x.shape[0], 3))
        lam[:, :2] = np.einsum('ij,kj->ki', self.Tinv, x - self.xv[2])
        lam[:, 2] = 1. - lam[:, 0] - lam[:, 1]
        return lam

    def trigrid(self, n=10):
        """Uniform grid on triangle in barycentric coordinates."""
        lam = []
        for lam1 in range(n):
            for lam2 in range(n - lam1):
                lam3 = n - lam1 - lam2
                lam.append([lam1, lam2, lam3])
        return np.array(lam) / float(n)

    def randomgrid(self, n):
        lam = np.random.random((n, 3))
        return self.normalize(lam)

    def normalize(self, lam):
        """Normalize Barycentric coordinates to 1."""
        return (lam.T / np.sum(lam, axis=1)).T

def tricolor_plot_tensor(a, data_i, title=None):


    mesh_x = data_i['meshRANS'][0, :, :]
    mesh_y = data_i['meshRANS'][1, :, :]

    mesh_x_flat = mesh_x.flatten(order='F').T
    mesh_y_flat = mesh_y.flatten(order='F').T

    xspace = np.vstack([mesh_x_flat, mesh_y_flat]).T
    tri_del = Delaunay(xspace)
    tris = tri_del.simplices

    # find mask for tris
    mask = tris <= mesh_x.shape[0]
    mask = ~mask.all(axis=1)

    trispace = tri.Triangulation(mesh_x_flat, mesh_y_flat, triangles=tris[mask])

    eigs = np.linalg.eigvalsh(a)
    eigs.sort(axis=1)
    eigs = eigs[:,::-1]

    # Barycentric coordinates
    lamspace = eigs.copy()
    lamspace[:,0] -= lamspace[:,1]
    lamspace[:,1] = 2*(lamspace[:,1]-lamspace[:,2])
    lamspace[:,2] = 3*lamspace[:,2]+1

    barymap = BarycentricColormap()

    # Grid/data for legend
    lamlegend  = barymap.trigrid(100)
    # xlegend    = barymap.bary2cartesian(lamlegend)
    # trilegend  = Delaunay(xlegend)

    # Build colormaps
    lamcolor = (lamspace.T / np.max(lamspace, axis=1)).T
    # # scale lamcolor?
    # lamcolor[:,0] = lamcolor[:,0]/max(lamcolor[:,0])
    # lamcolor[:,1] = lamcolor[:,1]/max(lamcolor[:,1])

    lamlegend = (lamlegend.T / np.max(lamlegend, axis=1)).T
    cmap_space = colors_to_cmap(lamcolor)
    # cmap_legend = colors_to_cmap(lamlegend)

    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.tripcolor(mesh_x_flat, mesh_y_flat, trispace.triangles,
                 np.linspace(0,1,mesh_x_flat.shape[0]),
                 edgecolors='none', cmap=cmap_space, shading='gouraud')
    plt.gca().axison = False
    plt.gca().set_aspect('equal')
    plt.title(title)
    # plt.show()



if __name__ == "__main__":

    dsrpath = os.path.abspath(__file__)
    if platform.system() == 'Windows':
        os.chdir(dsrpath[:dsrpath.find('\\dsr\\')+4]) # change the working directory to main dsr dir
    else:
        os.chdir(dsrpath[:dsrpath.find('/dsr/')+4]) # change the working directory to main dsr dir with the config files

    case = 'PH10595'
    frozen = pickle.load(open(f'turbulence/frozen_data/{case}_frozen_var.p', 'rb'))
    data_i = frozen['data_i']

    Sij, Rij = calc_sij_rij(data_i['grad_u'], data_i['omega_frozen'], normalize=False)

    # bij = data_i['bij']
    # bij = np.moveaxis(bij, -1, 0)
    # tricolor_plot_tensor(bij, data_i, 'bij')

    nut = data_i['nut_frozen']
    k = data_i['k']

    bij_LEVM = -(nut/k)*Sij
    bij_LEVM = np.moveaxis(bij_LEVM, -1, 0)

    tricolor_plot_tensor(bij_LEVM, data_i, 'LEVM')

    bdelta =  np.moveaxis(data_i['bDelta'], -1, 0)
    bij_corr = bij_LEVM + bdelta

    tricolor_plot_tensor(bij_corr, data_i, 'LEVM plus bDelta')

    # load dataset:
    with open('config_bDelta.json', encoding='utf-8') as f:
        config = json.load(f)

    X, y = load_frozen_RANS_dataset(config['task'])

    x1 = X[:,0]
    x2 = X[:,1]
    x3 = X[:,2]
    x4 = X[:,3]
    x5 = X[:,4]
    x6 = X[:,5]
    x7 = X[:,6]
    x8 = X[:,7]
    x9 = X[:,8]
    x10 = X[:,9]

    dsr_bDelta = x1*(18.53628531372629*x7 + 0.53041498422016339) + 0.17597672099208503*x2/(-1.5907187870838878*x5 + x6 + 0.59071878708388784*np.exp(x5 - x6) + np.log(x5 - x9 + np.exp(np.exp(x5))) - 1.590944655918643) + x3*(x5 + x7 + x8 + 10.176040402349427) + x4*(13717.614941230593*x5*(x6 - 0.061601349115670084 + 0.013417116728867052*(x7 - 0.050292032841165346)/x7)*(x7 + x8*(np.exp(x5) - 3241.9616916815266) + np.exp(x5)) + 1.0781230466703526)

    # dsr_bDelta = x1 * (np.exp(x7) - 0.7400604774603605) + x2 * (0.032630438215257891 * np.exp(x10) - (
    #             -0.17095618044834858 * x6 - 0.0002021103539912845) / x5) / x6 + x3 * (x5 + x8) + x4 * (
    #             x10 - 4.424733413784568)

    inv_nrmse = 1 / (1 + np.sqrt(np.mean((y-dsr_bDelta)**2))/np.std(y))

    dsr_bDelta = de_flatten_tensor(dsr_bDelta)
    dsr_bDelta =  np.moveaxis(dsr_bDelta, -1, 0)

    dsr_bij_corrected = bij_LEVM + dsr_bDelta
    tricolor_plot_tensor(dsr_bij_corrected, data_i, f'DSR model, inv_nrmse = {inv_nrmse}')

    plt.show()



    print('end')
    print('end')