import pickle
import os
import platform
from dsr.turbulence.dataprocessing import load_frozen_RANS_dataset, de_flatten_tensor, calc_sij_rij
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib

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
    steps = (0.5 + np.asarray(range(len(colors) - 1), dtype=np.float)) / (len(colors) - 1)
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

def tricolor_plot_tensor(a, data_i):


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

    eigs = np.linalg.eigvalsh(aij_bous)
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
    # scale lamcolor?
    lamcolor[:,0] = lamcolor[:,0]/max(lamcolor[:,0])
    lamcolor[:,1] = lamcolor[:,1]/max(lamcolor[:,1])

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
    plt.show()



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

    # # aij_bous = -2 * data_i['nut_frozen'] * Sij
    # aij_bous = data_i['aij']
    #
    # aij_bous = np.moveaxis(aij_bous, -1, 0)



    # below calculation of  aij as done in tricolor_plot, uses "perfect" DNS data i think
    uu = data_i['uu']
    vv = data_i['vv']
    ww = data_i['ww']
    uv = data_i['uv']
    k = .5 * (uu+vv+ww)
                                        # Anisotropy tensor - where k==0 (e.g. on
                                        # wall) tensor is not defined.
    aij_bous = np.zeros((len(uu),3,3))
    aij_bous[:,0,0] = uu/(2*k) - 1./3
    aij_bous[:,1,1] = vv/(2*k) - 1./3
    aij_bous[:,2,2] = ww/(2*k) - 1./3
    aij_bous[:,0,1] = aij_bous[:,1,0] = uv/(2*k)
    aij_bous[k < 1.e-10] = 0.

    tricolor_plot_tensor(aij_bous, data_i)

    print('end')
    print('end')