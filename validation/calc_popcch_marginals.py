from quantities import ms
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import sys
from time import time
# from mpi4py import MPI
# from matrix import plot_matrix


def load(filename, rescale=False, return_pairs=True, array_name='cch_array',
         pairs_name='split_pairs'):
    file = np.load(filename)
    if return_pairs:
        if rescale:
            return np.squeeze(file[array_name]) / file['max_cc'][0], \
                   file[pairs_name]
        else:
            return np.squeeze(file[array_name]), file[pairs_name]
    else:
        if rescale:
            return np.squeeze(file[array_name])/file['max_cc'][0]
        else:
            return np.squeeze(file[array_name])


def summed_pop_cch(cch_array, plot=False, ax=None, symetric=True, binsize=None,
                   **kwargs):
    N = len(np.squeeze(cch_array))
    popcch = np.sum(np.squeeze(cch_array), axis=0)
    popcch /= float(N)
    if symetric:
        popcch = popcch + popcch[::-1]
        popcch /= 2.
    B = len(popcch)
    w = B/2
    if binsize is not None
        w = w * float(binsize)
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.bar(np.linspace(-w,w,2*w+1), popcch, **kwargs)
        ax.xlim((-w,w))
        if binsize is None:
            ax.set_xlabel(r'$\tau$ [bins]')
        else:
            ax.set_xlabel(r'$\tau$ [ms]')
    return popcch


def generalized_cc_dist(cch_array, bins=500, plot=False, ax=None):
    hist, edges = np.histogram(cch_array.flatten(), bins=bins, density=True)
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        dx = np.diff(edges)[0]
        xvalues = edges[:-1] + dx / 2.
        ax.plot(xvalues, hist)
    return hist, edges


def generalized_cc_matrix(cch_array, pair_ids, time_reduction='sum',
                          plot=False, ax=None, **kwargs):
    B = len(np.squeeze(cch_array)[0])
    if time_reduction == 'sum':
        cc_array = np.sum(np.squeeze(cch_array), axis=1)
    if time_reduction == 'max':
        cc_array = np.amax(np.squeeze(cch_array), axis=1)
    if time_reduction[:3] == 'lag':
        lag = int(time_reduction[3:])
        cc_array = np.squeeze(cch_array)[:, B/2 + lag]
    if time_reduction[:9] == 'threshold':
        th = float(time_reduction[10:])
        th_cch_array = np.array([a[a>th]-th for a in np.squeeze(cch_array)])
        cc_array = np.array([np.sum(cch) for cch in th_cch_array])
    N = len(cc_array)
    dim = .5*(1 + np.sqrt(8.*N + 1))
    assert not dim - int(dim)
    dim = int(dim)
    cc_mat = np.zeros((dim,dim))
    for count, (i,j) in enumerate(pair_ids):
        cc_mat[i,j] = cc_array[count]
        cc_mat[j,i] = cc_array[count]
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        plot_matrix(cc_mat, ax=ax, **kwargs)
    return cc_mat


def temporal_spread(cch_array, pair_ids, plot=False, ax=None, binsize=2*ms):
    # sum over all pairs of neuron i
    ccharray = np.squeeze(cch_array)
    N = len(ccharray)
    B = len(ccharray[0])
    dim = .5*(1 + np.sqrt(8.*N + 1))
    assert not dim - int(dim)
    dim = int(dim)
    involved_pair_ids = np.zeros((dim,dim-1))
    for i in range(dim):
        involved_pair_ids[i] = np.concatenate([np.where(pair_ids[:,idx]==i)[0]
                                               for idx in range(2)])
    pair_cch_array = np.array([np.sum(ccharray[ids.astype(int)], axis=0)
                               for ids in involved_pair_ids])
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        plot_matrix(pair_cch_array, ax=ax)
        ax.set_xlabel('time [bins]')
        ax.set_xticks([0,B/2,B])
        ax.set_xticklabels(['{}'.format(label)
                            for label in [-(B/2),0,B/2]])
        ax.set_ylabel('Neuron pair including neuron i')
    return pair_cch_array


def _alpha(color_inst, a):
    return [el + (1.-el)*a for el in np.array(color_inst)]


def cch_space(color_array, pair_tau_ids, N, B,
              palette=sns.cubehelix_palette(10, start=.3, rot=.6),
              binsize=2*ms, alpha=False, **kwargs):
    # color_array is an sparse int array of the thresholded cchs
    # transformed to [0..9] -> 0 is transparent, 1-9 is used for indexing the
    # color palette. Since the color_cch is rescaled there is exactly one
    # element with value 10, which always projected to k
    colorarray = np.squeeze(color_array)
    palette = palette + [[0,0,0]] # single max value is black

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Neuron #1')
    ax.set_xlim3d(0, N)
    ax.set_ylabel('Tau [ms]')
    ax.set_ylim3d(-B/2*float(binsize), B/2*float(binsize))
    ax.set_zlabel('Neuron #2')
    ax.set_zlim3d(0, N)

    tau = (np.arange(B) - B/2) * float(binsize)
    for count, (i,j,t) in enumerate(pair_tau_ids):
        if alpha:
            color = _alpha(palette[colorarray[count]], 1.-colorarray[count]/10.)
        else:
            color = palette[colorarray[count]]
        if t < 1:
            t = 1
            print 'border value shifted'
        elif t == B-1:
            t = B-2
            print 'border value shifted'
        ax.plot([j,j], tau[t-1:t+1], [i,i], c=color, **kwargs)
        # expects the outer most values for tau not to be significant
    return ax, palette


if __name__ == '__main__':
    start_time = time()

    path = '/home/robin/Projects/pop_cch_results/'

    for i in range(2):
        colorfilename = 'color_array_set{}_th{}.npz'.format(i,0.5)

        color_array, pair_tau_ids = load(path+colorfilename,
                                         array_name='color_array',
                                         pairs_name='pair_tau_ids')

        ax, palette = cch_space(color_array, pair_tau_ids, B=201, N=250)
        ax.set_title('set {}'.format(i))


    fig, cax = plt.subplots()
    cmap = mpl.colors.ListedColormap(palette)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap)
    fig.show()

    # filename = 'cch_array_set{}_bin2ms_lag100bins.npz'.format(1)

    # ccharray, pairs = load(path+filename, rescale=False, return_pairs=True)

    # summed_pop_cch(ccharray, plot=True, symetric=True)

    # generalized_cc_dist(ccharray, plot=True)

    # generalized_cc_matrix(ccharray, pairs, plot=True, time_reduction='lag 0',
    #                       sort=True)

    # temporal_spread(ccharray, pairs, plot=True)

    m, s = divmod(time() - start_time, 60)
    h, m = divmod(m, 60)
    print 'Computation took %d:%02d:%02d' % (h, m, s)

    plt.show()
