from quantities import ms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import sys
from time import time
# from mpi4py import MPI
from matrix import plot_matrix


def load(filename, rescale=False, return_pairs=True):
    file = np.load(filename)
    if return_pairs:
        if rescale:
            return np.squeeze(file['cch_array']) / file['max_cc'][0], \
                   file['split_pairs']
        else:
            return np.squeeze(file['cch_array']), file['split_pairs']
    else:
        if rescale:
            return np.squeeze(file['cch_array'])/file['max_cc'][0]
        else:
            return np.squeeze(file['cch_array'])


def summed_pop_cch(cch_array, plot=False, ax=None, symetric=True):
    N = len(np.squeeze(cch_array))
    popcch = np.sum(np.squeeze(cch_array), axis=0)
    popcch /= float(N)
    if symetric:
        popcch = popcch + popcch[::-1]
        popcch /= 2.
    B = len(popcch)
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.bar(np.linspace(-B/2,B/2,B), popcch)
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


def cch_space(color_array, pair_ids, binsize=2*ms, threshold=.0, plot=True,
             save=True, save_id=0, **kwargs):
    # Rewrite for new processed data input from blaustein
    ccharray = np.squeeze(cch_array)
    N = len(ccharray)
    B = len(ccharray[0])
    if plot:
        fig = plt.figure()
        palette = sns.color_palette('coolwarm')[::]
        ax = fig.gca(projection='3d')
        ax.set_xlabel('Neuron #1')
        ax.set_xlim3d(0, N)
        ax.set_ylabel('Tau')
        ax.set_ylim3d(0, B)
        ax.set_zlabel('Neuron #2')
        ax.set_zlim3d(0, N)
    else:
        ax = None
    tau = np.arange(B) * float(binsize)
    sig_cc = []
    for count, (i,j) in enumerate(pair_ids):
        for ti in range(B-2):
            cc = (ccharray[count,ti] + 1) / 2.
            if cc < 0:
                print "({} {}): Index Error {} -> 0".format(i,j,cc)
                cc = 0.
            elif cc > 1:
                print "({} {}): Index Error {} -> 1".format(i,j,cc)
                cc = 1.
            if cc > threshold:
                sig_cc += [(i,j)]
                if plot:
                    color_i = int(np.ceil(cc*len(palette))-1)
                    color = _alpha(palette[color_i], 1.-cc)
                    ax.plot([j,j], tau[ti:ti+2], [cc + i]*2, c=color)
    print "=> {} entries above threshold".format(len(sig_cc))
    if save:
        idfile = open('sig_cc_idx_.txt'.format(save_id), 'w')
        for item in sig_cc:
            idfile.write("{} \n".format(item))
        idfile.close()
    return sig_cc, ax


if __name__ == '__main__':
    path = '/home/robin/Projects/pop_cch_results/'
    filename = 'cch_array_set{}_bin2ms_lag100bins.npz'.format(1)

    start_time = time()

    ccharray, pairs = load(path+filename, rescale=False, return_pairs=True)

    # cch_space(ccharray, pairs, plot=True, threshold=0.15)

    # summed_pop_cch(ccharray, plot=True, symetric=True)

    generalized_cc_dist(ccharray, plot=True)

    # generalized_cc_matrix(ccharray, pairs, plot=True, time_reduction='lag 0',
    #                       sort=True)

    # temporal_spread(ccharray, pairs, plot=True)

    m, s = divmod(time() - start_time, 60)
    h, m = divmod(m, 60)
    print 'Computation took %d:%02d:%02d' % (h, m, s)

    plt.show()
