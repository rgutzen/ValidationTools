from quantities import ms
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import seaborn as sns
import numpy as np
from copy import copy
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


def summed_pop_cch(cch_array, plot=False, ax=None, binsize=None, symmetric=True,
                   hist_filter=None, filter_to_binary=False, average=False,
                   **pltargs):
    ccharray = copy(np.squeeze(cch_array))
    N = len(ccharray)
    if hist_filter is not None:
        if hist_filter == 'max':
            max_array = np.amax(ccharray, axis=1)
            if filter_to_binary:
                for i, cch in enumerate(ccharray):
                    ccharray[i] = np.where(cch < max_array[i], 0, 1)
            else:
                for i, cch in enumerate(ccharray):
                    ccharray[i] = np.where(cch < max_array[i], 0, max_array[i])
        if hist_filter[:9] == 'threshold':
            th = float(hist_filter[9:])
            if filter_to_binary:
                for i, cch in enumerate(ccharray):
                    ccharray[i] = np.where(cch < th, 0, 1)
            else:
                for i, cch in enumerate(ccharray):
                    ccharray[i] = np.where(cch < th, 0, cch)
    popcch = np.sum(ccharray, axis=0)
    if average:
        nonzeros = np.sum(ccharray.astype(bool).astype(float), axis=0)
        popcch = popcch / nonzeros
    popcch = np.where(np.isfinite(popcch), popcch, 0)
    if symmetric:
        popcch = popcch + popcch[::-1]
        popcch /= 2.

    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        B = len(popcch)
        w = B / 2
        if binsize is None:
            binsize = 1
            ax.set_xlabel('Time lag [bins]')
        else:
            w = w * float(binsize)
            ax.set_xlabel('Time lag [ms]')
        if 'width' not in pltargs:
            pltargs['width'] = 0.9
        if 'edgecolor' not in pltargs:
            pltargs['edgecolor'] = 'w'
        pltargs['width'] *= float(binsize)
        ax.bar(np.linspace(-w,w,B/2*2+1)-float(binsize)/2., popcch, **pltargs)
        ax.set_ylabel('average cross correlation')
        ax.set_xlim((-w,w))
    return popcch


def generalized_cc_dist(cch_array, bins=500, plot=False, ax=None,
                        hist_filter=None, **pltargs):
    ccharray = copy(np.squeeze(cch_array))
    if hist_filter is None:
        ccharray = ccharray.flatten()
    else:
        if hist_filter == 'max':
            max_array = np.amax(ccharray, axis=1)
            for i, cch in enumerate(ccharray):
                ccharray[i] = np.where(cch < max_array[i], 0, max_array[i])
            ccharray = ccharray.flatten()
            ccharray = ccharray[np.where(ccharray)[0]]
        if hist_filter[:9] == 'threshold':
            th = float(hist_filter[9:])
            for i, cch in enumerate(ccharray):
                ccharray[i] = np.where(cch < th, 0, cch)
            ccharray = ccharray.flatten()
            ccharray = ccharray[np.where(ccharray)[0]]
    hist, edges = np.histogram(ccharray.flatten(), bins=bins, density=True)
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        dx = np.diff(edges)[0]
        xvalues = edges[:-1] + dx / 2.
        ax.plot(xvalues, hist, **pltargs)
        ax.set_xlabel('Generalized Correlation Coefficient')
    return hist, edges


def pop_cc_hist_dist(cch_array, ax=None, binsize=None, bins=500,
                    hist_filter=None, filter_to_binary=False,
                    **pltargs):
    if ax is None:
        fig, ax = plt.subplots()
    left, bottom, width, height = ax.get_position()._get_bounds()
    scaling = .75
    ax.set_position([left, bottom,
                     scaling * width, height])
    axdist = plt.axes([left + scaling * width, bottom,
                       (1 - scaling) * width, height])
    axdist.yaxis.tick_right()
    axdist.get_xaxis().tick_bottom()
    axdist.yaxis.set_label_position("right")
    axdist.spines["left"].set_visible(False)
    axdist.spines["top"].set_visible(False)

    barpltargs = copy(pltargs)
    if 'width' not in barpltargs:
        barpltargs['width'] = 0.95
    if hist_filter is None:
        symmetric = True
    else:
        symmetric = False
    summed_pop_cch(cch_array, plot=True, ax=ax, binsize=binsize, symmetric=symmetric,
                   hist_filter=hist_filter, filter_to_binary=filter_to_binary,
                   average=True, **barpltargs)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel('')
    ax.yaxis.set_ticks_position('none')
    ax.get_xaxis().tick_bottom()
    ax.yaxis.set_major_formatter(NullFormatter())

    hist, edges = generalized_cc_dist(cch_array, bins=bins, plot=False,
                        hist_filter=hist_filter, **pltargs)
    dx = np.diff(edges)[0]
    xvalues = edges[:-1] + dx / 2.
    axdist.plot(hist, xvalues, **pltargs)
    axdist.set_ylim([min(cch_array.flatten()), max(cch_array.flatten())])
    axdist.set_xlabel('Density')
    ymin = min([axdist.get_ylim()[0], 0])
    axdist.set_ylim((ymin, axdist.get_ylim()[1]))
    ax.set_ylim((ymin, axdist.get_ylim()[1]))
    ax.plot(ax.get_xlim(),[0,0], color='0.5', lw=.5)
    axdist.plot(axdist.get_xlim(),[0,0], color='0.5', lw=.5)

    return ax, axdist


def tau_cc_cluster(cch_array, hist_filter=None, binsize=None, kind='hex', **kwargs):
    ccharray = copy(np.squeeze(cch_array))
    N = len(ccharray)
    B = len(ccharray[0])
    if binsize is None:
        binsize = 1
    w = B/2 * float(binsize)
    tau = np.array(list(np.linspace(-w, w, B/2 * 2 + 1)) * N)
    if hist_filter is None:
        ccharray = ccharray.flatten()
    else:
        if hist_filter == 'max':
            max_array = np.amax(ccharray, axis=1)
            for i, cch in enumerate(ccharray):
                ccharray[i] = np.where(cch < max_array[i], 0, max_array[i])
        if hist_filter[:9] == 'threshold':
            th = float(hist_filter[9:])
            for i, cch in enumerate(ccharray):
                ccharray[i] = np.where(cch < th, 0, cch)
        ccharray = ccharray.flatten()
        tau = tau[np.where(ccharray)[0]]
        ccharray = ccharray[np.where(ccharray)[0]]
    grid = sns.jointplot(tau, ccharray, kind=kind, xlim=(-w,w), **kwargs)
    if hist_filter is not None and hist_filter[:9] == 'threshold':
        ax = plt.gca()
        ax.text(-1.2, .9, 'Threshold = {}'.format(th), transform=ax.transAxes)
    return grid


def generalized_cc_matrix(cch_array, pair_ids, time_reduction='sum',
                          plot=False, ax=None, rescale=False, **kwargs):
    B = len(np.squeeze(cch_array)[0])
    if time_reduction == 'sum':
        cc_array = np.sum(np.squeeze(cch_array), axis=1)
        if rescale:
            cc_array = cc_array / float(B)
    if time_reduction == 'max':
        cc_array = np.amax(np.squeeze(cch_array), axis=1)
    if time_reduction[:3] == 'lag':
        lag = int(time_reduction[3:])
        cc_array = np.squeeze(cch_array)[:, B/2 + lag]
    if time_reduction[:9] == 'threshold':
        th = float(time_reduction[10:])
        th_cch_array = np.array([a[a>th] for a in np.squeeze(cch_array)])
        if rescale:
            cc_array = np.array([np.sum(cch)/float(len(cch)) if len(cch)
                                 else np.sum(cch)
                                 for cch in th_cch_array])
        else:
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
    sns.set(style='ticks', palette='Set2')
    sns.set_color_codes('colorblind')
    start_time = time()

    path = '/home/robin/Projects/pop_cch_results/'

    # for i in range(2):
    #     colorfilename = 'color_array_set{}_th{}.npz'.format(i,0.5)
    #
    #     color_array, pair_tau_ids = load(path+colorfilename,
    #                                      array_name='color_array',
    #                                      pairs_name='pair_tau_ids')
    #
    #     ax, palette = cch_space(color_array, pair_tau_ids, B=201, N=250)
    #     ax.set_title('set {}'.format(i))

    # fig, cax = plt.subplots()
    # cmap = mpl.colors.ListedColormap(palette)
    # cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap)
    # fig.show()

    filename = 'cch_array_set{}_bin2ms_lag100bins.npz'.format(0)

    ccharray, pairs = load(path+filename, rescale=False, return_pairs=True)

    summed_pop_cch(ccharray[:10], plot=True,
                   hist_filter='sum', filter_to_binary=False, color='r')

    # pop_cc_hist_dist(ccharray, ax=None, binsize=2, bins=500,
    #                  hist_filter='threshold 0.05',
    #                  filter_to_binary=False,
    #                  color='b')

    # tau_cc_cluster(ccharray, hist_filter=None, kind='scatter',
    #                marginal_kws=dict(bins=200), color='b')

    # generalized_cc_dist(ccharray, plot=True)

    # ccmat = generalized_cc_matrix(ccharray, pairs, plot=True, time_reduction='lag 0',
    #                       sort=False, cluster=True, remove_autocorr=True, rescale=True)
    #
    # sns.clustermap(ccmat, row_linkage=lmat, col_linkage=lmat, method='ward')

    # sns.clustermap(ccmat, metric='euclidean')

    # temporal_spread(ccharray, pairs, plot=True)

    m, s = divmod(time() - start_time, 60)
    h, m = divmod(m, 60)
    print 'Computation took %d:%02d:%02d' % (h, m, s)

    plt.show()
