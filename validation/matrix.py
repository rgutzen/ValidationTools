# -*- coding: utf-8 -*-
"""
Toolbox for analyzing a correlation matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import contextlib
import pandas as pd
import math
from quantities import Hz, ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef
import neo
# Seaborn is not included in the HBP environment
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.set_color_codes('colorblind')
sns.despine()


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def corr_matrix(spiketrains, binsize=2*ms, corr_type='pearson'):
    # ToDo: Implement Correlation (pearson, spearman, ...?)
    t_lims = [(st.t_start, st.t_stop) for st in spiketrains]
    tmin = min(t_lims, key=lambda f: f[0])[0]
    tmax = max(t_lims, key=lambda f: f[1])[1]
    if corr_type == 'pearson':
        binned_sts = BinnedSpikeTrain(spiketrains, binsize,
                                      t_start=tmin, t_stop=tmax)
        return corrcoef(binned_sts)
    return None


def pc_trafo(matrix, EWs=[], EVs=[]):
    assert len(EWs) == len(EVs)

    if len(EWs) == 0:
        EWs, EVs = np.linalg.eig(matrix)
    else:
        assert len(matrix) == len(EWs)

    diag_matrix = np.dot(np.dot(EVs.T, matrix), EVs)
    # print pd.DataFrame(diag_matrix[:6,:6]).round(3)
    # print pd.DataFrame(np.diag(EWs[:6])).round(3)
    return diag_matrix


def plot_matrix(matrix, ax=plt.gca()):
    labelnum = matrix.shape[0]/10
    if labelnum == 1:
        labelnum = 2
    sns.heatmap(matrix, ax=ax, cbar=True, xticklabels=labelnum, yticklabels=labelnum)
    # ToDo: offer 'sorted' option for estimated assemblies presentation
    return None


def eigenvalue_distribution(EWs, ax=plt.gca(), binnum=20, surrogate_EWs=None):
    lmin = min(EWs)
    lmax = max(EWs)
    print "\n Eigenvalue distribution:" \
          "\n\t EW_max = {:.2f}" \
          "\n\t EW_min = {:.2f}"\
          .format(lmax, lmin)

    edges = np.array([lmin + i*(lmax-lmin)/binnum for i in range(binnum+1)])
    EW_hist, edges = np.histogram(EWs, bins=edges, density=False)
    ax.bar(left=edges[:-1], height=EW_hist, width=edges[1]-edges[0], color='g')

    if surrogate_EWs != None:
        sEW_hist, __ = np.histogram(surrogate_EWs, bins=edges, density=False)
        ax.plot(edges[:-1] + (edges[1]-edges[0])/2., sEW_hist, color='r',
                alpha=.5)

        # y = 1
        # a = (1 - np.sqrt(y)) ** 2
        # b = (1 + np.sqrt(y)) ** 2
        # marchenko_pastur = lambda x: np.sqrt((x - a) * (b - x)) / (2 * np.pi * x * y)
        # marchenko_pastur = lambda x: np.sqrt(4*x - x**2) / (2*np.pi*x)
        # xaxis = np.linspace(0, int(math.ceil(edges[-1])), 50)
        # ax.plot(xaxis, [marchenko_pastur(x) for x in xaxis], color='r')
    else:
        # Reference to Random Correlation Matrix
        N = len(EWs)
        rand_matrix = np.random.rand(N, N) * 2. - 1
        corr_matrix = (rand_matrix + rand_matrix.T) / 2.
        surrogate_EWs, __ = np.linalg.eig(corr_matrix)
        # for i in range(N):
        #     corr_matrix[i,i] = 1.
        maxl = max([abs(min(surrogate_EWs)), abs(max(surrogate_EWs))])
        wigner_dist = lambda x: 2. / (np.pi*maxl**2) * np.sqrt(maxl**2-x**2)
        wigner_x = np.linspace(-maxl, maxl, 100, dtype=float)
        wigner_y = [wigner_dist(x) for x in wigner_x]
        ax.plot(wigner_x, wigner_y, color='r')

    ax.set_xlabel('EW')
    ax.set_ylabel('rel. occurence')
    print "\t{} eigenvalues are larger than the reference distribution \n"\
          .format(len(np.where(EWs > max(surrogate_EWs))[0]))
    return None


def redundancy(EWs):
    ### Measure of correlation of the matrix entries
    ### For 0 correlation sum(EW^2)=N -> phi=0
    ### For perfect correlation EW_1=N -> sum(EW^2)=N^2 -> phi=1
    N = len(EWs)
    phi = np.sqrt((np.sum(EWs**2)-N) / (N*(N-1)))
    print "\n Redundancy = {:.2f} \n".format(phi)
    return phi


def nbr_of_pcs(EWs, method='SCREE', alpha=.05, ax=plt.gca(), show_dist=True):
    EWs = np.sort(EWs)[::-1]
    total_v = np.sum(abs(EWs))

    if method == 'proportion':
        ### How many EWs can explain (1-alpha)% of the total variance
        pc_count = 0
        cum_var = 0
        while cum_var <= (1-alpha) * total_v:
            cum_var += EWs[pc_count]
            pc_count += 1

    elif method == 'res-variance':
        # ToDo: Can a reasonable residual variance be estimated from sample size?
        pc_count = 0

    elif method == 'broken-stick':
        ### Are EWs larger than the expected values of sorted random values
        N = len(EWs)
        series = [1. / (i+1) for i in range(N)]
        predictor = np.array([total_v / N * np.sum(series[k:])
                              for k in range(N)])
        pc_count = np.where(EWs < predictor)[0][0]

    elif method == "average-root":
        ### Are EWs larger than Tr(C)/N=1
        pc_count = np.where(EWs < 1)[0][0]

    elif method == "SCREE":
        ### Which EWs are above a dent in the EW spectra
        # line from first to last EW
        # y = a*x + b
        a = - EWs[0] / len(EWs)
        b = EWs[0]
        # orthonogal connection from current EW to line
        # y_s = a_s*x_s + b_s
        a_s = - 1. / a
        def cut(pc_count):
            b_s = EWs[pc_count] - a_s*pc_count
            x_s = (b_s-b) / (a-a_s)
            y_s = (a*b_s - a_s*b) / (a-a_s)
            # distance from EW to line
            return np.sqrt((pc_count-x_s)**2 + (EWs[pc_count]-y_s)**2)
        pc_count = 0
        prev_distance = 0
        current_distance = cut(pc_count)
        while current_distance >= prev_distance:
            pc_count += 1
            prev_distance = current_distance
            current_distance = cut(pc_count)

    if show_dist:
        mask = np.zeros(len(EWs), np.bool)
        mask[:pc_count] = 1
        ax.plot(np.arange(len(EWs)), abs(EWs)/total_v, color='r')
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=mask, color='r', alpha=.4)
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=np.logical_not(mask), color='r', alpha=.15)
        # ax.axhline(min(abs(PCs)) / total_v, 0, len(EWs), color='k', linestyle='--')
        ax.set_xlabel('EW#')
        ax.set_ylabel('rel. EW')

    print "\n Significance Test: " \
          "\n\t Method: {0} " \
          "\n\t {1} of {2} eigenvalues are significant"\
          .format(method, pc_count, len(EWs))

    print "\n Princial components:"
    print "\t"+"\n\t".join('{}: {:.2f}'
                           .format(*pc) for pc in enumerate(EWs[:pc_count])) \
          + "\n"
    return EWs[:pc_count]


def EV_angles(EVs1, EVs2, deg=True):
    ### EVs must be sorted; the angles are symetric in the EVs
    assert len(EVs1) == len(EVs2)

    N = len(EVs1)
    EV_angles = np.array([np.arccos(np.dot(ev1, ev2))
                          for (ev1, ev2) in zip(EVs1, EVs2)])

    M = np.ones((N, N))
    triu = np.triu_indices(N, 0)
    M[triu] = np.array([np.dot(EVs1[u1], EVs2[u2])
                        for (u1, u2) in zip(triu[0], triu[1])])
    triu = np.triu_indices(N, 1)
    M[triu[::-1]] = M[triu]
    space_angle = np.arccos(np.sqrt(np.linalg.det(np.dot(M, M))))

    if deg:
        EV_angles *= 180 / np.pi
        space_angle *= 180 / np.pi
        unit = "°"
    else:
        unit = " rad"

    print "\nAngles between the eigenvectors:" \
          + "\n\t" + "\n\t".join('{:.2f}{}'.format(a, unit) for a in EV_angles)\
          + "\n" \
          + "Angle between eigenspaces:" \
          + "\n\t {:.2f}{}".format(space_angle, unit)
    # ToDo: Understand the angle between spaces
    return EV_angles


def eigenvectors(matrix, ax=plt.gca()):
    EWs, EVs = np.linalg.eig(matrix)
    # ToDo: Sort the Correlation Matrix according to pairwise correlation strength
    # ToDo: by mapping the neurons along the largest eigenvectors
    return None

# ToDo: Write annotations for functions