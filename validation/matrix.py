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
    if corr_type=='pearson':
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


def eigenvalue_distribution(EWs, ax=plt.gca(), binnum=15, surrogate_EWs=None):
    lmin = min(EWs)
    lmax = max(EWs)
    print "\lambda_max = {max}\n\lambda_min = {min}"\
          .format(max=lmax, min=lmin)

    edges = np.array([lmin + i*(lmax-lmin)/binnum for i in range(binnum+1)])
    EW_hist, edges = np.histogram(EWs, bins=edges, density=False)
    ax.bar(left=edges[:-1], height=EW_hist, width=edges[1]-edges[0], color='g')

    if surrogate_EWs != None:
        # ToDo: Use custom surrogates as reference
        sEW_hist, __ = np.histogram(surrogate_EWs, bins=edges, density=False)
        ax.plot(edges[:-1] + (edges[1]-edges[0])/2., sEW_hist, color='r',
                alpha=.5)
        # ax.bar(left=edges[:-1], height=sEW_hist, width=edges[1]-edges[0],
        #        alpha=.3, color='r')
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
        rand_EWs, __ = np.linalg.eig(corr_matrix)
        # for i in range(N):
        #     corr_matrix[i,i] = 1.
        maxl = max([abs(min(rand_EWs)), abs(max(rand_EWs))])
        wigner_dist = lambda x: 2. / (np.pi*maxl**2) * np.sqrt(maxl**2-x**2)
        wigner_x = np.linspace(-maxl, maxl, 100, dtype=float)
        wigner_y = [wigner_dist(x) for x in wigner_x]
        ax.plot(wigner_x, wigner_y, color='r')
    return None


def redundancy(EWs):
    N = len(EWs)
    phi = np.sqrt((np.sum(EWs**2)-N) / (N*(N-1)))
    # sum(EW**2) ranges between N for no correlation to N^2 for perfect correlation
    # For uniform random phi ~ 0.4
    print "sum(EW^2) = {0} \n Redundancy = {1}".format(np.sum(EWs**2), phi)
    return phi

def total_variance(matrix):
    # Do something
    return Det, Tr, Tsq

def nbr_of_pcs(EWs, method='SCREE', alpha=.05, ax=plt.gca(), show_dist=True):
    EWs = np.sort(EWs)[::-1]
    total_v = np.sum(abs(EWs))

    if method == 'proportion':
        pc_count = 0
        cum_var = 0
        while cum_var <= (1-alpha) * total_v:
            cum_var += EWs[pc_count]
            pc_count += 1

    elif method == 'res-variance':
        # ToDo: Can a reasonable residual variance be estimated from sample size?
        pc_count = 0

    elif method == 'broken-stick':
        N = len(EWs)
        series = [1. / (i+1) for i in range(N)]
        predictor = np.array([total_v / N * np.sum(series[k:])
                              for k in range(N)])
        pc_count = np.where(EWs < predictor)[0][0]

    elif method == "average-root":
        pc_count = np.where(EWs < 1)[0][0]

    elif method == "SCREE":
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
        mask = np.ones(len(EWs), np.bool)
        mask[pc_count:] = 0
        ax.plot(np.arange(len(EWs)), abs(EWs)/total_v, color='r')
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=mask, color='r', alpha=.4)
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=np.logical_not(mask), color='r', alpha=.15)
        # ax.axhline(min(abs(PCs)) / total_v, 0, len(EWs), color='k', linestyle='--')
        ax.set_ylabel('|EW|/V')

    print "Significance Test: \n Method: {0} \n {1} of {2} are significant"\
          .format(method, pc_count-1, len(EWs))
    return EWs[:pc_count]

def EV_angles(EVs):
    # Do something
    return angles

def eigenvectors(matrix, ax=plt.gca()):
    EWs, EVs = np.linalg.eig(matrix)
    # ToDo: Sort the Correlation Matrix according to pairwise correlation strength
    # ToDo: by mapping the neurons along the largest eigenvectors
    return None

# ToDo: Write annotations for functions