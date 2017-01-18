"""
Toolbox for analyzing a correlation matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import contextlib
import pandas as pd
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


def corr_matrix(data1, data2, corr_type='pearson', sorted=True):
    # ToDo: Implement Correlation (pearson, spearman, ...?)
    return matrix


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
    sns.heatmap(matrix, ax=ax, cbar=True, xticklabels=labelnum,yticklabels=labelnum)
    # ToDo: offer 'sorted' option for estimated assemblies presentation
    return None


def eigenvalue_spectra(EWs, ax=plt.gca(), binnum=10):
    lmin = min(EWs)
    lmax = max(EWs)
    print "\lambda_max = {max}\n\lambda_min = {min}"\
          .format(max=lmax, min=lmin)

    edges = np.array([lmin + i*(lmax-lmin)/binnum for i in range(binnum+1)])
    EW_hist, edges = np.histogram(EWs, bins=edges, density=True)
    ax.bar(left=edges[:-1], height=EW_hist, width=edges[1]-edges[0])

    # Reference to Random Correlation Matrix
    N = len(EWs)
    rand_matrix = np.random.rand(N, N) * 2. - 1
    corr_matrix = (rand_matrix + rand_matrix.T) / 2.
    rand_EWs, __ = np.linalg.eig(corr_matrix)
    maxl = max([abs(min(rand_EWs)), abs(max(rand_EWs))])
    wigner_dist = lambda x: 2. / (np.pi*maxl**2) * np.sqrt(maxl**2-x**2)
    wigner_x = np.linspace(-maxl, maxl, 100, dtype=float)
    wigner_y = [wigner_dist(x) for x in wigner_x]
    ax.plot(wigner_x, wigner_y, color='r')
    # ToDo: Use custom surrogates as reference
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

def nbr_of_pcs(EWs, method='proportion', alpha=.05, ax=plt.gca(), show_dist=True):
    EWs = np.sort(EWs)[::-1]
    total_v = np.sum(abs(EWs))

    if method == 'proportion':
        absEWs = np.sort(abs(EWs))[::-1]
        i = 0
        cum_var = 0
        while cum_var <= (1-alpha) * total_v:
            cum_var += absEWs[i]
            i += 1
        mask = np.ones(len(EWs), np.bool)
        res_idx = np.where(abs(EWs)<=absEWs[i-1])[0]
        mask[res_idx] = 0
        PCs = EWs[mask]

    elif method == 'res-variance':
        # ToDo: Can a reasonable residual variance be estimated from sample size?
        mask = 0
        PCs = 0

    if show_dist:
        ax.plot(np.arange(len(EWs)), abs(EWs)/total_v, color='r')
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=mask, color='r', alpha=.4)
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=np.logical_not(mask), color='r', alpha=.15)
        ax.axhline(min(abs(PCs)) / total_v, 0, len(EWs), color='k', linestyle='--')
        ax.set_ylabel('|EW|/V')

    print "Significance Test: \n Method: {0} \n {1} of {2} are significant"\
          .format(method, len(PCs), len(EWs))
    # ToDo: Besseres Verstaendnis fuer die Rolle negativer Korrelationen
    return PCs

def EV_angles(EVs):
    # Do something
    return angles

def eigenvectors(matrix, ax=plt.gca()):
    EWs, EVs = np.linalg.eig(matrix)
    # ToDo: Sort the Correlation Matrix according to pairwise correlation strength
    # ToDo: by mapping the neurons along the largest eigenvectors
    return None

# ToDo: Write annotations for functions