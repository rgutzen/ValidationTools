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
    with printoptions(precision=3, suppress=True):
        print pd.DataFrame(diag_matrix[:6,:6]).round(3)
        print pd.DataFrame(np.diag(EWs[:6])).round(3)
    return diag_matrix

def plot_matrix(matrix, ax=plt.gca()):
    #plt.matshow(matrix, fignum=False)
    labelnum = matrix.shape[0]/10
    if labelnum == 1:
        labelnum = 2
    sns.heatmap(matrix, ax=ax, cbar=True, xticklabels=labelnum,
                yticklabels=labelnum)
    return None

def eigenvalue_spectra(EWs, ax=plt.gca(), binnum=10):
    lmin = min(EWs)
    lmax = max(EWs)
    print "\lambda_max = {max}\n\lambda_min = {min}"\
          .format(max=lmax, min=lmin)
    print "\lambda_max estimate: {0}".format((1+np.sqrt(len(EWs)/1.38))**2)
    # 1.38 is an empirical value (good (~+-0.02) for N>~500)
    edges = np.array([lmin + i*(lmax-lmin)/binnum for i in range(binnum+1)])
    EW_hist, edges = np.histogram(EWs, bins=edges, density=True)
    ax.bar(left=edges[:-1], height=EW_hist, width=edges[1]-edges[0])

    # Reference to Random Correlation Matrix
    N = len(EWs)
    rand_matrix = np.random.normal(size=(N, N))
    corr_matrix = (rand_matrix + rand_matrix.T) / 2.
    rand_EWs, __ = np.linalg.eig(corr_matrix)
    maxl = max([abs(min(rand_EWs)), abs(max(rand_EWs))])
    wigner_dist = lambda x: 2. / (np.pi*maxl**2) * np.sqrt(maxl**2-x**2)
    dx = 2*maxl/100.
    wigner_x = np.linspace(-maxl, maxl, 100, dtype=float)
    wigner_y = [wigner_dist(x) for x in wigner_x]
    ax.plot(wigner_x, wigner_y, color='r')
    # ToDo: Use custom surrogates as reference
    return None

def total_variance(matrix):
    # Do something
    return Det, Tr, Tsq

def nbr_of_pcs(EWs, alpha=.05):
    # Do something
    return sig_nbr

def EV_angles(EVs):
    # Do something
    return angles

def eigenvectors(matrix, ax=plt.gca()):
    EWs, EVs = np.linalg.eig(matrix)
    # ToDo: Sort the Correlation Matrix according to pairwise correlation strength
    # ToDo: by mapping the neurons along the largest eigenvectors
    return None