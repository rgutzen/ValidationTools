"""
Toolbox for analyzing a correlation matrix.
Input is respectively a NxN matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
# Seaborn is not included in the HBP environment
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()

def plot_matrix(matrix, ax=plt.gca()):
    #plt.matshow(matrix, fignum=False)
    labelnum = matrix.shape[0]/10
    if labelnum == 1:
        labelnum = 2
    sns.heatmap(matrix, ax=ax, cbar=True, xticklabels=labelnum,
                yticklabels=labelnum)
    return None

def eigenvalue_spectra(matrix, ax=plt.gca(), binnum=10):
    EWs, __ = np.linalg.eig(matrix)
    lmin = min(EWs)
    lmax = max(EWs)
    print "\lambda_max = {max}\n\lambda_min = {min}"\
          .format(max=lmax, min=lmin)
    # ToDo: draw theoretical Marcenkov-Pastur distribution
    print (len(EWs)**0.25+1)**2
    print "\lambda_max estimate: {0}".format((1+np.sqrt(len(EWs)/1.38))**2)
    # 1.38 is an empirical value (good (~+-0.02) for N>~500)
    edges = np.array([lmin + i*(lmax-lmin)/binnum for i in range(binnum+1)])
    EW_hist, edges = np.histogram(EWs, bins=edges)
    x = edges[:-1] + (edges[1]-edges[0])/2.
    ax.bar(left=edges[:-1], height=EW_hist, width=edges[1]-edges[0])
    #sns.barplot(x, EW_hist, ax=ax, width=edges[1]-edges[0])
    # ToDo: compare to random matrix spectra as reference
    return None

def eigenvectors(matrix, ax=plt.gca()):
    EWs, EVs = np.linalg.eig(matrix)
    # ToDo: Sort the Correlation Matrix according to pairwise correlation strength
    # ToDo: by mapping the neurons along the largest eigenvectors
    return None