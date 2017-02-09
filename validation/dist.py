"""
Toolbox for comparing the distribution of two sets of sample data.
Input is respectively two data arrays of arbitrary length.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
# # Seaborn is not included in the HBP environment
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()
sns.set_color_codes('colorblind')

def plot_comparison(dist1,dist2):
    binnum = len(dist1)
    bins = np.linspace(-binnum/2., binnum/2., binnum+1)
    fig = plt.figure('Distribution Comparison')
    plt.plot(bins[:-1], dist1, color='g', lw=2, label='dist1')
    plt.plot(bins[:-1], dist2, color='y', lw=2, label='dist2')
    D_KS, p_value = st.ks_2samp(dist1, dist2)
    plt.draw()
    # do sth.
    return None

def KS_test(sample1, sample2,  show=True, xlabel='Measured Parameter'):
    """Kolmogorov-Smirnov two-sample test

       Takes two sets of sample variables of possibly different size.

       Returns: D_KS, pvalue
                D_KS is the maximal distance between the two cumulative
                distribution functions. The pvalue describes the probability
                of finding D_KS value as larger or larger as observed under
                the assumption of the null hypothesis, which is the
                probability of the two underlying distributions are equal.
    """
    # filtering out nans
    sample1 = np.array(sample1)[np.isfinite(sample1)]
    sample2 = np.array(sample2)[np.isfinite(sample2)]

    D_KS, pvalue = st.ks_2samp(sample1, sample2)
    print 'KS-Test: \n', \
          'length 1 = {len1}; length 2 = {len2}; D_KS = {Dks}; p value = {p}' \
          .format(len1=len(sample1), len2=len(sample2), Dks=D_KS, p=pvalue)

    if show:
        fig = plt.figure('KS-Test')
        plt.ylabel('CDF')
        plt.xlabel(xlabel)
        ax = np.empty((2),dtype=object)
        ax[0] = fig.add_subplot(111)
        ax[1] = ax[0].twiny()
        color = ['r', 'g']
        for i, A in enumerate([sample1, sample2]):
            A_sorted = np.sort(A)
            A_sorted = A_sorted[np.isfinite(A_sorted)]
            CDF = np.array(range(len(A))) / float(len(A))
            ax[i].plot(A_sorted, CDF, color=color[i])
            ax[i].set_xticks(A_sorted)
            ax[i].set_xticklabels([''] * len(A_sorted))
            ax[i].tick_params(axis='x', length=15, color=color[i])
            plt.draw()
    return D_KS, pvalue

def KL_test(sample1, sample2, bins=10, excl_zeros=False, show=True, xlabel='a.u.'):
    """Kullback-Leibner Divergence D_KL(P||Q)

       Takes two normed discrete distributions. Or two data sets from which
       the distribution will be calculated for the number of bins (default 10)

       Q must not have zero values else the function will return inf.

       Returns: D_KL, D_KL_as
                D_KL can be interpreted as the amount of information lost
                when approximating P by Q. D_KL_as is D_KL(Q||P), as the
                divergence is asymmetric.
    """

    def isdist(sample, prec=0.001):
        if sum(sample) > (1-prec) and sum(sample) < (1+prec):
            return True
        return False

    print 'KL-Test:'
    if isdist(sample1) and isdist(sample2):
        print 'Interpreting input as distribution...'
        P = sample1
        Q = sample2
        edges = np.linspace(0, len(P), len(P)+1)
        edges -= len(P)/2.
    else:
        print 'Interpreting input as data sample...'
        # filtering out nans
        sample1 = np.array(sample1)[np.isfinite(sample1)]
        sample2 = np.array(sample2)[np.isfinite(sample2)]

        P, edges = np.histogram(sample1, bins=bins, density=True)
        Q, ____  = np.histogram(sample2, bins=edges, density=True)
        P *= np.diff(bins)[0]
        Q *= np.diff(bins)[0]

    if excl_zeros:
        _init_len = len(P)
        P = P[np.where(Q != 0.)[0]]
        Q = Q[np.where(Q != 0.)[0]]
        Q = Q[np.where(P != 0.)[0]]
        P = P[np.where(P != 0.)[0]]
        edges = np.linspace(0, len(P), len(P)+1)
        edges -= len(P)/2.
        _final_len = len(P)
        print '{0} zero values have been discarded.'\
              .format(_init_len - _final_len)
    else:
        if np.where(Q == 0.)[0].size:
            raise ValueError('Q must not have zero values!')
    D_KL = st.entropy(P, Q)
    D_KL_as = st.entropy(Q, P)

    print 'D_KL(P||Q) = {Dkl}; D_KL(Q||P) = {Dkla}' \
          .format(Dkl=D_KL, Dkla=D_KL_as)

    if show:
        fig = plt.figure('KL-Test')
        plt.ylabel('Probability Density')
        plt.xlabel(xlabel)
        ax = fig.add_subplot(111)
        xvalues = edges[:-1] + (edges[1]-edges[0])/2.
        diffy = P * np.log(P / Q.astype(float))
        fillx = np.append(np.append(xvalues[0],xvalues),xvalues[-1])
        filly = np.append(np.append(0.,diffy),0.)
        plt.fill(fillx, filly, color='LightGrey')
        ax.plot(xvalues, P, lw=2, label='P', color='r')
        ax.plot(xvalues, Q, lw=2, label='Q', color='g')
        plt.draw()
    return D_KL, D_KL_as


def MWW_test(sample1, sample2, excl_nan=True, show=True):
    """Mann-Whitney-Wilcoxon test

        Takes two sets of sample variables.

        Returns: U, pvalue
                U = U2 is the summed rank of the second sample.
                pvalue: probability to observe such a rank difference under
                the assumption of the null hypothesis.
    """
    if excl_nan:
        sample1 = np.array(sample1)[np.isfinite(sample1)]
        sample2 = np.array(sample2)[np.isfinite(sample2)]

    if len(sample1) < 20 or len(sample2) < 20:
        raise Warning('The sample size is too small. '
                      'The test might lose its validity!')

    U, pvalue = st.mannwhitneyu(sample1, sample2, alternative='two-sided')

    print 'MWW-Test: \n', \
          'length 1 = {len1}; length 2 = {len2}; U = {U}; p value = {p}' \
          .format(len1=len(sample1), len2=len(sample2), U=U, p=pvalue)

    if show:
        ranks1 = [[sample, 1] for sample in sample1]
        ranks2 = [[sample, 2] for sample in sample2]
        ranks = ranks1 + ranks2
        ranks = sorted(ranks, key=lambda e: e[0])
        ranks = [[i, ranks[i][1]] for i in range(len(ranks))]
        ranks1 = [rank[0] if rank[1]==1 else np.nan for rank in ranks]
        ranks2 = [rank[0] if rank[1] == 2 else np.nan for rank in ranks]
        fig = plt.figure('MWW-Test')
        plt.ylabel('Rank')
        ax = fig.add_subplot(111)
        color = ['r', 'g']
        for i, ranklist in enumerate([ranks1, ranks2]):
            for rank in ranklist:
                ax.axhline(rank, xmin=-1, xmax=1, lw=4, color=color[i])
        plt.draw()

    return U, pvalue
