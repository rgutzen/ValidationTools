"""
Toolbox for comparing the distribution of two sets of sample data.
Input is respectively two data arrays of arbitrary length.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats as st
import math as m
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
    plt.draw()
    return None


def KL_test(sample1, sample2, bins=10, excl_zeros=True, ax=None, xlabel='a.u.'):
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
        if (1-prec) < sum(sample) < (1+prec):
            return True
        return False

    print '\n\033[4mKullback-Leidler-Divergence\033[0m'
    if isdist(sample1) and isdist(sample2):
        print '\tInterpreting input as distribution...'
        P = sample1
        Q = sample2
        edges = np.linspace(0, len(P), len(P)+1)
        edges -= len(P)/2.
        dx = np.diff(edges)[0]

    else:
        print '\tInterpreting input as data sample...'
        # filtering out nans
        sample1 = np.array(sample1)[np.isfinite(sample1)]
        sample2 = np.array(sample2)[np.isfinite(sample2)]

        P, edges = np.histogram(sample1, bins=bins, density=True)
        Q, _____ = np.histogram(sample2, bins=edges, density=True)
        dx = np.diff(edges)[0]
        edges = edges[:-1]
        P *= dx
        Q *= dx

    if excl_zeros:
        _init_len = len(P)
        Qnot0 = np.where(Q != 0.)[0]
        P = P[Qnot0]
        Q = Q[Qnot0]
        edges = edges[Qnot0]
        Pnot0 = np.where(P != 0.)[0]
        Q = Q[Pnot0]
        P = P[Pnot0]
        edges = edges[Pnot0]
        _final_len = len(P)
        discard = _init_len - _final_len
        print '\t{} zero value{} have been discarded.'\
              .format(discard, "s" if discard-1 else "")
    else:
        if np.where(Q == 0.)[0].size:
            raise ValueError('Q must not have zero values!')
    D_KL = st.entropy(P, Q, base=2)
    D_KL_as = st.entropy(Q, P, base=2)

    print '\tD_KL(P||Q) = {:.2f}\n\tD_KL(Q||P) = {:.2f}\n' \
          .format(D_KL, D_KL_as)

    if ax:
        ax.set_ylabel('Probability Density')
        ax.set_xlabel(xlabel)
        xvalues = edges + dx/2.
        xvalues = np.append(np.append(edges[0]-dx, xvalues), edges[-1]+dx)
        diffy = P * np.log(P / Q.astype(float))
        P = np.append(np.append(0, P), 0)
        Q = np.append(np.append(0, Q), 0)
        filly = np.append(np.append(0., diffy), 0.)
        xticks = np.arange(len(P))
        ax.fill_between(xticks, filly, 0,  color='LightGrey')
        ax.plot(xticks, P, lw=2, label='P', color='r')
        ax.plot(xticks, Q, lw=2, label='Q', color='g')
        ax.set_xlim(xticks[0], xticks[-1])
        ax.set_xticklabels(["{:.2f}".format(xv) for xv in xvalues])
    return D_KL, D_KL_as


def KS_test(sample1, sample2, ax=None, xlabel='Measured Parameter'):
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
    print "\n\033[4mKolmogorov-Smirnov-Distance\033[0m" \
        + "\n\tlength 1 = {} \t length 2 = {}" \
          .format(len(sample1), len(sample2)) \
        + "\n\tD_KS = {:.2f} \t p value = {:.2f}\n" \
          .format(D_KS, pvalue)

    if ax:
        ax.set_ylabel('CDF')
        ax.set_xlabel(xlabel)
        color = ['r', 'g']
        for i, A in enumerate([sample1, sample2]):
            A_sorted = np.sort(A)
            A_sorted = np.append(A_sorted[0], A_sorted)
            CDF = (np.arange(len(A)+1)) / float(len(A))
            ax.step(A_sorted, CDF, where='post', color=color[i])
            ax.scatter(A_sorted, [i+.01-i*.02]*len(A_sorted),
                       color=color[i], marker='D')
        xlim_lower = min(min(sample1), min(sample2))
        xlim_upper = max(max(sample1), max(sample2))
        xlim_lower -= .03*(xlim_upper-xlim_lower)
        xlim_upper += .03*(xlim_upper-xlim_lower)
        ax.set_xlim(xlim_lower, xlim_upper)
        ax.set_ylim(0, 1)
    return D_KS, pvalue


def MWU_test(sample1, sample2, excl_nan=True, ax=None):
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

    # if len(sample1) < 20 or len(sample2) < 20:
    #     raise Warning('The sample size is too small. '
    #                   'The test might lose its validity!')

    U, pvalue = st.mannwhitneyu(sample1, sample2, alternative='two-sided')

    print "\n\033[4mMann-Whitney-U-Test\033[0m" \
        + "\n\tlength 1 = {} \t length 2 = {}" \
          .format(len(sample1), len(sample2)) \
        + "\n\tU = {:.2f}   \t p value = {:.2f}" \
          .format(U, pvalue)

    if ax:
        ranks = np.empty((2, len(sample1)+len(sample2)))
        ranks[0, :len(sample1)] = sample1
        ranks[1, :len(sample1)] = 0
        ranks[0, len(sample1):] = sample2
        ranks[1, len(sample1):] = 1
        ranks[0] = st.rankdata(ranks[0])

        ax.set_ylabel('Rank')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.set_ylim(0, len(sample1)+len(sample2))
        color = ['r', 'g']
        bbox = ax.get_window_extent()
        linewidth = bbox.height/(len(sample1)+len(sample2))

        for i in range(len(ranks[0])):
            ax.axhline(ranks[0, i], xmin=-1, xmax=1, lw=linewidth,
                       color=color[int(ranks[1, i])])

    return U, pvalue
