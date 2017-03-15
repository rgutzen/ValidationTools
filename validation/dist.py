"""
Toolbox for comparing the distribution of two sets of sample data.
Input is respectively two data arrays of arbitrary length.
"""

import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()
sns.set_color_codes('colorblind')

def show(sample1, sample2, bins=100, ax=plt.gca()):
    P, edges = np.histogram(sample1, bins=bins, density=True)
    Q, _____ = np.histogram(sample2, bins=edges, density=True)
    dx = np.diff(edges)[0]
    xvalues = edges[:-1] + dx/2.
    ax.plot(xvalues, P)
    ax.plot(xvalues, Q)
    ax.set_xlim(xvalues[0], xvalues[-1])
    ax.set_ylim(0, max(max(P), max(Q)))
    return ax

def KL_test(sample1, sample2, bins=10, excl_zeros=True, ax=None,
            xlabel='a.u.', mute=False):
    """
    Kullback-Leibner Divergence D_KL(P||Q)

    Calculates the difference of two sampled distributions P and Q in form of
    an entropy measure. The D_KL measure is effectively the difference of the
    cross-entropy of the of both distribution P,Q and the entropy of P.

    . math $$ D\mathrm{KL}(P||Q) =\sum{i} P(i) \log_2 \frac{P(i)}{Q(i)}
                                 = H(P,Q) - H(P) $$

    D_KL can be interpreted as the amount of information lost when
    approximating P by Q.
    D_KL(P||Q) =/= D_KL(Q||P), therefore both values are returned.

    :param sample1: array, list
        Can either be normed distribution or sample of values from which a
        histogram distributions with *bins* is calculated. (= P)
    :param sample2: array, list
        Can either be normed distribution or sample of values from which a
        histogram distributions with *bins* is calculated. (= Q)
    :param bins: int, array of edges
        Parameter is forwarded to numpy.histogram(). Default = 10
    :param excl_zeros: Bool
        D_KL can't be calculated if there are zero values in the distribution.
        Therefore all zeros are omitted when excl_zeros is True (default).
    :param ax: matplotlib.axis
        An visual representation of the distributions and their divergence is
        given onto the matplotlib axis when provided
    :param xlabel: string
        Label string for plot
    :return:
        D_KL(P||Q)
        D_KL(Q||P)
    """

    def isdist(sample, prec=0.001):
        if (1-prec) < sum(sample) < (1+prec):
            return True
        return False

    if not mute:
        print '\n\033[4mKullback-Leidler-Divergence\033[0m'
    if isdist(sample1) and isdist(sample2):
        if not mute:
            print '\tInterpreting input as distribution...'
        P = sample1
        Q = sample2
        edges = np.linspace(0, len(P), len(P)+1)
        edges -= len(P)/2.
        dx = np.diff(edges)[0]

    else:
        if not mute:
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
        if not mute:
            print '\t{} zero value{} have been discarded.'\
                  .format(discard, "s" if discard-1 else "")
    else:
        if np.where(Q == 0.)[0].size:
            raise ValueError('Q must not have zero values!')
    D_KL = st.entropy(P, Q, base=2)
    D_KL_as = st.entropy(Q, P, base=2)

    if not mute:
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
    """
    Kolmogorov-Smirnov-Distance D_KS

    .math $$ D_\mathrm{KS} = \sup | \hat{P}(x) - \hat{Q}(x) | $$

    The KS-Distance measures the maximal vertical distance of the cumulative
    distributions $\hat{P}$ and $\hat{Q}$. This is a sensitive tool for
    detecting differences in mean, variance or distribution type.

    The null hypothesis that the underlying distributions are identical is
    rejected when the D_KS statistic is larger than a critical value or
    equivalently when the correponding p-value is less than the signficance
    level.

    :param sample1: array, list
        Sample of the parmeter of interest.
    :param sample2: array, list
        Sample of the parmeter of interest.
    :param ax: matplotlib.axis
        An visual representation of the distributions and their divergence is
        given onto the matplotlib axis when provided.
    :param xlabel: string
        Label string for plot
    :return:
        D_Ks
        p-value
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
    """
    Mann-Whitney-U test

    .math $$ U_i = R_i - \frac{n_i(n_i + 1)}{2}\\ U = min(U_1,U_2) $$

    With the rank sum R and the sample size n_i.

    The Mann-Whitney U is a rank statistic which test the null hypothesis
    that a random value of sample 1 is equally likely to be larger or a smaller
    value than a randomly chosen value of sample 2.

    The U_i statistic is in the range of [0,n_1 n_2],
    and the U=min(U_1,U_2) statistic is in the range of [0,n_1*n_2/2].

    For sample sizes >20, U follows approximately a normal distribution.
    With this assumption a p-value can be inferred. The null hypothesis is
    consequently rejected when the p-value is less than the significance level.

        Returns: U, pvalue
                U = U2 is the summed rank of the second sample.
                pvalue: probability to observe such a rank difference under
                the assumption of the null hypothesis.
    :param sample1: array, list
        Sample of the parmeter of interest.
    :param sample2: array, list
        Sample of the parmeter of interest.
    :param excl_nan: Bool
        When True it excludes all non finite values from the sample which
        influences the U statistic because of the change in sample size.
    :param ax: matplotlib.axis
        An visual representation of the distributions and their divergence is
        given onto the matplotlib axis when provided.
    :return:
        U
        p-value
    """
    if excl_nan:
        sample1 = np.array(sample1)[np.isfinite(sample1)]
        sample2 = np.array(sample2)[np.isfinite(sample2)]

    if len(sample1) < 20 or len(sample2) < 20:
        raise Warning('The sample size is too small. '
                      'The test might lose its validity!')

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
