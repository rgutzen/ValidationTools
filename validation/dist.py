"""
Toolbox for comparing the distribution of two sets of sample data.
Input is respectively two data arrays of arbitrary length.
"""

import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()
sns.set_color_codes('colorblind')

def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)


def show(sample1, sample2, bins=100, ax=plt.gca()):
    P, edges = np.histogram(sample1, bins=bins, density=True)
    Q, _____ = np.histogram(sample2, bins=edges, density=True)
    dx = np.diff(edges)[0]
    xvalues = edges[:-1] + dx/2.
    xvalues = np.append(np.append(xvalues[0]-dx, xvalues), xvalues[-1]+dx)
    P = np.append(np.append(0., P), 0.)
    Q = np.append(np.append(0., Q), 0.)
    ax.plot(xvalues, P)
    ax.plot(xvalues, Q)
    ax.set_xlim(xvalues[0], xvalues[-1])
    ax.set_ylim(0, max(max(P), max(Q)))
    return ax


def KL_test(sample1, sample2, bins=10, ax=None, xlabel='', mute=False):
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

    _init_len = len(P)
    Qnot0 = np.where(Q != 0.)[0]
    P_non0 = P[Qnot0]
    Q_non0 = Q[Qnot0]
    Pnot0 = np.where(P_non0 != 0.)[0]
    Q_non0 = Q_non0[Pnot0]
    P_non0 = P_non0[Pnot0]
    _final_len = len(P_non0)
    discard = _init_len - _final_len
    if not mute:
        print '\t{} zero value{} have been discarded.'\
              .format(discard, "s" if discard-1 else "")

    D_KL = st.entropy(P_non0, Q_non0, base=2)
    D_KL_as = st.entropy(Q_non0, P_non0, base=2)

    if not mute:
        print '\tD_KL(P||Q) = {:.2f}\n\tD_KL(Q||P) = {:.2f}\n' \
              .format(D_KL, D_KL_as)

    if ax:
        ax.set_ylabel('Probability Density')
        ax.set_xlabel(xlabel)
        xvalues = edges + dx/2.
        xvalues = np.append(np.append(xvalues[0]-dx, xvalues), xvalues[-1]+dx)

        def secure_log(E, D):
            log = np.zeros_like(E)
            i = 0
            for e, d in zip(E, D):
                if e == 0 or d == 0:
                    log[i] = 0.
                else:
                    log[i] = np.log(e/d)
                i += 1
            return log

        diffy = (P - Q) * secure_log(P, Q.astype(float))
        P = np.append(np.append(0, P), 0)
        Q = np.append(np.append(0, Q), 0)
        filly = np.append(np.append(0., diffy), 0.)
        ax.fill_between(xvalues, filly, 0,  color='0.8')
        ax.plot(xvalues, P, lw=2, label='P')
        ax.plot(xvalues, Q, lw=2, label='Q')
        ax.set_xlim(xvalues[0], xvalues[-1])

    return D_KL, D_KL_as


def KS_test(sample1, sample2, ax=None, xlabel='Measured Parameter', mute=False):
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

    :param sample1: numpy.ndarray
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
    if not mute:
        print "\n\033[4mKolmogorov-Smirnov-Distance\033[0m" \
            + "\n\tlength 1 = {} \t length 2 = {}" \
              .format(len(sample1), len(sample2)) \
            + "\n\tD_KS = {:.2f} \t p value = {}\n" \
              .format(D_KS, to_precision(pvalue, 2))

    if ax:
        ax.set_ylabel('CDF')
        ax.set_xlabel(xlabel)
        color = sns.color_palette()

        # print cumulative distributions and scatterplot
        for i, A in enumerate([sample1, sample2]):
            A_sorted = np.sort(A)
            A_sorted = np.append(A_sorted[0], A_sorted)
            CDF = (np.arange(len(A)+1)) / float(len(A))
            ax.step(A_sorted, CDF, where='post', color=color[i])
            ax.scatter(A_sorted, [.99-i*.02]*len(A_sorted),
                       color=color[i], marker='D')
        # calculate vertical distance
        N = len(sample1) + len(sample2)
        sample = np.zeros((4, N))
        sample[0] = np.append(sample1, sample2)
        sample[1] = np.append(np.ones(len(sample1)), np.zeros(len(sample2)))
        sample[2] = np.append(np.zeros(len(sample1)), np.ones(len(sample2)))
        sort_idx = np.argsort(sample[0])
        sample[0] = sample[0][sort_idx]
        sample[1] = np.cumsum(sample[1][sort_idx]) / float(len(sample1))
        sample[2] = np.cumsum(sample[2][sort_idx]) / float(len(sample2))
        distance = sample[1] - sample[2]
        distance_plus = [d if d >= 0 else 0 for d in distance]
        distance_minus = [-d if d <= 0 else 0 for d in distance]

        # plot distance
        ax.plot(sample[0], distance_plus, color=color[0], alpha=.5)
        ax.fill_between(sample[0], distance_plus, 0, color=color[0], alpha=.5)
        ax.plot(sample[0], distance_minus, color=color[0], alpha=.5)
        ax.fill_between(sample[0], distance_minus, 0, color=color[1], alpha=.5)

        # plot max distance marker
        ax.axvline(sample[0][np.argmax(abs(distance))],
                   color='.8', linestyle=':', linewidth=1.5)

        xlim_lower = min(min(sample1), min(sample2))
        xlim_upper = max(max(sample1), max(sample2))
        xlim_lower -= .03*(xlim_upper-xlim_lower)
        xlim_upper += .03*(xlim_upper-xlim_lower)
        ax.set_xlim(xlim_lower, xlim_upper)
        ax.set_ylim(0, 1)
    return D_KS, pvalue


def MWU_test(sample1, sample2, sample_names=None, linewidth=None,
             excl_nan=True, ax=None, mute=False):
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

    if not mute:
        print "\n\033[4mMann-Whitney-U-Test\033[0m" \
            + "\n\tlength 1 = {} \t length 2 = {}" \
              .format(len(sample1), len(sample2)) \
            + "\n\tU = {:.2f}   \t p value = {}" \
              .format(U, to_precision(pvalue, 2))

    if ax:
        if sample_names is None:
            sample_names = ['P', 'Q']
        N = len(sample1) + len(sample2)
        ranks = [[0]*2, [0]*N]
        # ranks = list(np.empty((2, len(sample1)+len(sample2))))
        ranks[0][:len(sample1)] = sample1
        ranks[1][:len(sample1)] = [sample_names[0]]*len(sample1)
        ranks[0][len(sample1):] = sample2
        ranks[1][len(sample1):] = [sample_names[1]]*len(sample2)
        ranks[0] = st.rankdata(ranks[0])

        dataframe = pd.DataFrame({'Ranks': ranks[0],
                                  'Group': ranks[1],
                                  'Kernel density estimate': np.zeros(N)})

        sns.violinplot(data=dataframe, x='Kernel density estimate', y='Ranks',
                       hue='Group', split=True, palette=sns.color_palette(),
                       inner='quartile', cut=0, ax=ax,
                       scale_hue=True, scale='width')

        # ax.set_ylabel('Rank')
        # ax.tick_params(axis='x', which='both', bottom='off', top='off',
        #                labelbottom='off')
        # ax.set_ylim(0, len(sample1)+len(sample2))
        # color = sns.color_palette()
        # bbox = ax.get_window_extent()
        # if linewidth is None:
        #     linewidth = bbox.height/(len(sample1)+len(sample2))
        #
        # for i in range(len(ranks[0])):
        #     ax.axhline(ranks[0, i], xmin=-1, xmax=1, lw=linewidth,
        #                color=color[int(ranks[1, i])])

    return U, pvalue
