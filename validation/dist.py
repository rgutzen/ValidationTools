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
    if np.amax(sample1) >= np.amax(sample2):
        P, edges = np.histogram(sample1, bins=bins, density=True)
        Q, _____ = np.histogram(sample2, bins=edges, density=True)
    else:
        Q, edges = np.histogram(sample2, bins=bins, density=True)
        P, _____ = np.histogram(sample1, bins=edges, density=True)
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


def effect_size(sample1, sample2, true_var=None, comparison=np.mean,
                bias_correction=True, comparison_args={}, dof=0,
                log_scale=False):
    n1 = len(sample1)
    n2 = len(sample2)

    if true_var is not None:
        pooled_var = true_var
    else:
        summed_dev = (n1-1) * np.var(sample1, ddof=1) \
                     + (n2-1) * np.var(sample2, ddof=1)
        pooled_var = summed_dev / (n1 + n2 - 2)

    def measure(sample):
        n = len(sample)
        return comparison(sample, **comparison_args) * n/(n-dof)

    diff = np.abs(measure(sample1) - measure(sample2))

    es = diff / np.sqrt(pooled_var)

    if bias_correction:
        es *= (1 - 3 / (4*(n1 + n2 - 2) - 1))

    se = np.sqrt((n1 + n2) / (n1 * n2) + es**2 / (2*(n1 + n2 - 2)))

    ci_bound = 1.96 * se  # 95% confidence

    prec = int(np.ceil(np.abs(np.log10(ci_bound))))
    if ci_bound >= 1:
        prec = 0

    print "\n\tEffect Size = {:.{}f} (+- {:.{}f})"\
          .format(es, prec, ci_bound, prec)
    print "\tConfidence Interval (95%) = " \
          "[{:.{}f} .. {:.{}f}]\n".format(es-ci_bound, prec, es+ci_bound, prec)

    return es, ci_bound


def KL_test(sample1, sample2, bins=100, ax=None, xlabel='', mute=False,
            color=None):
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
        Parameter is forwarded to numpy.histogram(). Default = 100
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

        if np.amax(sample1) >= np.amax(sample2):
            P, edges = np.histogram(sample1, bins=bins, density=True)
            Q, _____ = np.histogram(sample2, bins=edges, density=True)
        else:
            Q, edges = np.histogram(sample2, bins=bins, density=True)
            P, _____ = np.histogram(sample1, bins=edges, density=True)
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
        print '\tD_KL(P||Q) = {:.4f}\n\tD_KL(Q||P) = {:.4f}\n' \
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
        if color is None:
            color = [sns.color_palette()[0], sns.color_palette()[1]]
        ax.plot(xvalues, P, lw=2, label='P', color=color[0])
        ax.plot(xvalues, Q, lw=2, label='Q', color=color[1])
        ax.set_xlim(xvalues[0], xvalues[-1])

    return D_KL, D_KL_as


def KS_test(data_sample_1, data_sample_2, ax=None, palette=None, mute=False,
            include_scatterplot=False):
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
    :return:
        D_Ks
        p-value
    """
    # Filtering out nans and infs
    init_length = [len(smpl) for smpl in [data_sample_1, data_sample_2]]
    sample1 = np.array(data_sample_1)[np.isfinite(data_sample_1)]
    sample2 = np.array(data_sample_2)[np.isfinite(data_sample_2)]

    if init_length[0]-len(sample1) or init_length[1]-len(sample2):
        print "Warning: {} non-finite elements of the given data samples " \
              "were filtered." \
              .format(sum(init_length)-sum([len(s) for s in [sample1, sample2]]))

    # Performing the KS-Test
    D_KS, pvalue = st.ks_2samp(sample1, sample2)
    if not mute:
        print "\n\033[4mKolmogorov-Smirnov-Distance\033[0m" \
            + "\n\tdatasizes: {} \t {}" \
              .format(len(sample1), len(sample2)) \
            + "\n\tD_KS = {:.2f} \t p value = {}\n" \
              .format(D_KS, to_precision(pvalue, 2))

    # Plotting a representation of the test
    if ax:
        ax.set_ylabel('CDF')
        ax.set_xlabel('Measured Parameter')
        if palette is None:
            palette = [sns.color_palette()[0], sns.color_palette()[1]]

        # plot cumulative distributions and scatterplot
        for i, sample in enumerate([sample1, sample2]):
            sorted_sample = np.sort(sample)
            sorted_sample = np.append(sorted_sample[0], sorted_sample)
            CDF = (np.arange(len(sample)+1)) / float(len(sample))
            ax.step(sorted_sample, CDF, where='post', color=palette[i])
            if include_scatterplot:
                ax.scatter(sorted_sample, [.99-i*.02]*len(sorted_sample),
                           color=palette[i], marker='D', linewidth=1)

        # calculate vertical distance
        N = len(sample1) + len(sample2)
        cdf_array = np.zeros((4, N))
        cdf_array[0] = np.append(sample1, sample2)
        cdf_array[1] = np.append(np.ones(len(sample1)), np.zeros(len(sample2)))
        cdf_array[2] = np.append(np.zeros(len(sample1)), np.ones(len(sample2)))
        sort_idx = np.argsort(cdf_array[0])
        cdf_array[0] = cdf_array[0][sort_idx]
        cdf_array[1] = np.cumsum(cdf_array[1][sort_idx]) / float(len(sample1))
        cdf_array[2] = np.cumsum(cdf_array[2][sort_idx]) / float(len(sample2))
        distance = cdf_array[1] - cdf_array[2]
        distance_plus = [d if d >= 0 else 0 for d in distance]
        distance_minus = [-d if d <= 0 else 0 for d in distance]

        # plot distance
        ax.plot(cdf_array[0], distance_plus, color=palette[0], alpha=.5)
        ax.fill_between(cdf_array[0], distance_plus, 0, color=palette[0], alpha=.5)
        ax.plot(cdf_array[0], distance_minus, color=palette[0], alpha=.5)
        ax.fill_between(cdf_array[0], distance_minus, 0, color=palette[1], alpha=.5)

        # plot max distance marker
        ax.axvline(cdf_array[0][np.argmax(abs(distance))],
                   color='.8', linestyle='--', linewidth=1.7)

        xlim_lower = min(min(sample1), min(sample2))
        xlim_upper = max(max(sample1), max(sample2))
        xlim_lower -= .03*(xlim_upper-xlim_lower)
        xlim_upper += .03*(xlim_upper-xlim_lower)
        ax.set_xlim(xlim_lower, xlim_upper)
        ax.set_ylim(0, 1)

    return D_KS, pvalue


def MWU_test(sample1, sample2, sample_names=None, linewidth=None,
             excl_nan=True, ax=None, mute=False, color=None):
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

    U, pvalue = st.mannwhitneyu(sample1, sample2, alternative=None)
    pvalue *= 2

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

        dataframe = pd.DataFrame({'Rank': ranks[0],
                                  'Group': ranks[1],
                                  'Kernel Density Estimate': np.zeros(N)})

        if color is None:
            color = [sns.color_palette()[0], sns.color_palette()[1]]

        sns.violinplot(data=dataframe, x='Kernel Density Estimate', y='Rank',
                       hue='Group', split=True, palette=color,
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
