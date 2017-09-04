"""
Toolbox for analyzing a correlation matrix.
"""

import numpy as np
from scipy.linalg import eigh
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy import stats as st
from scipy.misc import comb
from quantities import Hz, ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.set_color_codes('colorblind')
sns.despine()


def corr_matrix(spiketrains, binsize=2*ms, corr_type='pearson'):
    """
    Generates a correlation matrix from a set of spiketrains.

    :param spiketrains: list
        List of spiketrains
    :param binsize: quantity
        Binsize for temporal binning of the spiketrains
    :param corr_type: 'pearson', ... other to come?
    :return: numpy.ndarray
        Matrix of pairwise correlation coefficients
    """
    # ToDo: Implement Correlation (pearson, spearman, ...?)
    t_lims = [(st.t_start, st.t_stop) for st in spiketrains]
    tmin = min(t_lims, key=lambda f: f[0])[0]
    tmax = max(t_lims, key=lambda f: f[1])[1]
    if corr_type == 'pearson':
        binned_sts = BinnedSpikeTrain(spiketrains, binsize,
                                      t_start=tmin, t_stop=tmax)
        return np.nan_to_num(corrcoef(binned_sts))

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


def plot_matrix(matrix, ax=plt.gca(), remove_autocorr=False, labels=None,
                sort=False, **kwargs):
    """
    Plot correlation matrix as seaborn.heatmap

    :param matrix:
    :param ax:
    :param remove_autocorr:
    :param labels:
    :param sorted:
    :return:
    """
    pltmatrix = matrix[:][:]
    if sort:
        EWs, EVs = eigh(pltmatrix)
        _, order = detect_assemblies(EVs, EWs, detect_by='eigenvalues', sort=True)
        pltmatrix = pltmatrix[order, :][:, order]

    if labels is None:
        labels = matrix.shape[0]/10
        if labels == 1:
            labels = 2
    else:
        assert len(labels) == len(pltmatrix)

    if remove_autocorr:
        for i in range(len(pltmatrix)):
            pltmatrix[i, i] = 0

    sns.heatmap(pltmatrix, ax=ax, cbar=True,
                xticklabels=labels, yticklabels=labels, **kwargs)
    return None

def estimate_largest_eigenvalue(N, trials, t_stop, rate, bins):
    lmax = np.zeros(trials)

    for i in range(trials):
        spiketrains = np.array([HPP(rate=rate*Hz, t_stop=t_stop*ms)
                                for _ in range(N)])
        binned_sts = BinnedSpikeTrain(spiketrains, bins*ms,
                                      t_start=0*ms, t_stop=t_stop*ms)
        corr_matrix = corrcoef(binned_sts)
        EWs, __ = eigh(corr_matrix)
        lmax[i] = max(EWs)

    return np.mean(lmax), np.sqrt(np.var(lmax))


def eigenvalue_significance(EWs, ax=plt.gca(), bins=50, N=None, B=None,
                            spectra_method='SCREE', EW_max=None, ylim=None,
                            color=sns.color_palette()[0], mute=False):

    left, bottom, width, height = ax.get_position()._get_bounds()
    scaling = .55
    ax.set_position([left, bottom,
                     scaling * width, height])
    axhist = plt.axes([left + scaling * width, bottom,
                       (1-scaling) * width, height])
    axhist.yaxis.tick_right()
    axhist.get_xaxis().tick_bottom()
    axhist.yaxis.set_label_position("right")
    axhist.spines["left"].set_visible(False)
    axhist.spines["top"].set_visible(False)

    eigenvalue_spectra(EWs, ax=ax, method=spectra_method, alpha=.05,
                       color=color, mute=mute)

    ax.invert_xaxis()
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel('')
    ax.yaxis.set_ticks_position('none')
    ax.get_xaxis().tick_bottom()
    ax.yaxis.set_major_formatter(NullFormatter())

    if ylim is None:
        ylim = ax.get_ylim()
    else:
        ax.set_ylim(ylim)

    axhist.set_xlabel('Occurence')
    axhist.set_ylabel('Eigenvalue')

    if type(color) == int:
        color = sns.color_palette()[color]

    EW_hist, edges = np.histogram(EWs, bins=bins, density=False)
    axhist.barh(bottom=edges[:-1], width=EW_hist, height=edges[1]-edges[0],
                color=color, edgecolor='w')
    axhist.set_ylim(ylim)

    # wigner semi circle
    res = 100
    q = B / float(N)
    assert q >= 1
    x_min = (1 - np.sqrt(1. / q)) ** 2
    x_max = (1 + np.sqrt(1. / q)) ** 2

    def wigner_dist(x):
        return q / (2 * np.pi) * np.sqrt((x_max - x) * (x - x_min)) / x

    ev_values = np.linspace(x_min, x_max, res)
    dx = edges[1] - edges[0]
    wigner_values = [wigner_dist(ev) * N * dx for ev in ev_values]
    axhist.plot(wigner_values, ev_values, color='k',
                label='Marchenko-Pastur Distribution')
    tw_bound = x_max + N ** (-2 / 3.)
    axhist.axhline(tw_bound, color='k', linestyle=':',
                   label='Tracy-Widom Bound')

    axhist.legend()
    # handles, labels = axhist.get_legend_handles_labels()
    # ax.legend(handles, labels, loc='upper left')

    return edges, axhist, tw_bound



def eigenvalue_distribution(EWs, ax=plt.gca(), bins=20, reference_EWs=[],
                            reference_EW_max=None, wigner_params=None, color=0,
                            mute=False):
    """
    Plot histogram of the eigenvalue distribution in order to determine
    significant outliers.

    :param EWs: list, array
        eigenvalues
    :param ax: matplotlib.axis
    :param bins: int, list
        Number of bins or list of bin edges
    :param reference_EWs: list
        List of eigenvalues to which the provided eigenvalues should be
        compared. Those can be eigenvalues from surrogate data or theoretical
        predictions.
        ToDo:
        (If no reference is given the eigenvalue distribution is plotted with
        the expected spectral radius as reference)
    :param color:
    :return:
    """
    if not mute:
        print "\n\033[4mEigenvalue distribution:\033[0m" \
              "\n\tEW_max = {:.2f}" \
              "\n\tEW_min = {:.2f}"\
              .format(max(EWs), min(EWs))

    if type(color) == int:
        color = sns.color_palette()[color]

    EW_hist, edges = np.histogram(EWs, bins=bins, density=False)
    ax.bar(left=edges[:-1], height=EW_hist, width=edges[1]-edges[0],
           color=color, edgecolor='w')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Occurrence')
    ax.set_xlim(min(edges), max(edges))

    if len(reference_EWs):
        ref_EW_hist, __ = np.histogram(reference_EWs, bins=edges,
                                       density=False)
        dx = edges[1]-edges[0]
        ref_x = np.append(edges[0] - dx, edges)
        ref_x += dx / 2.
        ref_y = np.zeros_like(ref_x)
        ref_y[1:-1] = ref_EW_hist
        ax.plot(ref_x, ref_y, color='k')

    if reference_EW_max is not None:
        ax.axvline(reference_EW_max, color='k', linestyle='--', linewidth=2)
        ax.plot([reference_EW_max], [1], color='r')
        ax.set_xlim(0, max([max(edges), reference_EW_max]))

    if wigner_params is not None:
        res = 100
        N = wigner_params['N']
        B = wigner_params['B']
        q = B / float(N)
        assert q >= 1
        x_min = (1 - np.sqrt(1. / q)) ** 2
        x_max = (1 + np.sqrt(1. / q)) ** 2

        def wigner_dist(x):
            return q/(2*np.pi) * np.sqrt((x_max - x)*(x - x_min)) / x

        ev_values = np.linspace(x_min, x_max, res)
        dx = edges[1]-edges[0]
        wigner_values = [wigner_dist(ev)*N*dx for ev in ev_values]
        ax.plot(ev_values, wigner_values, color='k')
        ax.axvline(x_max + N**(-2/3), color='k', linestyle=':')

    return None


def redundancy(EWs, mute=False):
    """
    The redundancy is a measure of correlation in the eigenvalues.

    .math $$ \phi = \sqrt{\frac{\sum EW_i^2 - N}{N(N-1)}} $$

    For no redundancy all EW=1 -> sum(EW^2)=N -> phi=0
    For maximal redundancy EW_1=N -> sum(EW^2)=N^2 -> phi=1

    :param EWs:
    :param show:
    :return:
    """

    N = len(EWs)
    phi = np.sqrt((np.sum(EWs**2)-N) / (N*(N-1)))
    if not mute:
        print "\nRedundancy = {:.2f} \n".format(phi)

    return phi


def eigenvalue_spectra(EWs, method='SCREE', alpha=.05, ax=None, color='r',
                        mute=False, relative=False):
    """

    :param EWs:
    :param method: 'SCREE', 'proportion', 'broken-stick', 'average-root'
    :param alpha:
    :param ax:
    :param color:
    :return:
    """
    EWs = np.sort(EWs)[::-1]
    if relative:
        total_v = np.sum(abs(EWs))
    else:
        total_v = 1

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
        pc_count = len(np.where(EWs > 1)[0])

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
        while current_distance >= prev_distance \
          and pc_count < len(EWs)-1:
            pc_count += 1
            prev_distance = current_distance
            current_distance = cut(pc_count)

    if ax:
        def alpha(color_inst, a):
            return [el + (1. - el) * a for el in color_inst]

        mask = np.zeros(len(EWs), np.bool)
        mask[:pc_count] = 1
        ax.plot(np.arange(len(EWs)), abs(EWs)/total_v, color=color)
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=mask, color=alpha(color,.5))
        if pc_count - 1:
            mask[pc_count-1] = 0
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=np.logical_not(mask), color=alpha(color,.9))
        ax.set_xlabel('Eigenvalue #')
        ax.set_ylabel('rel. eigenvalue')
        ax.set_xlim(0, len(EWs))
        ax.set_ylim(0, np.ceil((max(EWs)/total_v)*10)/10.)

    if not mute:
        print "\n\033[4mSignificance Test:\033[0m" \
              "\n\tMethod: {0} " \
              "\n\t{1} of {2} eigenvalues are significant"\
              .format(method, pc_count, len(EWs))

        print "\n\033[4mPrincial components:\033[0m"
        print "\t"+"\n\t".join('{}: {:.2f}'
                               .format(*pc) for pc in enumerate(EWs[:pc_count])) \
              + "\n"

    return pc_count


def print_eigenvectors(EVs, EWs=[], pc_count=0,
                       colormap=[90,90,94,92,93,91,96,95,97]):
    """

    :param EVs:
    :param EWs:
    :param pc_count:
    :param colormap:
    :return:
    """
    colorcode = lambda v: int(abs(v) * len(colormap))

    print "\n\033[4mEW:\033[0m   \033[4mEigenvectors:\033[0m ",
    for c in colormap[1:]:
        print "\033[{}m \033[0m".format(c+10),
    print "\033[0m"

    if not pc_count:
        pc_count = len(EVs)

    if not len(EWs):
        EWs = np.arange(len(EVs))

    for i, EV in enumerate(EVs.T[:-pc_count:-1]):
        print "\033[47m{:3.1f}:\033[0m\t".format(EWs[-(i+1)]),
        # np.testing.assert_almost_equal(np.linalg.norm(EV), 1., decimal=7)
        for n_coord in EV:
            print "\033[{}m {:+.2f}\033[0m"\
                .format(colormap[colorcode(n_coord)], n_coord),
        print "\033[0m"
    return None


def plot_EVs(EVs, ax, color, hatch=None, ordered=False):
    if hatch is None:
        hatch = [''] * len(color)
    if ordered:
        vector_loads = np.sort(np.absolute(EVs.T[::-1]), axis=-1)
    else:
        vector_loads = np.absolute(EVs.T[::-1])
    for i, _ in enumerate(color):
        i = len(color) - i - 1
        ax.bar(np.arange(len(EVs.T[0]))+.5,vector_loads[i], 1., edgecolor='w',
                             label=r'$v_{}$'.format(i+1), color=color[i], hatch=hatch[i])
    ax.set_xlim(0,len(EVs.T[0])+1)
    ax.set_ylabel('Vector Load')
    ax.set_xlabel('Neuron ID')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], [r'$v_{}$'.format(j + 1) for j in range(len(color))])
    return None


def EV_angles(EVs1, EVs2, deg=True, mute=False, all_to_all=False):
    """
    Calculate the angles between the vectors EVs1_i and EVs2_i and the angle
    between their spanned eigenspaces.

    :param EVs1, EVs2: numpy.ndarray
        The eigenvectors must be presented column-wise in ascending order, just
        as returned by scipy.linalg.eigh().
    :param deg: Boolean
        If True angles are return in degrees, else in rad.
    :return:
        vector angles
        space angle
    """

    # Transform into descending array of the eigenvector arrays
    EVs1 = np.absolute(EVs1.T[::-1])
    EVs2 = np.absolute(EVs2.T[::-1])

    assert len(EVs1) == len(EVs2)

    if len(EVs1[0]) != len(EVs2[0]):
        min_length = min([len(EVs1[0]), len(EVs2[0])])
        print "Warning: Samples are not the same size! [{} / {}] " \
              "\n The vectors will be cut to length {} "\
              .format(len(EVs1[0]), len(EVs2[0]), min_length)
        EVs1 = EVs1[:, :min_length]
        EVs2 = EVs2[:, :min_length]

    for EVs in [EVs1, EVs2]:
        for EV in EVs:
            if abs(np.linalg.norm(EV) - 1) > 0.0001:
                print "Warning: Eigenvector norm deviates from 1: ({:.7f})"\
                      .format(np.linalg.norm(EV))

    M = np.dot(EVs1, EVs2.T)

    if len(M) == 1:
        vector_angles = np.arccos(M[0])
    else:
        if all_to_all:
            vector_angles = np.arccos(M.flatten())
            # ?
        else:
            vector_angles = np.arccos(np.diag(M))

    vector_angles[np.where(np.isnan(vector_angles))[0]] = 0.

    space_angle = np.arccos(np.sqrt(np.linalg.det(np.dot(M, M.T))))

    if deg:
        vector_angles *= 180. / np.pi
        space_angle *= 180. / np.pi
        unit = r"$^\circ$"
    else:
        unit = " rad"

    if not mute:
        print "\n\033[4mAngles between the eigenvectors\033[0m" \
              + "\n\t" + "\n\t".join('{:.2f}{}'.format(a, unit)
                                     for a in vector_angles)\
              + "\n\n" \
              + "\033[4mAngle between eigenspaces\033[0m" \
              + "\n\t{:.2f}{}".format(space_angle, unit)
    # ToDo: Understand the angle between spaces

    return vector_angles, space_angle


def angle_significance(angles, dim=100, s=.0001, sig_level=.01, res=10**7,
                       rand_angles=None, bins=100., abs=True, ax=None,
                       mute=False):
    if type(angles) != list:
        angles = [angles]

    N_angle = len(angles)

    if rand_angles is None:
        # Generate random angles
        N_rand = int(.5 * (1 + np.sqrt(8 * res + 1)))
        vectors = np.random.normal(size=(N_rand, dim))
        if abs:
            vectors = np.absolute(vectors)
        vectorsT = vectors.T / np.linalg.norm(vectors, axis=1)
        vectors = vectorsT.T
        dotprods = np.dot(vectors, vectorsT)[np.triu_indices(n=N_rand, m=N_rand, k=1)]
        rand_angles = np.arccos(dotprods)
        N_rand_angles =  N_rand * (N_rand - 1) / 2.
    else:
        N_rand_angles = len(rand_angles)

    if abs:
        max_angle = np.pi / 2.
    else:
        max_angle = np.pi

    threshold_angle = np.sort(rand_angles)[int(N_rand_angles*s)]
    x = len(np.where(np.array(angles) < threshold_angle)[0])

    # n = int(N_rand_angles*s) + dim # max number of 'small' angles (formally N_angle)
    n = N_angle
    comb_prob = np.array([s**j * (1. - s)**(n - j) * comb(n, j)
                         for j in range(x)])

    comb_prob = comb_prob[np.where(np.isfinite(comb_prob))[0]]
    p_diff = 1. - sum(comb_prob)

    if not mute:
        if len(comb_prob) < x:
            print "Warning: The probability could not be adequately calculated! " \
                  "Try reducing the similarity quantile s. " \
                  "(Encountering {} non-finite numbers)".format(
                x - len(comb_prob))
        print "{} / {} angles in {}%-quantile \n"\
              .format(x, N_angle, s*100.)\
            + "p = {}\n"\
              .format(p_diff)\
            + "{} {} {} ==> {} "\
              .format(sig_level, '<' if sig_level < p_diff else '>', 'p',
                      'Similar' if p_diff < sig_level else 'Different')\
            + "\n(Distribution approximated by {:.0f} sampled angles)"\
              .format(N_rand_angles)

    if ax is not None:
        edges = np.linspace(0, max_angle, bins)
        hist_rand, _ = np.histogram(rand_angles, bins=edges, density=True)
        ax.bar(edges[:-1], hist_rand, np.diff(edges) * .9,
               color=sns.color_palette()[0], edgecolor='w')
        # hist_evs, _ = np.histogram(angles, bins=edges, density=True)
        # ax.bar(edges[:-1], hist_evs, np.diff(edges) * .9,
        #        edgecolor=sns.color_palette()[1], fill=False, lw=2)
        ax.set_xlim(0, max_angle)
        ax.set_xticks(np.arange(0, max_angle + .125*np.pi, .125*np.pi))
        ticklabels = ['', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$',
                      r'$\frac{3}{8}\pi$', r'$\frac{\pi}{2}$',
                      r'$\frac{5}{8}\pi$', r'$\frac{3}{4}\pi$',
                      r'$\frac{7}{8}\pi$', r'$\pi$']
        ax.set_xticklabels(ticklabels[: 5 if abs else 9])
        ax.set_xlabel(r'Plane Angle in $\mathbf{R}$'
                      + r'{}'.format('$_+$' if abs else '')
                      + r'$^{}$'.format('{'+str(dim)+'}'))
        ax.set_ylabel('Angle Density')
    return p_diff, rand_angles


def detect_assemblies(EVs, EWs, detect_by='eigenvalue', mute=False, EW_lim=2,
                      jupyter=False, sort=False):
    """

    :param EVs:
    :param EWs:
    :param detect_by: 'eigenvalue', int, float
        'eigenvalue'    The respective sizes of the assemblies are estimated
                        by the next larger int of the eigenvalue.
        int             a direct estimate of the assembly size
        float           a threshold for the vector elements
    :param show:
    :param EW_lim:
    :param jupyter:
    :param sort:
        For EW > EW_lim the significant contributions according to the
        detection method in the eigenvectors determine the most significant
        and therefore top ordered neurons.
        For all other EW < EW_lim the neurons are ordered by their above chance
        contribution (> 1/sqrt(N)) to the eigenvectors ordered by their eigen-
        values.
        In case their are still neurons left they are appended in order of
        their id.
    :return:
    """

    EVs = np.absolute(EVs.T[::-1])
    EWs = EWs[::-1]
    if type(detect_by) == float or type(detect_by) == int:
        th = detect_by
    else:
        th = 0

    def _get_ids(EV, th):
        if th:
            if th < 1:
                ids = np.where(EV > th)[0]
                size = len(ids)
            elif th >= 1:
                size = int(th)
                ids = np.argpartition(EV, -size)[-size:]
        else:
            size = int(np.ceil(EWs[i]))
            ids = np.argpartition(EV, -size)[-size:]

        return ids[np.argsort(EVs[i][ids])][::-1]


    i = 0
    n_ids = []
    relevant_EVs = []

    while EWs[i] > EW_lim:

        n_ids += [_get_ids(EVs[i], th)]

        if len(n_ids[i]-1):
            cur_rel_EVs = EVs[i][n_ids[i]]
        else:
            cur_rel_EVs = np.array(EVs[i][n_ids[i]])

        relevant_EVs += [cur_rel_EVs]

        if not mute and EWs[i] > EW_lim:
            print "\033[4mAssembly {}, eigenvalue {:.2f}, size {}\033[0m"\
                  .format(i+1, EWs[i], len(n_ids[i]))
            print "Neuron:\t",
            for n in n_ids[i]:
                print "{:2.0f}{}\t".format(n, "" if jupyter else "\t"),
            print "\tNorm"
            print "Load:\t",
            for n in n_ids[i]:
                print "{:.2f}\t".format(EVs[i][n]),
            print "\t{:.2f}\n".format(np.linalg.norm(EVs[i][n_ids[i]]))
        i += 1

    for ev in EVs:
        th = 1/np.sqrt(len(ev))
        n_ids += [_get_ids(ev, th)]

    if not len(relevant_EVs):
        max_EV = EVs[np.where(EWs == max(EWs))[0][0]]
        relevant_EVs = [[max(max_EV)]]

    if sort:
        st_num_list = []
        for ids in n_ids:
            for id in ids:
                if id not in st_num_list:
                    st_num_list += [id]

        for id in np.arange(len(EVs[0])):
            if id not in st_num_list:
                st_num_list += [id]
        return relevant_EVs, st_num_list

    return relevant_EVs
