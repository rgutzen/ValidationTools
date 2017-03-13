# -*- coding: utf-8 -*-
"""
Toolbox for analyzing a correlation matrix.
"""

import numpy as np
from scipy.linalg import eigh
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from cycler import cycler
from scipy import stats as st
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


def plot_matrix(matrix, ax=plt.gca(), remove_autocorr=False, labels=None,
                sorted=False):
    """
    Plot correlation matrix as seaborn.heatmap

    :param matrix:
    :param ax:
    :param remove_autocorr:
    :param labels:
    :param sorted:
    :return:
    """
    if sorted:
        EWs, EVs = eigh(corr_matrix)
        labels = detect_assemblies(EVs, EWs, detect_by='eigenvalues', sort=True)
        matrix = matrix[labels, :][:, labels]

    if labels is None:
        labels = matrix.shape[0]/10
        if labels == 1:
            labels = 2
    else:
        assert len(labels) == len(matrix)

    if remove_autocorr:
        for i in range(len(matrix)):
            matrix[i, i] = 0
    sns.heatmap(matrix, ax=ax, cbar=True,
                xticklabels=labels, yticklabels=labels)
    if remove_autocorr:
        for i in range(len(matrix)):
            matrix[i, i] = 1

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



def eigenvalue_distribution(EWs, ax=plt.gca(), bins=20, reference_EWs=[],
                            reference_EW_max=None, color=0):
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

    print "\n\033[4mEigenvalue distribution:\033[0m" \
          "\n\tEW_max = {:.2f}" \
          "\n\tEW_min = {:.2f}"\
          .format(max(EWs), min(EWs))

    if type(color) == int:
        color = sns.color_palette()[color]

    EW_hist, edges = np.histogram(EWs, bins=bins, density=False)
    ax.bar(left=edges[:-1], height=EW_hist, width=edges[1]-edges[0],
           color=color)
    ax.set_xlabel('EW')
    ax.set_ylabel('Occurrence')
    ax.set_xlim(0, max(edges))

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

    #     # Reference to Random Correlation Matrix
    #     N = len(EWs)
    #     rand_matrix = np.random.rand(N, N) * 2. - 1
    #     corr_matrix = (rand_matrix + rand_matrix.T) / 2.
    #     surrogate_EWs, __ = np.linalg.eig(corr_matrix)
    #     # for i in range(N):
    #     #     corr_matrix[i,i] = 1.
    #     maxl = max([abs(min(surrogate_EWs)), abs(max(surrogate_EWs))])
    #     wigner_dist = lambda x: 2. / (np.pi*maxl**2) * np.sqrt(maxl**2-x**2)
    #     wigner_x = np.linspace(-maxl, maxl, 100, dtype=float)
    #     wigner_y = [wigner_dist(x) for x in wigner_x]
    #     ax.plot(wigner_x, wigner_y, color='r')
    #
    # nbr_of_sig_ew = len(np.where(EWs > ref_x[-1])[0])

    return None


def redundancy(EWs):
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
    print "\nRedundancy = {:.2f} \n".format(phi)

    return phi


def eigenvalue_spectra(EWs, method='SCREE', alpha=.05, ax=None, color='r'):
    """

    :param EWs:
    :param method: 'SCREE', 'proportion', 'broken-stick', 'average-root'
    :param alpha:
    :param ax:
    :param color:
    :return:
    """
    EWs = np.sort(EWs)[::-1]
    total_v = np.sum(abs(EWs))

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
        mask = np.zeros(len(EWs), np.bool)
        mask[:pc_count] = 1
        ax.plot(np.arange(len(EWs)), abs(EWs)/total_v, color=color)
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=mask, color=color, alpha=.4)
        ax.fill_between(np.arange(len(EWs)), abs(EWs) / total_v, 0,
                        where=np.logical_not(mask), color=color, alpha=.15)
        ax.set_xlabel('EW#')
        ax.set_ylabel('rel. EW')
        ax.set_xlim(0, len(EWs))
        ax.set_ylim(0, np.ceil((max(EWs)/total_v)*10)/10.)

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
        np.testing.assert_almost_equal(np.linalg.norm(EV), 1., decimal=7)
        for n_coord in EV:
            print "\033[{}m {:+.2f}\033[0m"\
                .format(colormap[colorcode(n_coord)], n_coord),
        print "\033[0m"
    return None


def EV_angles(EVs1, EVs2, deg=True):
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
    assert len(EVs1) == len(EVs2)
    # Transform into descending array of the eigenvector arrays
    EVs1 = np.absolute(EVs1.T[::-1])
    EVs2 = np.absolute(EVs2.T[::-1])

    for EVs in [EVs1, EVs2]:
        for EV in EVs:
            np.testing.assert_almost_equal(np.linalg.norm(EV), 1., decimal=7)

    M = np.dot(EVs1, EVs2.T)
    vector_angles = np.arccos(np.diag(M))
    space_angle = np.arccos(np.sqrt(np.linalg.det(np.dot(M, M.T))))

    if deg:
        vector_angles *= 180. / np.pi
        space_angle *= 180. / np.pi
        unit = "°"
    else:
        unit = " rad"

    print "\n\033[4mAngles between the eigenvectors\033[0m" \
          + "\n\t" + "\n\t".join('{:.2f}{}'.format(a, unit)
                                 for a in vector_angles)\
          + "\n\n" \
          + "\033[4mAngle between eigenspaces\033[0m" \
          + "\n\t{:.2f}{}".format(space_angle, unit)
    # ToDo: Understand the angle between spaces

    return vector_angles, space_angle


def detect_assemblies(EVs, EWs, detect_by='eigenvalue', show=True, EW_lim=2,
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

        if show and EWs[i] > EW_lim:
            print "\033[4mAssembly {}, eigenvalue {:.2f}, size {}\033[0m"\
                  .format(i+1, EWs[i], len(n_ids[i]))
            print "Neuron ID:\t",
            for n in n_ids[i]:
                print "{:2.0f}{}\t".format(n, "" if jupyter else "\t"),
            print "\tNorm"
            print "Portion:\t",
            for n in n_ids[i]:
                print "{:.2f}\t".format(EVs[i][n]),
            print "\t{:.2f}\n".format(np.linalg.norm(EVs[i][n_ids[i]]))
        i += 1

    for ev in EVs:
        th = 1/np.sqrt(len(ev))
        n_ids += [_get_ids(ev, th)]

    if not len(relevant_EVs):
        relevant_EVs = [[0]]

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
