# -*- coding: utf-8 -*-
"""
Toolbox for analyzing a correlation matrix.
"""

import numpy as np
from scipy.linalg import eigh
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy import stats as st
import contextlib
from quantities import Hz, ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef
import neo
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


def corr_matrix(spiketrains, binsize=2*ms, corr_type='pearson'):
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


def plot_matrix(matrix, ax=plt.gca(), remove_autocorr=False, labels=None, sorted=False):
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

    # ToDo: offer 'sorted' option for estimated assemblies presentation
    return None


def eigenvalue_distribution(EWs, ax=plt.gca(), binnum=20, reference_EWs=[],
                            color='g'):
    lmin = min(EWs)
    lmax = max(EWs)
    print "\n\033[4mEigenvalue distribution:\033[0m" \
          "\n\tEW_max = {:.2f}" \
          "\n\tEW_min = {:.2f}"\
          .format(lmax, lmin)

    edges = np.array([lmin + i*(lmax-lmin)/binnum for i in range(binnum+1)])
    EW_hist, edges = np.histogram(EWs, bins=edges, density=False)
    ax.bar(left=edges[:-1], height=EW_hist, width=edges[1]-edges[0],
           color=color)
    ax.set_xlabel('EW')
    ax.set_ylabel('Occurrence')
    ax.set_xlim(0, max(edges))

    if len(reference_EWs):
        ref_EW_hist, __ = np.histogram(reference_EWs, bins=edges, density=False)
        dx = edges[1]-edges[0]
        ref_x = np.append(edges[0] - dx, edges)
        # ref_x += dx / 2.
        ref_y = np.zeros_like(ref_x)
        ref_y[1:-1] = ref_EW_hist
        ax.plot(ref_x, ref_y, color='k')

        # y = 1
        # a = (1 - np.sqrt(y)) ** 2
        # b = (1 + np.sqrt(y)) ** 2
        # marchenko_pastur = lambda x: np.sqrt((x - a) * (b - x)) / (2 * np.pi * x * y)
        # marchenko_pastur = lambda x: np.sqrt(4*x - x**2) / (2*np.pi*x)
        # xaxis = np.linspace(0, int(math.ceil(edges[-1])), 50)
        # ax.plot(xaxis, [marchenko_pastur(x) for x in xaxis], color='r')
    else:
        # Reference to Random Correlation Matrix
        N = len(EWs)
        rand_matrix = np.random.rand(N, N) * 2. - 1
        corr_matrix = (rand_matrix + rand_matrix.T) / 2.
        surrogate_EWs, __ = np.linalg.eig(corr_matrix)
        # for i in range(N):
        #     corr_matrix[i,i] = 1.
        maxl = max([abs(min(surrogate_EWs)), abs(max(surrogate_EWs))])
        wigner_dist = lambda x: 2. / (np.pi*maxl**2) * np.sqrt(maxl**2-x**2)
        wigner_x = np.linspace(-maxl, maxl, 100, dtype=float)
        wigner_y = [wigner_dist(x) for x in wigner_x]
        ax.plot(wigner_x, wigner_y, color='r')

    nbr_of_sig_ew = len(np.where(EWs > ref_x[-1])[0])
    # print "\t{} eigenvalue{} are larger than the reference distribution \n"\
    #       .format(nbr_of_sig_ew, "" if nbr_of_sig_ew-1 else "s")
    return nbr_of_sig_ew


def redundancy(EWs, show=True):
    ### Measure of correlation of the matrix entries
    ### For 0 correlation sum(EW^2)=N -> phi=0
    ### For perfect correlation EW_1=N -> sum(EW^2)=N^2 -> phi=1
    N = len(EWs)
    phi = np.sqrt((np.sum(EWs**2)-N) / (N*(N-1)))
    if show:
        print "\nRedundancy = {:.2f} \n".format(phi)
    return phi


def eigenvalue_spectra(EWs, method='SCREE', alpha=.05, ax=None, show_dist=True, color='r'):
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
        pc_count = np.where(EWs < 1)[0][0]

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


def print_eigenvectors(EVs, EWs=[], pc_count=0, colormap=[90,90,94,92,93,91,96,95,97]):
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
    ### EVs must be sorted in ascending order as returned by scipy.linalg.eigh()
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

    # plt.figure()
    # Angle histogram
    # hist, edges = np.histogram(vector_angles, bins=20, density=True)
    # plt.bar(edges[:-1], hist, np.diff(edges), color='g')
    #
    # # Sample random vectors
    # N = 50
    # res = 1000
    # rand_angles = []
    # for j in range(res):
    #     vector = np.random.normal(size=(2, N))
    #     for i in range(2):
    #         vector[i] /= np.linalg.norm(vector[i])
    #     vector = np.absolute(vector)
    #     rand_angles += [np.arccos(np.dot(vector[0], vector[1]))]
    # hist, edges = np.histogram(rand_angles, bins=20, density=True)
    # plt.plot(edges[:-1]+np.diff(edges), hist, color='k')

    # N = len(EVs1[0])
    # res = 500
    # step = np.pi/(res)
    # phi = [step * (i+1) for i in range(res)]
    # norm = integrate.quad(lambda a: np.sin(a) ** (N - 2), 0, np.pi)[0]
    # f = [(np.sin(p)) ** (N - 2) for p in phi]
    # f = [f_it / (sum(f)*step) for f_it in f]
    # plt.plot(phi, f, color='r')

    if deg:
        vector_angles *= 180 / np.pi
        space_angle *= 180 / np.pi
        unit = "°"
    else:
        unit = " rad"

    print "\n\033[4mAngles between the eigenvectors\033[0m" \
          + "\n\t" + "\n\t".join('{:.2f}{}'.format(a, unit) for a in vector_angles)\
          + "\n\n" \
          + "\033[4mAngle between eigenspaces\033[0m" \
          + "\n\t{:.2f}{}".format(space_angle, unit)
    # ToDo: Understand the angle between spaces

    return vector_angles, space_angle


def detect_assemblies(EVs, EWs, detect_by='eigenvalue', show=True, EW_lim=2, jupyter=False, sort=False):
    EVs = np.absolute(EVs.T[::-1])
    EWs = EWs[::-1]
    if type(detect_by) == float or type(detect_by) == int:
        th = detect_by
    else:
        th = 0
    i = 0
    n_ids = []
    relevant_EVs = []
    while EWs[i] > EW_lim:
        if th:
            if th < 1:
                ids = np.where(EVs[i] > th)[0]
                size = len(ids)
            elif th >= 1:
                size = int(th)
                ids = np.argpartition(EVs[i], -size)[-size:]
        else:
            size = int(np.ceil(EWs[i]))
            ids = np.argpartition(EVs[i], -size)[-size:]

        n_ids += [ids[np.argsort(EVs[i][ids])][::-1]]

        relevant_EVs += [EVs[i][n_ids[i]]]

        if show:
            print "\033[4mAssembly {}, eigenvalue {:.2f}, size {}\033[0m"\
                  .format(i+1, EWs[i], size)
            print "Neuron ID:\t",
            for n in n_ids[i]:
                print "{:2.0f}{}\t".format(n, "" if jupyter else "\t"),
            print "\tNorm"
            print "Portion:\t",
            for n in n_ids[i]:
                print "{:.2f}\t".format(EVs[i][n]),
            print "\t{:.2f}\n".format(np.linalg.norm(EVs[i][n_ids[i]]))
        i += 1

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


# ToDo: Write annotations