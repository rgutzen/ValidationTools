""""""


import numpy as np
from elephant.spike_train_generation import single_interaction_process as SIP
from elephant.spike_train_generation import compound_poisson_process as CPP
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
import elephant.spike_train_surrogates as sg
from scipy.stats import poisson
import quantities
import neo
import random as r


def load_data(path, file_name_list, N):
    """
    Loads spiketrains from a hdf5 file in the neo data format.

    :param path: string
        Path to file
    :param file_name_list: list
        List of hdf5 filenames
    :param N:
        Number of returned spiketrains. When less are found in the file empty
        spiketrains are added; when more are found only the first N are
        returned
    :return:
        N   List of N tuples of neo.Spiketrains
    """
    # Load NEST or SpiNNaker data using NeoHdf5IO
    spike_train_list = []
    for file_name in file_name_list:
        # exc. and inh. as tuples, layerwise
        nest_data_path_exc = '/' + file_name + 'E.h5'
        nest_data_path_inh = '/' + file_name + 'I.h5'
        data = (neo.io.NeoHdf5IO(path + nest_data_path_exc),
                neo.io.NeoHdf5IO(path + nest_data_path_inh))
        spiketrains = (data[0].read_block().list_children_by_class(neo.SpikeTrain),
                       data[1].read_block().list_children_by_class(neo.SpikeTrain))
        tstart = spiketrains[0][0].t_start
        tstop = spiketrains[0][0].t_stop
        unit = spiketrains[0][0].units
        # if the recorded spiketrains are less than number of samples
        for spiketrain in spiketrains:
            while len(spiketrain) < N:
                spiketrain.append(neo.SpikeTrain([] * unit, t_start=tstart, t_stop=tstop))
            del spiketrain[N:]

        for st_count in range(len(spiketrains[0])):
            spiketrains[0][st_count].annotations['type'] = 'exc'
            spiketrains[0][st_count].annotations['layer'] = \
                file_name.split('_')[-1]
        for st_count in range(len(spiketrains[1])):
            spiketrains[1][st_count].annotations['type'] = 'inh'
            spiketrains[1][st_count].annotations['layer'] = \
                file_name.split('_')[-1]

        spike_train_list.append(spiketrains)

    return spike_train_list


def test_data(size, corr, t_stop, rate, method="CPP", assembly_sizes=[],
              bkgr_corr=0., shuffle=True, shuffle_seed=None):
    """
    Function to generate list of spiketrains with subsets of correlated
    assemblies.

    :param size: int
        Number of spiketrains
    :param corr: float
        Parameter given to the method which generates the correlated
        spiketrains.
        In case of 'CPP' corr is the probability that a spike occurs in all
        spiketrains of the assembly.
        In case of 'SIP' corr is the quotient of rates of spikes in all vs
        only in one spiketrain of the assembly.
    :param t_stop: quantity
        stop time
    :param rate: quantity
        Mean firing rate of the neurons
    :param method: 'CPP', 'SIP', ('MIP' to come)
        Method to generate the correlation between spiketrains.
        CPP Cumulative Poisson Process
        SIP Single Interaction Process
        MIP Multiple Interaction Process
    :param assembly_sizes: list of int
        Sizes of the non-overlapping assemblies
    :param bkgr_corr: float
        Probability for spike to randomly pairwise coincide throughout the
        netowork.
    :param shuffle: Boolean
        When True the spiketrains are shuffled after generation so that
        the assemblies consist of randomly chosen neurons.
    :param shuffle_seed: int, float
        In order to replicate a set of spiketrains with the same neuron ids in
        the respective assemblies a seed for the shuffle randomization can be
        set.
    :return: list of neo.Spiketrains
        spiketrains
    """

    def _generate_assembly(size, corr, bkgr_corr, method, rate, t_stop):
        if method == 'SIP':
            # "$corr of the spikes are correlated"
            return SIP(rate=rate, rate_c=corr * rate, n=size, t_stop=t_stop,
                       coincidences='stochastic')
        elif method == 'MIP':
            # do sth
            return None
        elif method == 'CPP':
            # "$corr of the neurons are pairwise correlated"
            # amp_dist = poisson(1).pmf(np.arange(size+1))
            amp_dist = np.zeros(size + 1)
            amp_dist[1] = 1 - corr - bkgr_corr
            amp_dist[2] = bkgr_corr
            amp_dist[size] = corr
            # amp_dist[:size] = [1./(m.e*m.factorial(k)) for k in range(size)]
            # np.testing.assert_almost_equal(sum(amp_dist), 1., decimal=7)
            # norm_factor = (1. - corr) / np.sum(amp_dist[:size])
            amp_dist *= (1./sum(amp_dist))
            return CPP(rate=rate, A=amp_dist, t_stop=t_stop)

        else:
            raise NameError("Method name not known!")

    spiketrains = [None] * size

    if not type(corr) == list:
        corr = [corr] * len(assembly_sizes)
    elif len(corr) == 1:
        corr *= len(assembly_sizes)

    for i, a_size in enumerate(assembly_sizes):
        assert a_size >= 2
        generated_sts = int(np.sum(assembly_sizes[:i]))
        spiketrains[generated_sts:generated_sts+a_size]\
            = _generate_assembly(a_size, corr[i], bkgr_corr,
                                 method, rate, t_stop)

    if bkgr_corr > 0:
        bkgr_size = size-sum(assembly_sizes)+1
        # amp_dist = poisson(bkgr_corr*bkgr_size).pmf(np.arange(bkgr_size))
        # amp_dist = amp_dist / sum(amp_dist)
        amp_dist = np.zeros(bkgr_size)
        amp_dist[1] = 1 - bkgr_corr
        amp_dist[2] = bkgr_corr
        spiketrains[int(np.sum(assembly_sizes)):]\
            = CPP(rate=rate, A=amp_dist, t_stop=t_stop)
    else:
        spiketrains[sum(assembly_sizes):] \
            = np.array([HPP(rate=rate, t_stop=t_stop)
                        for _ in range(size-sum(assembly_sizes))])

    if shuffle:
        if shuffle_seed is None:
            r.shuffle(spiketrains)
        else:
            r.Random(shuffle_seed).shuffle(spiketrains)

    return spiketrains


def generate_surrogates(spiketrains, surrogate_function=sg.dither_spike_train,
                        **args):
    sts_surrogates = []
    for st in spiketrains:
        sts_surrogates.append(surrogate_function(st, **args)[0])
    return sts_surrogates


def vine(d, cc_matrix):
    """
    VINE METHOD to generate random correlation matrices
    """
    P = np.zeros((d,d))          # storing partial correlations
    S = np.eye(d)
    cc_values = np.triu(cc_matrix,1).flatten()
    np.random.shuffle(cc_values)
    for k in range(d):
        for i in np.arange(k+1, d):
            P[k,i] = cc_values[k*d + i - 0.5 * (k+1)*(k+2)]
            p = P[k,i]
            # converting partial correlation to raw correlation
            for l in np.arange(k-1,0,-1):
                p = p * np.sqrt((1.-P[l,i]**2)*(1.-P[l,k]**2)) + P[l,i]*P[l,k]
            if not np.isfinite(p):
                p = 0
            S[k,i] = p
            S[i,k] = p
    # permuting the variables to make the distribution permutation-invariant
    # permutation = np.random.permutation(d)
    # S = S[permutation][permutation]
    return S


def number_of_sync_spikes(cc, spikenum, B):
    if not hasattr(spikenum, "__len__"):
        spikenum = [spikenum, spikenum]

    mean_spikenum = [0, 0]
    for i, sn in enumerate(spikenum):
        mean_spikenum[i] = sn / float(B)

    norm_factor = 1.
    for N, m in zip(spikenum, mean_spikenum):
        norm_factor *= (N * (1. - 2.*m) + B*m**2)

    async_term = spikenum[0] * mean_spikenum[1] \
               + spikenum[1] * mean_spikenum[0] \
               - B * mean_spikenum[0] * mean_spikenum[1]

    sync_term = 1. - 3. * mean_spikenum[0] * mean_spikenum[1]

    return (cc * np.sqrt(norm_factor) + async_term) / sync_term


def sync_spike_matrix(cc_mat, spikenums, binnum):
    N = len(spikenums)
    SSM = np.eye(N) * spikenums

    for k in range(N):
        for j in np.arange(k+1, N):
            nss = number_of_sync_spikes(cc_mat[k, j],
                                        [spikenums[k], spikenums[j]],
                                        binnum)
            nss = np.round(nss)
            SSM[k,j] = nss
            SSM[j,k] = nss

    return SSM


def corr_to_sync_prob(cc, A_size, rate, T, nbr_of_bins):
    if A_size < 2:
        raise ValueError
    if cc == 1:
        return 1
    m0 = rate * T / float(nbr_of_bins)
    if type(m0) == quantities.quantity.Quantity:
        m0 = m0.rescale('dimensionless')
    n = float(A_size)

    root = np.sqrt(    cc**2  * n**2
                   - 2*cc**2  * n
                   +   cc**2
                   + 4*cc *m0 * n
                   - 4*cc *m0
                   - 2*cc     * n**2
                   + 2*cc     * n
                   -     4*m0 * n
                   +     4*m0
                   +           n**2)

    adding = (- 2*cc*m0*n
              + 2*cc*m0
              + cc*n**2
              - cc*n
              + 2*m0*n
              - 2*m0
              - n**2)

    denominator = 2*(cc - 1.) * m0 * (n - 1.)**2

    sync_prob = (n * root + adding) / denominator

    if type(sync_prob) == quantities.quantity.Quantity:
        if bool(sync_prob.dimensionality):
            raise ValueError
        else:
            return sync_prob.magnitude
    else:
        return sync_prob


def sync_prob_to_corr(sync_prob, A_size, rate, T, nbr_of_bins):
    # mean number of spikes per bin
    m_0 = rate * T / float(nbr_of_bins)
    # mean number of spikes per bin within the  assembly
    m = m_0 * (1. + (A_size - 1.) * sync_prob) / float(A_size)
    # mean number of synchronous spikes per bin in the assembly
    m_s = m_0 * sync_prob

    corr = (m_s - m**2) / (m - m**2)

    if type(corr) == quantities.quantity.Quantity:
        if bool(corr.dimensionality):
            raise ValueError
        else:
            return corr.magnitude
    else:
        return corr


def transform_sync_prob(sync_prob_0, A_size_0, rate_0, T_0, B_0,
                        A_size_1=None, rate_1=None,
                        T_1=None, B_1=None):
    if A_size_1 is None:
        A_size_1 = A_size_0
    if rate_1 is None:
        rate_1 = rate_0
    if T_1 is None:
        T_1 = T_0
    if B_1 is None:
        B_1 = B_0
    corrcoef = sync_prob_to_corr(sync_prob_0, A_size_0, rate_0, T_0, B_0)
    return corr_to_sync_prob(corrcoef, A_size_1, rate_1, T_1, B_1)
