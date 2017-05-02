""""""


import numpy as np
from elephant.spike_train_generation import single_interaction_process as SIP
from elephant.spike_train_generation import compound_poisson_process as CPP
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
import elephant.spike_train_surrogates as sg
from scipy.stats import poisson
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
            print amp_dist
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
