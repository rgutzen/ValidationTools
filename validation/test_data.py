""""""


import numpy as np
from quantities import Hz, ms
from elephant.spike_train_generation import single_interaction_process as SIP
from elephant.spike_train_generation import compound_poisson_process as CPP
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
import elephant.spike_train_surrogates as sg
import neo
import math as m


def generate_assembly(size, corr, method, rate, t_stop):
    if method == 'SIP':
        # "$corr of the spikes are correlated"
        return SIP(rate=rate, rate_c=corr*rate, n=size, t_stop=t_stop,
                   coincidences='stochastic')

    elif method == 'MIP':
        # so sth
        return None

    elif method == 'CPP':
        # "$corr of the neurons are pairwise correlated"
        amp_dist = np.zeros(size+1)
        amp_dist[size] = corr
        amp_dist[:size] = [1./(m.e*m.factorial(k)) for k in range(size)]
        norm_factor = (1. - corr) / np.sum(amp_dist[:size])
        amp_dist[:size] *= norm_factor
        return CPP(rate=rate, A=amp_dist, t_stop=t_stop)

    else:
        raise NameError("Method name not known!")

    return None


def test_data(size, corr, t_stop, rate, method="SIP", assembly_sizes=[1],
              sorted=False):

    spiketrains = [None] * size

    for i, a_size in enumerate(assembly_sizes):
        generated_sts = int(np.sum(assembly_sizes[:i]))
        spiketrains[generated_sts:generated_sts+a_size]\
            = generate_assembly(a_size, corr, method, rate, t_stop)

    spiketrains[np.sum(assembly_sizes):] \
        = np.array([HPP(rate=rate, t_stop=t_stop)
                    for x in range(size-sum(assembly_sizes))])
    # Background Correlation
    return spiketrains


def generate_surrogates(spiketrains, surrogate_function=sg.dither_spike_train,
                        **args):
    sts_surrogates = []
    for st in spiketrains:
        sts_surrogates.append(surrogate_function(st, **args)[0])
    return sts_surrogates
