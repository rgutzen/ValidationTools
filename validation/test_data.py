""""""


import numpy as np
import quantities as qt
from elephant.spike_train_generation import single_interaction_process as sip
from elephant.spike_train_generation import compound_poisson_process as cpp
from elephant.spike_train_generation import homogeneous_poisson_process as hpp
import neo
import math as m


def generate_assembly(size, corr, method, rate, t_stop, **kwargs):
    if method == 'SIP':
        # "$corr of the spikes are correlated"
        return sip(rate=rate, rate_c=corr * rate, n=size, t_stop=t_stop,
                   coincidences='stochastic', **kwargs)

    elif method == 'MIP':
        # so sth
        return None

    elif method == 'CPP':
        # "$corr of the neurons are pairwise correlated"
        amp_dist = np.zeros(size+1)
        amp_dist[size+1] = corr
        poi = lambda k: 1./(m.e*m.factorial(k))
        amp_dist[:size] = [poi(k) for k in range(size)]
        norm_factor = (1. - corr) / np.sum(amp_dist[:size])
        amp_dist[:size] *= norm_factor
        return cpp(rate=rate, A=amp_dist, t_stop=t_stop, **kwargs)

    else:
        raise NameError("Method name not known!")

    return None


def test_data(size, corr, t_stop, method="SIP", assembly_size=[],
              rate=10.*qt.Hz, sorted=False, **kwargs):
    spiketrains = np.array(size, dtype=object)
    poisson_sts = np.array([hpp(rate=rate, t_stop=t_stop) for x in range(size)])
    for i, assembly in enumerate(assembly_size):
        generated_sts = np.sum(assembly_size[:i-1])
        spiketrains[generated_sts:generated_sts+assembly]\
            = generate_assembly(assembly, corr, method, rate, t_stop, **kwargs)
    return None