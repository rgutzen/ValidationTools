
import numpy as np
import matplotlib.pyplot as plt
from numpy import log
import scipy.integrate as integrate
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
from quantities import Hz, ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef

spiketrains = [HPP(10*Hz, t_stop=10*ms) for _ in range(10)]

binned_sts = BinnedSpikeTrain(spiketrains, 1*ms,
                              t_start=0*ms, t_stop=10*ms)

print np.nan_to_num(corrcoef(binned_sts))