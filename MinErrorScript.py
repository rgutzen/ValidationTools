
import numpy as np
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
from quantities import Hz, ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef

N = 100
t_stop =70*ms

spiketrain_list = [HPP(rate=100*Hz, t_stop=t_stop) for i in range(N)]

binned_spiketrains = BinnedSpikeTrain(spiketrain_list, binsize=2*ms, t_stop=t_stop)

corr_matrix = corrcoef(binned_spiketrains)

EWs, EVs = np.linalg.eigh(corr_matrix)

print sorted(EWs)
print np.array([np.linalg.norm(ev) for ev in EVs])
print np.all(np.array([np.linalg.norm(ev) for ev in EVs])) == 1.
