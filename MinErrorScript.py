from quantities import ms
from neo import SpikeTrain
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cch

times = [[50.,   229.],
         [24.,    65.,   101.,   138.,   155.,   283.]]

sts = [SpikeTrain(t, units='ms', t_stop=300*ms) for t in times]

bsts = [BinnedSpikeTrain(st, binsize=50*ms) for st in sts]

print cch(bsts[0], bsts[1], cross_corr_coef=True)[0].T[0]
