import elephant
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
from quantities import Hz, ms
import imp
import neo
import numpy as np
import matplotlib.pyplot as plt
viziphant_path = '../INM6/Tasks/viziphant/plots/generic.py'
vizi = imp.load_source('*', viziphant_path)

r2gio_path = '../INM6/reach_to_grasp/python/reachgraspio/reachgraspio.py'
r2gio = imp.load_source('*', r2gio_path)


N = 100  # Number of neurons
T = 10000 *ms  # Sampling time
T_in_s = T.magnitude/1000.

# Generate Data

spiketrains = []

for n in range(N):
    spiketrains += [HPP(rate=10*Hz, t_stop=T)]


# Show Data

# vizi.rasterplot(spiketrains)
# plt.show()


# Calculate mean firing rates

MFR = []

for st in spiketrains:
    MFR += [len(st)/T_in_s]

# Calculate Population mean firing rate
print 'Mean firing rate of first neuron = %f Hz'%(MFR[0])
print 'Population mean firing rate = {:.2f} +- {:.2f} Hz'.format(PMFR, s_PMFR)


# Population Histogram

binsize = 2*ms
bins = np.zeros(T.magnitude/binsize.magnitude)

for st in spiketrains:
    for spike in st:
        bins[int(spike.magnitude/binsize.magnitude)] += 1



PMFR = np.mean(MFR)

# V_PMFR = 1./(N-1) * sum((MFR - PMFR)**2)
V_PMFR = np.var(MFR)

s_PMFR = np.sqrt(V_PMFR)
plt.plot(bins)
# plt.show()

print a
print a.read_block().list_children_by_class(neo.SpikeTrain),
