import quantities as pq
import numpy as np
import sys
import neo
import elephant
import imp
from elephant.statistics import time_histogram, mean_firing_rate, cv, isi
from elephant.spike_train_correlation import corrcoef
from elephant.conversion import BinnedSpikeTrain

import matplotlib.pyplot as plt

# Define path names

COLLAB_PATH = '/home/robin/Projects/ValidationTools'
COLLAB_PATH_NEST = COLLAB_PATH + "/sim_data/NEST_data"
COLLAB_PATH_SPINNAKER = COLLAB_PATH + "/sim_data/SpiNNaker_data"

plotting_path = './plotting_functions.py'
plotting = imp.load_source('*', plotting_path)

statistics_path = './validation/dist.py'
stat = imp.load_source('*', statistics_path)

def load_data(path, file_name_list, N):
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
        spike_train_list.append(spiketrains)
    return spike_train_list


file_names = ["spikes_L23", "spikes_L4", "spikes_L5", "spikes_L6"]

# Number of spiketrains to load
N = 100

sts_NEST = load_data(COLLAB_PATH_NEST, [f for f in file_names], N)
sts_SPINNAKER = load_data(COLLAB_PATH_SPINNAKER, [f for f in file_names], N)

# Rasterplot

binsize = 2 * pq.ms

# Population histogram and mean firing rate
pophist_NEST = []
mean_rate_NEST = []
for sts_NEST_layer in sts_NEST:
    pophist_NEST.append((time_histogram(sts_NEST_layer[0], binsize), time_histogram(sts_NEST_layer[1], binsize)))
    mean_rate_NEST.append(([mean_firing_rate(st).rescale('Hz') for st in sts_NEST_layer[0]],
                           [mean_firing_rate(st).rescale('Hz') for st in sts_NEST_layer[1]]))

pophist_SPINNAKER = []
mean_rate_SPINNAKER = []
for sts_SPINNAKER_layer in sts_SPINNAKER:
    pophist_SPINNAKER.append((time_histogram(sts_SPINNAKER_layer[0], binsize),
                              time_histogram(sts_SPINNAKER_layer[1], binsize)))
    mean_rate_SPINNAKER.append(([mean_firing_rate(st).rescale('Hz') for st in sts_SPINNAKER_layer[0]],
                                [mean_firing_rate(st).rescale('Hz') for st in sts_SPINNAKER_layer[1]]))

layers = ['L6', 'L5', 'L4', 'L2/3']
# fig = plt.figure('Rasterplot', figsize=(14, 8))
# ax = fig.add_subplot(1, 2, 1)
# plotting.plot_raster_pophist_rate(ax,
#                                   sts_NEST, pophist_NEST, mean_rate_NEST, binsize, N, \
#                                   layers, 'NEST', ['b', 'r'], 14)
# ax = fig.add_subplot(1, 2, 2)
# plotting.plot_raster_pophist_rate(ax,
#                                   sts_SPINNAKER, pophist_SPINNAKER, mean_rate_SPINNAKER, binsize, N, \
#                                   layers, 'SpiNNaker', ['g', 'y'], 14)
# plt.draw()
# plt.close(fig)

# Calculate CV and ISI

cv_NEST = []
for sts_NEST_layer in sts_NEST:
    cv_NEST.append(([cv(isi(i)) for i in sts_NEST_layer[0]],
                    [cv(isi(i)) for i in sts_NEST_layer[1]]))

cv_Spinnaker = []
for sts_spinnaker_layer in sts_SPINNAKER:
    cv_Spinnaker.append(([cv(isi(i)) for i in sts_spinnaker_layer[0]],
                         [cv(isi(i)) for i in sts_spinnaker_layer[1]]))

for idx_layer, (n_layer, s_layer) in enumerate(zip(cv_NEST, cv_Spinnaker)):
    for idx, (n, s) in enumerate(zip(n_layer, s_layer)):
        for ii, (ne, si) in enumerate(zip(n, s)):
            if isinstance(ne, np.ndarray):
                cv_NEST[idx_layer][idx][ii] = np.nan
            if isinstance(si, np.ndarray):
                cv_Spinnaker[idx_layer][idx][ii] = np.nan

# fig = plt.figure('ISI CV', figsize=(14, 10))
# hist_data = plotting.plot_distribution_comparison(cv_NEST, cv_Spinnaker, np.linspace(0, 1.5, 30),
#                                                  layers[::-1], ['NEST', 'SpiNNaker'], ['b', 'r'],
#                                                  ['g', 'y'], 20, (0, 1.5), (0, 0.25),
#                                                  'coefficient of variation (CV)'
#                                                  )

# plt.draw()
# plt.close(fig)

def statistical_testing(sample1,sample2):
    # KS Test:
    stat.KS_test(sample1, sample2)

    # KL Test:
    stat.KL_test(sample1, sample2, bins=np.linspace(0, 1.5, 30), excl_zeros=True)

    # MWW Test:
    stat.MWW_test(sample1, sample2, excl_nan=True)
    return None



statistical_testing(cv_NEST[1][0],cv_Spinnaker[1][0])
plt.show()
