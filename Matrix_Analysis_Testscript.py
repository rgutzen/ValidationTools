import numpy as np
import matplotlib.pyplot as plt
import imp
from elephant.spike_train_surrogates import *
from elephant.statistics import mean_firing_rate, cv, isi
from quantities import Hz, ms
from scipy.linalg import eigh
# from neo.io.nestio import NestIO
import neo


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


COLLAB_PATH = '/home/robin/Projects/ValidationTools'
COLLAB_PATH_NEST = COLLAB_PATH + "/sim_data/NEST_data"
COLLAB_PATH_SPINNAKER = COLLAB_PATH + "/sim_data/SpiNNaker_data"

statistics_path = './validation/dist.py'
dist = imp.load_source('*', statistics_path)

matrix_analysis_path = './validation/matrix.py'
matstat = imp.load_source('*', matrix_analysis_path)

test_data_path = './validation/test_data.py'
testdata = imp.load_source('*', test_data_path)

viziphant_path = '../INM6/Tasks/viziphant/plots/generic.py'
vizi = imp.load_source('*', viziphant_path)

gdfio_path = '../INM6/Tasks/UP-Tasks/Elephant/gdf2NeoH5_task/gdfio.py'
gdfio = imp.load_source('*', gdfio_path)

# ToDo: method takes also list of method strings
# ToDo: What relative size of assemblies can still be detected?
# ToDo: Background vs Assembly correlation
# ToDo: Write Annotations

def analyze_distributions(sample1, sample2):
    # Create Figure
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.tight_layout()

    # Kullback-Leidner Divergence
    dist.KL_test(sample1, sample2, ax=ax[0],
                 bins=10, excl_zeros=True)

    # Kolmogorov-Smirnov Disance
    dist.KS_test(sample1, sample2, ax=ax[1])

    # Mann-Whitnes-U Test
    dist.MWU_test(sample1, sample2, ax=ax[2], excl_nan=True)

    return None


def analyze_correlations(spiketrain_list, filename='testfile'):
    # Generate Surrogates
    surrogate_spiketrain_list = testdata.generate_surrogates(spiketrain_list,
                                                         dither_spike_train,
                                                         shift=10*ms)

    # Generate Correlation Matrix
    corr_matrix = matstat.corr_matrix(spiketrain_list)

    # Generate Surrogate Correlation Matrix
    surrogate_corr_matrix = matstat.corr_matrix(surrogate_spiketrain_list)

    # Crate Figure
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout()

    # Rasterplot
    vizi.rasterplot(spiketrain_list, ax=ax[0,0])

    # Heatmap
    matstat.plot_matrix(corr_matrix, ax[0,1])

    # EW Spectra
    EWs, EVs = eigh(corr_matrix)
    sEWs, sEVs = eigh(surrogate_corr_matrix)
    EWmax_mean, EWmax_std = matstat.estimate_largest_eigenvalue(100,
                                                                10,
                                                                500,
                                                                100,
                                                                2)

    pc_count = matstat.eigenvalue_distribution(EWs, ax[1,0],
                                               # reference_EWs=sEWs,
                                               reference_EW_max=EWmax_mean + 2*EWmax_std,
                                               bins=int(max(EWs))*5)

    # EW Redundancy
    matstat.redundancy(EWs)

    # EW significance
    pc_count = matstat.eigenvalue_spectra(EWs, method='SCREE', ax=ax[1,1])

    # Print eigenvectors
    matstat.print_eigenvectors(EVs, EWs, pc_count)

    # Detect assemblies
    matstat.detect_assemblies(EVs, EWs, detect_by='eigenvalue')


    return EWs, EVs, pc_count


N = 50

# Generate Spiketrains
spiketrain_list1 = testdata.test_data(size=N, corr=.0, t_stop=500*ms,
                                      rate=100*Hz, assembly_sizes=[10,5],
                                      method="CPP", bkgr_corr=0.0)
for i, st in enumerate(spiketrain_list1):
    st.annotations['id'] = i

# Load NEST L4 exh Spiktrains
# spiketrain_list1 = load_data(COLLAB_PATH_NEST, ['spikes_L4'], N)[0][0]

# Load Brunel Network spiketrains (gdf)
# filename = "./sim_data/brunel_exp/J0.1_D1.5_g3.0_v2.0_T1000.0%0.1_ex-12502-0.gdf"
# r = NestIO(filenames=filename)
# segment = r.read_segment(gid_list=[], id_column_gdf=0, time_column_gdf=1,
#                          t_start=0.*ms, t_stop=1000.*ms)
# spiketrain_list1 = segment.spiketrains

# Calculate CVs
dist_sample_1 = [cv(isi(st)) for st in spiketrain_list1]


# Analyze Correlations of spiketrains 1
# EWs1, EVs1, pc_count1 = analyze_correlations(spiketrain_list1)


# Generate second dataset
spiketrain_list2 = testdata.test_data(size=N, corr=.0, t_stop=500*ms,
                                      rate=100*Hz, assembly_sizes=[2],
                                      method="CPP", bkgr_corr=0.0)

# Load SpiNNaker L4 exh Spiketrains
# spiketrain_list2 = load_data(COLLAB_PATH_SPINNAKER, ['spikes_L4'], N)[0][0]

# Load Brunel Network spiketrains (gdf)
# filename = "./sim_data/brunel_exp/J0.1_D1.5_g3.0_v4.0_T1000.0%0.1_ex-12502-0.gdf"
# r = NestIO(filenames=filename)
# segment = r.read_segment(gid_list=[], id_column_gdf=0, time_column_gdf=1,
#                          t_start=0.*ms, t_stop=1000.*ms)
# spiketrain_list2 = segment.spiketrains

# Calculate CVs
dist_sample_2 = [cv(isi(st)) for st in spiketrain_list2]

# Analyze Correlations of spiketrains 2
# EWs2, EVs2, pc_count2 = analyze_correlations(spiketrain_list2)

# Compare Spiketrain Correlations
## Angles between eigenspaces
# assemblysize = pc_count1
# matstat.EV_angles(EVs1[:, :], EVs2[:, :])

# Compare CV(ISI) Distributions
analyze_distributions(dist_sample_1, dist_sample_2)


plt.show()
