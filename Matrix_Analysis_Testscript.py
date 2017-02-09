import numpy as np
import matplotlib.pyplot as plt
import imp
from elephant.spike_train_surrogates import *
from quantities import Hz, ms
from scipy.linalg import eigh


COLLAB_PATH = '/home/robin/NeuroSim-Comparison-Tools'
COLLAB_PATH_NEST = COLLAB_PATH + "/sim_data/NEST_data"
COLLAB_PATH_SPINNAKER = COLLAB_PATH + "/sim_data/SpiNNaker_data"

plotting_path = './plotting_functions.py'
plotting = imp.load_source('*', plotting_path)

statistics_path = './validation/dist.py'
stat = imp.load_source('*', statistics_path)

matrix_analysis_path = './validation/matrix.py'
matstat = imp.load_source('*', matrix_analysis_path)

test_data_path = './validation/test_data.py'
testdata = imp.load_source('*', test_data_path)

viziphant_path = '../INM6/Tasks/viziphant/plots/generic.py'
vizi = imp.load_source('*', viziphant_path)

# ToDo: method takes also list of method strings
# ToDo: Complete matrix analysis methods
# ToDo: What relative size of assemblies can still be detected?
# ToDo: Background vs Assembly correlation
# ToDo: Write Annotations

def analyze_spiketrains(spiketrain_list, filename='testfile'):
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
    vizi.rasterplot(ax[0,0], spiketrain_list)

    # Heatmap
    matstat.plot_matrix(corr_matrix, ax[0,1])

    # EW Spectra
    EWs, EVs = eigh(corr_matrix)
    sEWs, sEVs = eigh(surrogate_corr_matrix)
    matstat.eigenvalue_distribution(EWs, ax[1,0], surrogate_EWs=sEWs,
                                    binnum=int(max(EWs))*5)

    matstat.redundancy(EWs)

    # EW significance
    PCs = matstat.nbr_of_pcs(EWs, method='SCREE', alpha=.05, ax=ax[1,1])
    pc_count = len(PCs)

    return EWs, EVs, pc_count


# Generate Spiketrains
N = 30
spiketrain_list = testdata.test_data(size=N, corr=[.9,.9,.9], t_stop=500*ms,
                                     rate=100*Hz, assembly_sizes=[8,8,8],
                                     method="CPP", bkgr_corr=0.9)
for i, st in enumerate(spiketrain_list):
    st.annotations['id'] = i

EWs1, EVs1, pc_count1 = analyze_spiketrains(spiketrain_list)

# Generate second dataset
spiketrain_list = testdata.test_data(size=N, corr=[.9,.9,.9], t_stop=500*ms,
                                     rate=100*Hz, assembly_sizes=[8,8,8],
                                     method="CPP", bkgr_corr=0.9)

EWs2, EVs2, pc_count2 = analyze_spiketrains(spiketrain_list)

# Angles between eigenspaces
assemblysize = pc_count1
assemblysize = N
matstat.EV_angles(EVs1[:, -assemblysize:], EVs2[:, -assemblysize:])

# Print eigenvectors
matstat.print_eigenvectors(EVs1, EWs1)
matstat.print_eigenvectors(EVs2, EWs2)

# plt.show()
