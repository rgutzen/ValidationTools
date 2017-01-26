import numpy as np
import matplotlib.pyplot as plt
import imp
from elephant.spike_train_surrogates import *
from quantities import Hz, ms

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

# Generate Spiketrains
N = 30
spiketrain_list = testdata.test_data(size=N, corr=.2, t_stop=100*ms,
                                     rate=100*Hz, assembly_sizes=[5],
                                     method="CPP", bkgr_corr=.05)
for i, st in enumerate(spiketrain_list):
    st.annotations['id'] = i

# Generate Surrogates
surrogate_spiketrain_list = testdata.generate_surrogates(spiketrain_list,
                                                         dither_spike_train,
                                                         shift=10*ms)

# Generate Correlation Matrix
corr_matrix = matstat.corr_matrix(spiketrain_list)

# Generate Surrogate Correlation Matrix
surrogate_corr_matrix = matstat.corr_matrix(surrogate_spiketrain_list)


fig, ax = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()

# Rasterplot
vizi.rasterplot(ax[0,0], spiketrain_list)

# Heatmap
matstat.plot_matrix(corr_matrix, ax[0,1])

# EW Spectra
EWs, EVs = np.linalg.eig(corr_matrix)
sEWs, sEVs = np.linalg.eig(surrogate_corr_matrix)
matstat.eigenvalue_distribution(EWs, ax[1,0], surrogate_EWs=sEWs, binnum=40)
matstat.pc_trafo(corr_matrix)

# EW redundancy
matstat.redundancy(EWs)

# EW significance
PCs = matstat.nbr_of_pcs(EWs, method='SCREE', alpha=.05, ax=ax[1,1])
pc_count = len(PCs)

# Generate second dataset
spiketrain_list = testdata.test_data(size=N, corr=.2, t_stop=100*ms,
                                     rate=100*Hz, assembly_sizes=[5],
                                     method="CPP", bkgr_corr=.05)
corr_matrix = matstat.corr_matrix(spiketrain_list)
EWs2, EVs2 = np.linalg.eig(corr_matrix)

# Sort both sets of EVs
EVs = np.array([ev for (ew,ev) in sorted(zip(EWs,EVs))[::-1]])
EVs2 = np.array([ev for (ew,ev) in sorted(zip(EWs2,EVs2))[::-1]])

# Angles between eigenspaces
print sorted(EWs2)[::-1]
matstat.EV_angles(EVs[:pc_count], EVs2[:pc_count])

# plt.show()
