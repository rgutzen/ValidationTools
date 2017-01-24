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


#ToDo: Implement background correlation
#ToDo: Complete matrix analysis methods
#ToDo: What relative size of assemblies can still be detected?
#ToDo: What differences in background and assembly correlation can be detected?

# Generate Spiketrains
N = 100
spiketrain_list = testdata.test_data(size=N, corr=.5, t_stop=100*ms,
                                     rate=100*Hz, assembly_sizes=[],
                                     method="CPP")
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

fig = plt.figure('Spiketrains', figsize=(8,8))

# Rasterplot
ax1 = fig.add_subplot(221)
vizi.rasterplot(ax1, spiketrain_list)

# Heatmap
ax2 = fig.add_subplot(222)
matstat.plot_matrix(corr_matrix, ax2)

# EW Spectra
ax3 = fig.add_subplot(223)
EWs, EVs = np.linalg.eig(corr_matrix)
sEWs, sEVs = np.linalg.eig(surrogate_corr_matrix)
matstat.eigenvalue_distribution(EWs, ax3, surrogate_EWs=sEWs)
matstat.pc_trafo(corr_matrix)

# EW significance
ax4 = fig.add_subplot(224)
matstat.nbr_of_pcs(EWs, method='proportion', alpha=.05, ax=ax4)
matstat.redundancy(EWs)

plt.show()


