import numpy as np
import matplotlib.pyplot as plt
import imp

COLLAB_PATH = '/home/robin/NeuroSim-Comparison-Tools'
COLLAB_PATH_NEST = COLLAB_PATH + "/sim_data/NEST_data"
COLLAB_PATH_SPINNAKER = COLLAB_PATH + "/sim_data/SpiNNaker_data"

plotting_path = './plotting_functions.py'
plotting = imp.load_source('*', plotting_path)

statistics_path = './validation/dist.py'
stat = imp.load_source('*', statistics_path)

matrix_analysis_path = './validation/matrix.py'
matstat = imp.load_source('*', matrix_analysis_path)

N = 10

rand_matrix = np.random.normal(size=(N,N))
corr_matrix = (rand_matrix + rand_matrix.T)/2.

fig = plt.figure('Heatmap', figsize=(4,8))
ax1 = fig.add_subplot(211)
matstat.plot_matrix(corr_matrix, ax1)
ax2 = fig.add_subplot(212)
EWs, EVs = np.linalg.eig(corr_matrix)
matstat.eigenvalue_spectra(EWs, ax2)
matstat.pc_trafo(corr_matrix, EWs, EVs)

plt.show()