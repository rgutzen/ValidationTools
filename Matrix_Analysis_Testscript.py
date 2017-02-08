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

def analyze_spiketrains(spiketrain_list, filename='testfile'):
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
    EWs, EVs = np.linalg.eigh(corr_matrix)
    sEWs, sEVs = np.linalg.eigh(surrogate_corr_matrix)
    matstat.eigenvalue_distribution(EWs, ax[1,0], surrogate_EWs=sEWs,
                                    binnum=int(max(EWs))*5)
    matstat.pc_trafo(corr_matrix)

    # EW redundancy
    matstat.redundancy(EWs)

    # EW significance
    PCs = matstat.nbr_of_pcs(EWs, method='SCREE', alpha=.05, ax=ax[1,1])
    pc_count = len(PCs)

    # Print to file
    import scipy.io
    scipy.io.savemat("a.mat", {"corrmatrix": corr_matrix, "ew": EWs, "ev": EVs})
    N = len(EWs)
    f = open(filename,'w+')
    print >>f, "Correlation Matrix ({}x{}): \n".format(N, N)
    print >>f, corr_matrix
    print >>f, "\nEigenvalues:\n"
    print >>f, EWs
    print >>f, "\nEigenvectors:\n"
    print >>f, EVs

    return EWs, EVs, pc_count


# Generate Spiketrains
N = 30
spiketrain_list = testdata.test_data(size=N, corr=.7, t_stop=500*ms,
                                     rate=100*Hz, assembly_sizes=[10, 5],
                                     method="CPP", bkgr_corr=0.01)
for i, st in enumerate(spiketrain_list):
    st.annotations['id'] = i

EWs1, EVs1, pc_count1 = analyze_spiketrains(spiketrain_list)

# Generate second dataset
spiketrain_list = testdata.test_data(size=N, corr=.7, t_stop=100*ms,
                                     rate=100*Hz, assembly_sizes=[5],
                                     method="CPP", bkgr_corr=0)

# EWs2, EVs2, pc_count2 = analyze_spiketrains(spiketrain_list)

# Angles between eigenspaces
# assemblysize = int(max([max(EWs1), max(EWs2)]))
# matstat.EV_angles(EVs1[:assemblysize], EVs2[:assemblysize])

# The eigenvectors with the smallest eigenvalues point to the assembly
# neurons because the assemblies with high correlation show the least variance.
# Therefore the relevant angels are of EV[:-assemblysize]
# The assemblysizes can be estimate by the size of the eigenvalues

# Print eigenvectors
matstat.print_eigenvectors(EVs1[::-1], EWs1[::-1])
# matstat.print_eigenvectors(EVs2[::-1], EWs2[::-1])

plt.show()

# ToDo: Print results of organized test runs into file for further analysis
# Covariance matrix
# vergleich zu matlab
