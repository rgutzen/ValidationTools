from pypet import Environment, cartesian_product
import imp
from quantities import ms, Hz
from scipy.linalg import eigh, norm
import numpy as np

base_path = '/home/robin/Projects/ValidationTools'

matrix_analysis_path = base_path + '/validation/matrix.py'
matstat = imp.load_source('*', matrix_analysis_path)

test_data_path = base_path + '/validation/test_data.py'
testdata = imp.load_source('*', test_data_path)


def assembly_detection(traj):
    """"""

    spiketrain_list = testdata.test_data(size=traj.N,
                                         corr=traj.corr,
                                         t_stop=traj.T * ms,
                                         rate=traj.rate * Hz,
                                         assembly_sizes=[traj.A_size],
                                         method="CPP",
                                         bkgr_corr=traj.bkgr_corr)

    corr_matrix = matstat.corr_matrix(spiketrain_list)

    corrcoef = np.mean(corr_matrix[:traj.A_size, :traj.A_size].flatten())

    EWs, EVs = eigh(corr_matrix)

    redundancy = matstat.redundancy(EWs, show=False)

    SCREE_count = matstat.eigenvalue_spectra(EWs, method='SCREE')

    broken_stick_count = matstat.eigenvalue_spectra(EWs, method='broken-stick')

    relevant_EVs = matstat.detect_assemblies(EVs, EWs, show=False,
                                             detect_by='eigenvalue')[0]

    norm_estimate = norm(relevant_EVs)
    min_n_estimate = min(relevant_EVs)

    relevant_EVs = matstat.detect_assemblies(EVs, EWs, show=False,
                                             detect_by=traj.A_size)[0]

    norm_exact = norm(relevant_EVs)
    min_n_exact = min(relevant_EVs)

    traj.f_add_result('Corrcoef', corrcoef,
                      comment='Mean pairwise correlation coefficients within the assembly')
    traj.f_add_result('EW', EWs[0],
                      comment='largest eigenvalue, related to assembly')
    traj.f_add_result('Redundancy', redundancy)
    traj.f_add_result('SCREE_count', SCREE_count,
                      comment='Estimate number of significant eigenvalues by SCREE method')
    traj.f_add_result('broken_stick_count', broken_stick_count,
                      comment='Estimate number of significant eigenvalues by broken-stick method')
    traj.f_add_result('Norm_estimate', norm_estimate,
                      comment='Norm of the subset of the first eigenvector of length of the eigenvalue')
    traj.f_add_result('Norm_exact', norm_exact,
                      comment='Norm of the subset of the first eigenvector of length of the assembly size')
    traj.f_add_result('min_n_estimate', min_n_estimate,
                      comment='Smallest contribution to the estimated relevant subset of the eigenvector')
    traj.f_add_result('min_n_exact', min_n_exact,
                      comment='Smallest contribution to the exact relevant subset of the eigenvector')


env = Environment(trajectory='Correlation_vs_Datasize',
                  filename='./assembly/corr_vs_T.hdf5',
                  file_title='corr_vs_T_01',
                  large_overview_tables=True,
                  git_repository=base_path,
                  overwrite_file=True,
                  log_folder='./logs/')

traj = env.trajectory

traj.f_add_parameter('N', 100, comment='Number of neurons')
traj.f_add_parameter('corr', .0, comment='Correlation within assembly')
traj.f_add_parameter('T', 100, comment='Runtime')
traj.f_add_parameter('rate', 100, comment='Mean spiking rate')
traj.f_add_parameter('A_size', 10, comment='size of assembly')
traj.f_add_parameter('bkgr_corr', .0, comment='Background correlation')

traj.f_explore(cartesian_product({'corr': [.0, .1 ,.2],
                                  'T'   : [100, 200, 300]}))

env.run(assembly_detection)

env.disable_logging()



