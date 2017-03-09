from pypet import Environment, cartesian_product
import imp
from quantities import ms, Hz
from scipy.linalg import eigh, norm
import numpy as np

base_path = '/home/robin/Projects/ValidationTools'

statistics_path = base_path + '/validation/dist.py'
dist = imp.load_source('*', statistics_path)

matrix_analysis_path = base_path + '/validation/matrix.py'
matstat = imp.load_source('*', matrix_analysis_path)

test_data_path = base_path + '/validation/test_data.py'
testdata = imp.load_source('*', test_data_path)


def assembly_detection(traj):
    """"""

    print '\n\t\033[93mRepetition: {}\033[0m\n'.format(traj.repetition)

    spiketrain_list = testdata.test_data(size=traj.N,
                                         corr=traj.corr,
                                         t_stop=traj.T * ms,
                                         rate=traj.rate * Hz,
                                         assembly_sizes=[traj.A_size],
                                         method="CPP",
                                         bkgr_corr=traj.bkgr_corr,
                                         shuffle=False)

    ref_spiketrain_list = testdata.test_data(size=traj.N,
                                             corr=0,
                                             t_stop=traj.T * ms,
                                             rate=traj.rate * Hz,
                                             method="CPP",
                                             bkgr_corr=traj.bkgr_corr,
                                             shuffle=False)

    # Distribution Comparison

    def func(sts):
        return matstat.corr_matrix(sts).flatten()
        # cv(isi(x)), mean_firing_rate(x)

    dist_sample_1 = func(spiketrain_list)
    dist_sample_2 = func(ref_spiketrain_list)

    DKL = dist.KL_test(dist_sample_1, dist_sample_2, excl_zeros=True)

    DKS = dist.KS_test(dist_sample_1, dist_sample_2)

    MWU = dist.MWU_test(dist_sample_1, dist_sample_2, excl_nan=True)

    # Correlation Analysis

    corr_matrix = matstat.corr_matrix(spiketrain_list)

    corrcoef = np.mean(corr_matrix[:traj.A_size, :traj.A_size].flatten())

    EWs, EVs = eigh(corr_matrix)

    redundancy = matstat.redundancy(EWs)

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

    # Angle Comparison

    ref_corr_matrix = matstat.corr_matrix(ref_spiketrain_list)
    __, ref_EVs = eigh(ref_corr_matrix)

    vector_angles, space_angle = matstat.EV_angles(EVs, ref_EVs)

    # Define Results

    traj.f_add_result('Corrcoef', corrcoef,
                      comment='Mean pairwise correlation coefficients within the assembly')
    traj.f_add_result('EW', EWs[-1],
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
    traj.f_add_result('D_KL', DKL,
                      comment='Kullback-Leidler-Divegence as tuple (D_KL(P||Q),D_KL(Q||P))')
    traj.f_add_result('D_KS', DKS,
                      comment='Kolmogorov-Smirnov-Distance as tuple (D_KS,p-value)')
    traj.f_add_result('MWU', MWU,
                      comment='Mann-Whitney-U statistic as tupe (MWU,p-value)')
    traj.f_add_result('Vector_angles', vector_angles,
                      comment='Angles between the i-th eigenvectors respectively')
    traj.f_add_result('Space_angle', space_angle,
                      comment='Angle between the eigenspaces')


env = Environment(trajectory='2corr_2T_5rep',
                  filename=base_path + '/ParameterExploration/assembly/corr_T.hdf5',
                  file_title='corr_vs_T_01',
                  large_overview_tables=True,
                  git_repository=base_path,
                  overwrite_file=True,
                  log_folder=base_path + '/ParameterExploration/logs/',
                  results_per_run=9)

traj = env.trajectory

traj.f_add_parameter('N', 100, comment='Number of neurons')
traj.f_add_parameter('corr', .0, comment='Correlation within assembly')
traj.f_add_parameter('T', 100, comment='Runtime')
traj.f_add_parameter('rate', 100, comment='Mean spiking rate')
traj.f_add_parameter('A_size', 10, comment='size of assembly')
traj.f_add_parameter('bkgr_corr', .0, comment='Background correlation')
traj.f_add_parameter('repetition', 0, comment='Iterator to produce statistics')

traj.f_explore(cartesian_product({'corr': [.0, .02, .04, .06, .08, .1],
                                  'T'   : [200, 400, 600, 800, 1000, 2000],
                                  'repetition': [0, 1, 2, 3, 4]}))

env.run(assembly_detection)

env.disable_logging()


