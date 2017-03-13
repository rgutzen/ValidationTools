from pypet import Environment, cartesian_product
from imp import load_source
from quantities import ms, Hz
from scipy.linalg import eigh, norm
import numpy as np
import time
from guppy import hpy; h=hpy()
from os.path import expanduser
import sys
from StringIO import StringIO

# Set up paths and validation packages

start_time = time.time()

home_path = expanduser('~')

base_path = home_path + '/Projects/ValidationTools'

statistics_path = base_path + '/validation/dist.py'
dist = load_source('*', statistics_path)

matrix_analysis_path = base_path + '/validation/matrix.py'
matstat = load_source('*', matrix_analysis_path)

test_data_path = base_path + '/validation/test_data.py'
testdata = load_source('*', test_data_path)

# Define parameter print functions

def print_table(table):
    col_width = [max(len(str(x)) for x in col)
                 for col in table]

    for i in range(len(table[0])):
        for j, column in enumerate(table):
            print "{:{}}\t\t".format(column[i], col_width[j]),
        print ""


def show_params(traj):
    params = [key.split('.')[-1] for key in traj._parameters.keys()]
    # comments = [traj.f_get(p).v_comment for p in params]
    defaults = [traj.f_get(p).f_get() for p in params]
    # has_range = [traj.f_get(p).f_has_range() for p in params]
    # ranges = [np.unique(np.array(traj.f_get(p).f_get_range())) if has else []
    #           for p, has in zip(params, has_range)]
    table = [params, defaults]# , ranges, comments]

    print "\n\033[4m\033[94mParameters\033[0m\033[0m\n" \
        + "\033[93m"
    print_table(table)
    print "\033[0m"


def assembly_detection(traj):
    """"""
    # Print current parameters

    show_params(traj)

    # Catch print output

    actualstdout = sys.stdout
    sys.stdout = StringIO()


    # Generate spiketrain sets

    spiketrain_list = testdata.test_data(size=traj.N,
                                         corr=traj.corr,
                                         t_stop=traj.T * ms,
                                         rate=traj.rate * Hz,
                                         assembly_sizes=[traj.A_size],
                                         method="CPP",
                                         bkgr_corr=traj.bkgr_corr,
                                         shuffle=False)

    ref_spiketrain_list = testdata.test_data(size=traj.N,
                                             corr=0.,
                                             t_stop=traj.T * ms,
                                             rate=traj.rate * Hz,
                                             method="CPP",
                                             bkgr_corr=traj.bkgr_corr,
                                             shuffle=False)

    # ToDo: Run exploration also with reference spiketrain list with same corr

    # Distribution Comparison

    dist_sample_1 = matstat.corr_matrix(spiketrain_list).flatten()
    dist_sample_2 = matstat.corr_matrix(ref_spiketrain_list).flatten()

    DKL = dist.KL_test(dist_sample_1, dist_sample_2, excl_zeros=True)

    DKS = dist.KS_test(dist_sample_1, dist_sample_2)

    MWU = dist.MWU_test(dist_sample_1, dist_sample_2, excl_nan=True)

    # Correlation Analysis

    corr_matrix = matstat.corr_matrix(spiketrain_list)

    corrcoef = np.mean(corr_matrix[:traj.A_size, :traj.A_size].flatten())

    EWs, EVs = eigh(corr_matrix)
    EW_max = np.sort(EWs)[-1]

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
    ref_EWs, ref_EVs = eigh(ref_corr_matrix)

    ref_vector_angles, ref_space_angle = matstat.EV_angles(EVs, ref_EVs)

    spectral_count = len(np.where(EWs > np.max(ref_EWs))[0])

    # Mute print output from functions

    sys.stdout = actualstdout
    sys.stdout.flush()

    # Define Results

    # HDF5 has problems with nodes that have more than 20k children!
    # For large parameter exploration use wildcards ($set) for grouping in sets
    # of 1000 children.

    traj.f_add_result('$set.$.Corrcoef', corrcoef,
                      comment='Mean pairwise correlation coefficients within the assembly')
    traj.f_add_result('$set.$.EW_max', EW_max,
                      comment='largest eigenvalue, related to assembly')
    traj.f_add_result('$set.$.Redundancy', redundancy)
    traj.f_add_result('$set.$.SCREE_count', SCREE_count,
                      comment='Estimate number of significant eigenvalues by SCREE method')
    traj.f_add_result('$set.$.broken_stick_count', broken_stick_count,
                      comment='Estimate number of significant eigenvalues by broken-stick method')
    traj.f_add_result('$set.$.spectral_count', spectral_count,
                      comment='How many eigenvalues are larger than the larges eigenvalue of the reference network?')
    traj.f_add_result('$set.$.Norm_estimate', norm_estimate,
                      comment='Norm of the subset of the first eigenvector of length of the eigenvalue')
    traj.f_add_result('$set.$.Norm_exact', norm_exact,
                      comment='Norm of the subset of the first eigenvector of length of the assembly size')
    traj.f_add_result('$set.$.min_n_estimate', min_n_estimate,
                      comment='Smallest contribution to the estimated relevant subset of the eigenvector')
    traj.f_add_result('$set.$.min_n_exact', min_n_exact,
                      comment='Smallest contribution to the exact relevant subset of the eigenvector')
    traj.f_add_result('$set.$.D_KL', DKL,
                      comment='Kullback-Leidler-Divegence as tuple (D_KL(P||Q),D_KL(Q||P))')
    traj.f_add_result('$set.$.D_KS', DKS,
                      comment='Kolmogorov-Smirnov-Distance as tuple (D_KS,p-value)')
    traj.f_add_result('$set.$.MWU', MWU,
                      comment='Mann-Whitney-U statistic as tuple (MWU,p-value)')
    traj.f_add_result('$set.$.Vector_angles', ref_vector_angles,
                      comment='Angles between the i-th eigenvectors respectively')
    traj.f_add_result('$set.$.Space_angle', ref_space_angle,
                      comment='Angle between the eigenspaces')


# Set up Environment and Trajectory

trajname = '21corr_20T_9Asize_11bkgr_5rep_memtest'

env = Environment(trajectory=trajname,
                  add_time=True,
                  file_title=trajname,
                  comment='Exploration of large 4D parameter room. '
                          'Investigation of the correlation structure and'
                          'comparison to identical network without assemblies.',
                  filename=base_path + '/ParameterExploration/assembly/',
                  # large_overview_tables=True,
                  # git_repository=base_path,
                  # overwrite_file=True,
                  log_folder=base_path + '/ParameterExploration/logs/',
                  log_multiproc=True,
                  results_per_run=15,
                  multiproc=True,
                  ncores=1,
                  use_pool=True,
                  freeze_input=True,
                  wrap_mode='QUEUE')

traj = env.trajectory

traj.f_add_parameter('N', 100, comment='Number of neurons')
traj.f_add_parameter('corr', .0, comment='Correlation within assembly')
traj.f_add_parameter('T', 100, comment='Runtime')
traj.f_add_parameter('rate', 100, comment='Mean spiking rate')
traj.f_add_parameter('A_size', 10, comment='size of assembly')
traj.f_add_parameter('bkgr_corr', .0, comment='Background correlation')
traj.f_add_parameter('repetition', 0, comment='Iterator to produce statistics')

# Test exploration

traj.f_explore(cartesian_product({'corr': [.1, .12],
                                  'T': [2000, 3000],
                                  'repetition': [2, 3],
                                  'A_size': [8, 10],
                                  'bkgr_corr': [.05, .1],
                                  }))

# Full exploration

# traj.f_explore(cartesian_product({'corr': [.0, .01, .02, .03, .04, .05,
#                                            .06, .07, .08, .09, .1,
#                                            .12, .14, .16, .18, .2,
#                                            .25, .3, .35, .4, .5],
#                                   'T': [200, 400, 600, 800, 1000,
#                                         1200, 1400, 1600, 1800, 2000,
#                                         2200, 2400, 2600, 2800, 3000,
#                                         3200, 3400, 3600, 3800, 4000],
#                                   'A_size': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#                                   'bkgr_corr': [.0, .01, .02, .03, .04, .05,
#                                                 .06, .07, .08, .09, .1],
#                                   'repetition': [0, 1, 2, 3, 4]
#                                   }))

# Surpress pickling of config data to speed up runs

traj.f_store()
traj.config.f_remove(recursive=True)

# Run exploration

env.run(assembly_detection)

env.disable_logging()


print("--- %s seconds ---" % (time.time() - start_time))
print h.heap()

sys.exit(0)



