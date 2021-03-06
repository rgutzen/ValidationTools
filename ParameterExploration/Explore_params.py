from matplotlib import use
from pypet import Environment, cartesian_product
from pypet import pypetconstants
from imp import load_source
from quantities import ms, Hz
from scipy.linalg import eigh, norm
import numpy as np
import time
from guppy import hpy; h=hpy()
from os.path import expanduser
import sys
from StringIO import StringIO

use('Agg')

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


def assembly_detection(traj, print_params=False):
    """"""
    # Print current parameters

    if print_params:
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

    corr_matrix = matstat.corr_matrix(spiketrain_list, binsize=traj.binsize*ms)

    # Distribution Comparison

    dist_sample_1 = corr_matrix.flatten()
    dist_sample_2 = matstat.corr_matrix(ref_spiketrain_list, binsize=traj.binsize*ms).flatten()

    DKL = dist.KL_test(dist_sample_1, dist_sample_2, mute=True)

    DKS = dist.KS_test(dist_sample_1, dist_sample_2, mute=True)

    MWU = dist.MWU_test(dist_sample_1, dist_sample_2, excl_nan=True, mute=True)

    # Correlation Analysis

    corrcoef = np.mean(corr_matrix[:traj.A_size, :traj.A_size].flatten())

    EWs, EVs = eigh(corr_matrix)
    EW_max = np.sort(EWs)[-1]

    redundancy = matstat.redundancy(EWs, mute=True)

    SCREE_count = matstat.eigenvalue_spectra(EWs, method='SCREE', mute=True)

    broken_stick_count = matstat.eigenvalue_spectra(EWs, method='broken-stick', mute=True)

    # ToDo: Estimate Assembly by above chance load in eigenvector?

    # Angle Comparison

    ref_corr_matrix = matstat.corr_matrix(ref_spiketrain_list)
    ref_EWs, ref_EVs = eigh(ref_corr_matrix)

    ref_vector_angles, ref_space_angle = matstat.EV_angles(EVs, ref_EVs, mute=True)

    spectral_count = len(np.where(EWs > np.max(ref_EWs))[0])

    # Assembly Detection

    relevant_EVs = matstat.detect_assemblies(EVs, EWs, detect_by='eigenvalue', EW_lim=max(ref_EWs), mute=True)[0]

    norm_estimate = norm(relevant_EVs)
    min_n_estimate = min(relevant_EVs)  # Is this useful?

    relevant_EVs = matstat.detect_assemblies(EVs, EWs, detect_by=traj.A_size, EW_lim=max(ref_EWs), mute=True)[0]

    norm_exact = norm(relevant_EVs)
    min_n_exact = min(relevant_EVs)

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


def main(job_id):

    # Set up Environment and Trajectory

    trajname = '21corr_11T_9Asize_11bkgr'

    env = Environment(trajectory=trajname,
                      add_time=True,
                      file_title=trajname,
                      comment='Exploration of large 4D parameter room. '
                              'Investigation of the correlation structure and'
                              'comparison to identical network without assemblies.',
                      filename=home_path + '/ParameterExploration_results/results/'
                               + trajname + '_{}'.format(str(job_id)) + '.h5',
                      # large_overview_tables=True,
                      # git_repository=base_path,
                      # overwrite_file=True,
                      log_folder=home_path + '/ParameterExploration_results/logs/',
                      log_multiproc=True,
                      results_per_run=15,
                      multiproc=True,
                      ncores=16,
                      # use_scoop=True,
                      use_pool=True,
                      freeze_input=True,
                      # wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                      wrap_mode='LOCAL'
                      )

    traj = env.trajectory

    traj.f_add_parameter('N', 100, comment='Number of neurons')
    traj.f_add_parameter('corr', .0, comment='Correlation within assembly')
    traj.f_add_parameter('binsize', 2, comment='Bin size in ms with which the correlation is computed')
    traj.f_add_parameter('T', 100, comment='Runtime')
    traj.f_add_parameter('rate', 50, comment='Mean spiking rate')  # ToDo: ???????
    traj.f_add_parameter('A_size', 10, comment='size of assembly')
    traj.f_add_parameter('bkgr_corr', .0, comment='Background correlation')
    traj.f_add_parameter('job_id', 0, comment='Iterator to produce statistics')

    # Test exploration

    # traj.f_explore(cartesian_product({'corr': [.1, .12],
    #                                   'T': [2000, 3000],
    #                                   'repetition': [2, 3],
    #                                   'A_size': [8, 10],
    #                                   'bkgr_corr': [.05, .1],
    #                                   }))

    # Large Scan

    traj.f_explore(cartesian_product({'corr': [.0, .01, .02, .03, .04, .05,
                                               .06, .07, .08, .09, .1,
                                               .12, .14, .16, .18, .2,
                                               .25, .3, .35, .4, .5],
                                      'T': [500, 1000, 2000, 3000, 4000, 5000,
                                            6000, 7000, 8000, 9000, 10000],
                                      'A_size': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                                      'bkgr_corr': [.0, .01, .02, .03, .04, .05,
                                                    .06, .07, .08, .09, .1],
                                      'job_id': [int(job_id)]
                                      }))
    # ToDo: Do more repetitions ~100

    # Surpress pickling of config data to speed up runs

    traj.f_store()
    traj.config.f_remove(recursive=True)

    # Run exploration

    env.run(assembly_detection)

    # Let's check that all runs are completed!

    assert traj.f_is_completed()

    # Finally disable logging and close all log-files

    env.disable_logging()

    print("--- %s seconds ---" % (time.time() - start_time))
    print h.heap()

    sys.exit(0)

if __name__ == '__main__':
    # This will execute the main function in case the script is called from the one true
    # main process and not from a child processes spawned by your environment.
    # Necessary for multiprocessing under Windows.

    main(sys.argv[-1])



