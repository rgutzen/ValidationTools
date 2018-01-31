# import sys
# sys.path.insert(0,'/home/robin/Projects/sciunit')
# sys.path.append('/home/robin/Projects/NetworkUnit')
from pypet import Environment, cartesian_product
from pypet import pypetconstants
from imp import load_source
from guppy import hpy; h=hpy()
from os.path import expanduser
import sys
import sciunit
from networkunit import models, tests, scores
from copy import copy
import numpy as np
from quantities import Hz, ms
from time import time

start_time = time()
home_path = expanduser('~')


def calculate_score(traj):
    class o2o_eigenangle_test(sciunit.TestM2M, tests.correlation_matrix_test):
        score_type = scores.weighted_angle
        params = {'all_to_all': False,
                  'binsize': traj.binsize*ms,
                  'bin_num': traj.B,
                  't_start': traj.t_start*ms,
                  't_stop': traj.t_stop*ms}

        def compute_score(self, prediction1, prediction2):
            score = self.score_type.compute(prediction1, prediction2,
                                            **self.params)
            return score


    temp_cpp_cpp_o2o = np.zeros(50)

    for i in range(50):
        model_CPPA = models.stochastic_activity(size=traj.size, correlations=traj.cc1,
                                             assembly_sizes=traj.A1, rate=traj.rate*Hz,
                                             correlation_method='CPP',
                                             t_start=traj.t_start*ms, t_stop=traj.t_stop*ms,
                                             shuffle=False, name='CPP A')
        model_CPPB = models.stochastic_activity(size=traj.size, correlations=traj.cc2,
                                             assembly_sizes=traj.A2, rate=traj.rate*Hz,
                                             correlation_method='CPP',
                                             t_start=traj.t_start*ms, t_stop=traj.t_stop*ms,
                                             shuffle=False, name='CPP B')

        test_o2o = o2o_eigenangle_test()

        pred_CPPA = test_o2o.generate_prediction(model_CPPA)
        pred_CPPB = test_o2o.generate_prediction(model_CPPB)

        temp_cpp_cpp_o2o[i] = test_o2o.compute_score(pred_CPPA, pred_CPPB).pvalue


    traj.f_add_result('cpp_cpp_o2o_mean', np.mean(temp_cpp_cpp_o2o))
    traj.f_add_result('cpp_cpp_o2o_std', np.std(temp_cpp_cpp_o2o))



def main(job_id):
    trajname = '10A_10A_fixcc02'

    env = Environment(trajectory=trajname,
                      add_time=True,
                      file_title=trajname,
                      comment='Exploration of eigenangle score for different '
                              'aritfical networks with assembly size A and '
                              'correlation rho.',
                      filename=home_path + '/Output/eigenangle_param_scan/'
                               + trajname + '_result.h5',
                      # large_overview_tables=True,
                      # git_repository=base_path,
                      # overwrite_file=True,
                      log_folder=home_path + '/Output/eigenangle_param_scan/',
                      log_multiproc=True,
                      results_per_run=12,
                      multiproc=True,
                      ncores=16,
                      # use_scoop=True,
                      use_pool=True,
                      freeze_input=True,
                      # wrap_mode=pypetconstants.WRAP_MODE_LOCAL,
                      wrap_mode='LOCAL'
                      )

    traj = env.trajectory

    traj.f_add_parameter('size', 100, comment='Number of neurons')
    traj.f_add_parameter('rate', 10, comment='Mean firing rate')
    traj.f_add_parameter('t_start', 0, comment='')
    traj.f_add_parameter('t_stop', 10000, comment='')
    traj.f_add_parameter('binsize', 2, comment='')
    traj.f_add_parameter('B', 5000, comment='Number of bins')
    traj.f_add_parameter('job_id', 0, comment='Slurm queue id')
    traj.f_add_parameter('A1', [5], comment='Assembly size 1')
    traj.f_add_parameter('A2', [5], comment='Assembly size 2')
    traj.f_add_parameter('cc1', .2, comment='Mean correlation coefficient 1')
    traj.f_add_parameter('cc2', .2, comment='Mean correlation coefficient 2')


    traj.f_explore(cartesian_product({'A1': [[], [2], [3], [4], [5], [6], [7],
                                             [8], [9], [10]],
                                      'A2': [[], [2], [3], [4], [5], [6], [7],
                                             [8], [9], [10]],
                                      }))
    traj.f_store()
    traj.config.f_remove(recursive=True)

    env.run(calculate_score)

    assert traj.f_is_completed()

    env.disable_logging()

    m, s = divmod(time() - start_time, 60)
    h, m = divmod(m, 60)
    print 'time %d:%02d:%02d' % (h, m, s)

if __name__ == '__main__':

    main(sys.argv[-1])