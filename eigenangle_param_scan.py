import sys
import sciunit
from networkunit import models, tests, scores
from copy import copy
import numpy as np
from quantities import Hz, ms
from time import time
start_time = time()


class o2o_eigenangle_test(sciunit.TestM2M, tests.correlation_matrix_test):
    score_type = scores.weighted_angle
    params = {'all_to_all': False,
              'two_sided': False,
              'binsize': 2*ms,
              'bin_num': 5000,
              't_start': 0*ms,
              't_stop': 10000*ms}

    def compute_score(self, prediction1, prediction2):
        score = self.score_type.compute(prediction1, prediction2, **self.params)
        return score

class a2a_eigenangle_test(o2o_eigenangle_test):
    params = copy(o2o_eigenangle_test.params)
    params.update(all_to_all=True)


size = 100
rate = 10*Hz
t_start = 0*ms
t_stop = 10000*ms
binsize = 2*ms
B = int(t_stop/binsize)

assembly_sizes = [[2], [3], [4], [5], [6], [7], [8], [9], [10]]
correlations = [0, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1.]

same_cluster_o2o = np.zeros((len(assembly_sizes),len(correlations),2))
same_cluster_a2a = np.zeros((len(assembly_sizes),len(correlations),2))
temp_p1 = np.zeros(50)
temp_p2 = np.zeros(50)


for A_count, A in enumerate(assembly_sizes):
    for cc_count, cc in enumerate(correlations):
        for repeat in range(50):
            model_CPPA = models.stochastic_activity(size=size, correlations=cc,
                                                 assembly_sizes=A,
                                                 correlation_method='CPP',
                                                 t_start=t_start, t_stop=t_stop,
                                                 shuffle=False, name='CPP A')
            model_CPPB = models.stochastic_activity(size=size, correlations=cc,
                                                 assembly_sizes=A,
                                                 correlation_method='CPP',
                                                 t_start=t_start, t_stop=t_stop,
                                                 shuffle=False, name='CPP B')
            # model_PWC = models.stochastic_activity(size=size, correlations=cc,
            #                                      assembly_sizes=A,
            #                                      correlation_method='pairwise_equivalent',
            #                                      t_start=t_start, t_stop=t_stop,
            #                                      shuffle=False, name='PWC')

            test_o2o = o2o_eigenangle_test()
            test_a2a = a2a_eigenangle_test()

            prediction_A = test_o2o.generate_prediction(model_CPPA)
            prediction_B = test_o2o.generate_prediction(model_CPPB)

            score_o2o = test_o2o.compute_score(prediction_A, prediction_B)
            temp_p1[repeat] = score_o2o.pvalue
            score_a2a = test_a2a.compute_score(prediction_A, prediction_B)
            temp_p2[repeat] = score_a2a.pvalue

        same_cluster_o2o[A_count][cc_count][0] = np.mean(temp_p1)
        same_cluster_o2o[A_count][cc_count][1] = np.std(temp_p1)
        print 'o2o A:', A, ' cc: ', cc, '\t ', same_cluster_o2o[A_count][cc_count]

        same_cluster_a2a[A_count][cc_count][0] = np.mean(temp_p2)
        same_cluster_a2a[A_count][cc_count][1] = np.std(temp_p2)
        print 'a2a A:', A, ' cc: ', cc, '\t', same_cluster_o2o[A_count][cc_count]

np.save('/home/r.gutzen/Output/eigenangle_param_scan/arrays/o2o_same_cluster.npz', same_cluster_o2o)
np.save('/home/r.gutzen/Output/eigenangle_param_scan/arrays/a2a_same_cluster.npz', same_cluster_a2a)

m, s = divmod(time() - start_time, 60)
h, m = divmod(m, 60)
print 'time %d:%02d:%02d' % (h, m, s)
