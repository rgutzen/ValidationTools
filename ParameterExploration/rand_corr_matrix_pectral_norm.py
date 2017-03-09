
import numpy as np
from scipy.linalg import eigh
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef
from quantities import ms, Hz
import time
from guppy import hpy; h=hpy()
from pypet import Environment, cartesian_product


start_time = time.time()


def draw_lmax(N, rate, t_stop, bins):
    spiketrains = np.array([HPP(rate=rate*Hz, t_stop=t_stop*ms)
                            for _ in range(N)])

    binned_sts = BinnedSpikeTrain(spiketrains, bins*ms,
                                  t_start=0*ms, t_stop=t_stop*ms)

    corr_matrix = corrcoef(binned_sts)

    EWs, __ = eigh(corr_matrix)

    return max(EWs)


def get_datapoint(N, rate, t_stop, bins, trials):
    lmax = np.zeros(trials)

    for i in range(trials):
        lmax[i] = draw_lmax(N, rate, t_stop, bins)

    return np.mean(lmax), np.sqrt(np.var(lmax))


def explore_lmax(traj):
    print "N: {}".format(traj.N)
    lmax_mean, lmax_std = get_datapoint(traj.N,
                                        traj.rate,
                                        traj.t_stop,
                                        traj.bins,
                                        traj.trials)

    traj.f_add_result('Mean_lmax', lmax_mean)
    traj.f_add_result('Std_lmax', lmax_std)


base_path = '/home/robin/Projects/ValidationTools'

# env = Environment(trajectory='N_10-200',
#                   filename=base_path + '/ParameterExploration/lmax_dist/lmax_N.hdf5',
#                   file_title='lmax_vs_N',
#                   large_overview_tables=True,
#                   overwrite_file=True,
#                   log_folder=base_path + '/ParameterExploration/logs/')

# traj = env.trajectory

# traj.f_add_parameter('N', 100)
# traj.f_add_parameter('rate', 10)
# traj.f_add_parameter('t_stop', 10000)
# traj.f_add_parameter('bins', 2)
# traj.f_add_parameter('trials', 30)

# traj.f_explore({'N': [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]})

# env.run(explore_lmax)

# env.disable_logging()

print get_datapoint(100, 100, 1000, 2, 30)

print("--- %s seconds ---" % (time.time() - start_time))
print h.heap()

