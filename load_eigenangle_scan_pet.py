from pypet import Trajectory
import numpy as np
from os import listdir
from os.path import expanduser
from h5py import File
from time import time
import seaborn as sns
import matplotlib.pyplot as plt

data_path = '/home/robin/Projects/pop_cch_results/'
filename = '9A_13rho_result.h5'

traj = Trajectory(filename=data_path + filename)
traj.f_load(index=-1, load_parameters=0,
            load_results=2, force=True)
traj.v_auto_load = True

assembly_sizes = [[2], [3], [4], [5], [6], [7], [8], [9], [10]]
correlations = [.0, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1.]

cpp_cpp_o2o = np.zeros((len(assembly_sizes),len(correlations),2))
cpp_cpp_a2a = np.zeros((len(assembly_sizes),len(correlations),2))

pwc_pwc_o2o = np.zeros((len(assembly_sizes),len(correlations),2))
pwc_pwc_a2a = np.zeros((len(assembly_sizes),len(correlations),2))

cpp_pwc_o2o = np.zeros((len(assembly_sizes),len(correlations),2))
cpp_pwc_a2a = np.zeros((len(assembly_sizes),len(correlations),2))

for A_count, A in enumerate(assembly_sizes):
    for cc_count, cc in enumerate(correlations):
        run_count = A_count * len(correlations) + cc_count
        run = traj.results['run_{:08d}'.format(run_count)]

        cpp_cpp_o2o[A_count][cc_count][0] = run['cpp_cpp_o2o_mean']
        cpp_cpp_o2o[A_count][cc_count][1] = run['cpp_cpp_o2o_std']
        cpp_cpp_a2a[A_count][cc_count][0] = run['cpp_cpp_a2a_mean']
        cpp_cpp_a2a[A_count][cc_count][1] = run['cpp_cpp_a2a_std']

        pwc_pwc_o2o[A_count][cc_count][0] = run['pwc_pwc_o2o_mean']
        pwc_pwc_o2o[A_count][cc_count][1] = run['pwc_pwc_o2o_std']
        pwc_pwc_a2a[A_count][cc_count][0] = run['pwc_pwc_a2a_mean']
        pwc_pwc_a2a[A_count][cc_count][1] = run['pwc_pwc_a2a_std']

        cpp_pwc_o2o[A_count][cc_count][0] = run['cpp_pwc_o2o_mean']
        cpp_pwc_o2o[A_count][cc_count][1] = run['cpp_pwc_o2o_std']
        cpp_pwc_a2a[A_count][cc_count][0] = run['cpp_pwc_a2a_mean']
        cpp_pwc_a2a[A_count][cc_count][1] = run['cpp_pwc_a2a_std']

sns.set(palette=sns.color_palette("Set2", 8) + [(0.4, 0.4, 0.4)],
        style='ticks', context='poster')
fig, ax = plt.subplots(figsize=(8, 6))
for A_count, A in enumerate(assembly_sizes):
    curr_handle, = ax.plot(correlations, cpp_cpp_o2o[A_count,:,0])
    ax.fill_between(correlations,
                    cpp_cpp_o2o[A_count,:,0]-cpp_cpp_o2o[A_count,:,1],
                    cpp_cpp_o2o[A_count,:,0]+cpp_cpp_o2o[A_count,:,1],
                    color=curr_handle.get_color(),
                    alpha=0.1)
plt.show()
