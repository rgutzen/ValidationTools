from pypet import Trajectory
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.set_color_codes('colorblind')
sns.despine()


base_path = '/home/robin/Projects/ValidationTools/pet/assembly/'

filename = 'corr_vs_T.hdf5'

traj = Trajectory(filename=base_path + filename)

traj.f_load(index=-1, load_parameters=2, load_results=2)
traj.v_auto_load = True


def plot(x, z, y=None, fix_param=[], fix_value=[], lim=(0, False)):

    if fix_param:
        assert len(fix_param) == len(fix_value)

    if not lim[1]:
        lim = (lim[0], len(traj.f_get_run_names()[:]))

    def get_z(ztr, zlim):
        runlist = traj.f_get_run_names()[zlim[0]:zlim[1]]
        return [traj.f_get_from_runs(ztr)[run].f_get(ztr) for run in runlist]

    x = traj.parameters.f_get(x)[lim[0]:lim[1]]
    z = get_z(z, lim)

    if fix_value:
        filter_idx = []
        nbr_of_runs = len(traj.f_get_run_names()[lim[0]:lim[1]])
        for param, value in zip(fix_param, fix_value):
            param_values = traj.parameters.f_get(param)[lim[0]:lim[1]]
            filter_idx += [[i for i, v in enumerate(param_values)
                            if v == value]]

        combined_filter_idx = [i for i in range(nbr_of_runs)
                               if not sum([i not in idx for idx in filter_idx])]

        x = [x[i] for i in combined_filter_idx]
        z = [z[i] for i in combined_filter_idx]

        if y is not None:
            y = [y[i] for i in combined_filter_idx]

    if y is None:
        plt.plot(x, z)
        return x, z

plt.figure()
print traj.parameters.f_get('repetition')[:]
print traj.parameters.f_get('T')[:]
for r in range(max(traj.parameters.f_get('repetition')[:])+1):
    print plot('T', 'Norm_exact',
               fix_param=['corr', 'repetition'],
               fix_value=[0.2, r])
plt.show()
