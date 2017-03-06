from pypet import Trajectory
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.set_color_codes('colorblind')
sns.despine()

def load_traj(filename):
    base_path = '/home/robin/Projects/ValidationTools/pet/assembly/'

    traj = Trajectory(filename=base_path + filename)

    traj.f_load(index=-1, load_parameters=2, load_results=2)
    traj.v_auto_load = True

    return traj


def evaluate(x_name, z_name, y_name=None, fix_param=[], fix_value=[],
             lim=(0, False), color=None, ax=None,):

    if fix_param:
        assert len(fix_param) == len(fix_value)

    if not lim[1]:
        lim = (lim[0], len(traj.f_get_run_names()[:]))

    def get_z(ztr, zlim):
        runlist = traj.f_get_run_names()[zlim[0]:zlim[1]]
        return [traj.f_get_from_runs(ztr)[run].f_get(ztr) for run in runlist]

    x = traj.parameters.f_get(x_name)[lim[0]:lim[1]]
    z = get_z(z_name, lim)

    if fix_value:
        filter_idx = []
        nbr_of_runs = len(traj.f_get_run_names()[lim[0]:lim[1]])
        for param, value in zip(fix_param, fix_value):
            param_values = traj.parameters.f_get(param)[lim[0]:lim[1]]
            filter_idx += [[i for i, v in enumerate(param_values)
                            if v == value]]

        combined_filter_idx = [i for i in range(nbr_of_runs)
                               if not
                               sum([i not in idx for idx in filter_idx])]

        x = [x[i] for i in combined_filter_idx]
        z = [z[i] for i in combined_filter_idx]

        if y_name is not None:
            y = traj.parameters.f_get(y_name)[lim[0]:lim[1]]
            y = [y[i] for i in combined_filter_idx]

    if ax is not None:
        if y_name is None:
            label = ""
            for p, v in zip(fix_param, fix_value):
                label += "{}={}  ".format(p,v)
            if color is None:
                ax.plot(x, z, label=label)
            else:
                ax.plot(x, z, label=label, color=color)
            ax.set_xlabel(x_name)
            ax.set_ylabel(z_name)

    return x, z


def evaluate_repetition(x_name, z_name, y_name=None, fix_param=[],
                        fix_value=[],
                        lim=(0, False), color=None, ax=None):
    assert y_name is None
    nbr_of_reps = max(traj.parameters.f_get('repetition')[:]) + 1
    combined_response = []
    for r in range(nbr_of_reps):
        x, z = evaluate(x_name, z_name, y_name, fix_param + ['repetition'],
                        fix_value + [r], lim)
        combined_response += [z]

    combined_response = map(list, zip(*combined_response))
    mean_response = [np.mean(col) for col in combined_response]
    std_response = [np.sqrt(np.var(col)) for col in combined_response]
    upper_std = [mean + std for mean, std in zip(mean_response, std_response)]
    lower_std = [mean - std for mean, std in zip(mean_response, std_response)]

    if ax is not None:
        label = ""
        if color is None:
            color = 'r'
        for p, v in zip(fix_param, fix_value):
            label += "{}={}  ".format(p, v)
        ax.plot(x, mean_response, label=label)
        ax.fill_between(x, upper_std, lower_std, alpha=.2)
        ax.set_xlabel(x_name)
        ax.set_ylabel(z_name)


def print_table(table):
    col_width = [max(len(str(x)) for x in col)
                 for col in table]

    for i in range(len(table[0])):
        for j, column in enumerate(table):
            print "{:{}}\t\t".format(column[i], col_width[j]),
        print ""


def show_params(traj):
    params = [key.split('.')[-1] for key in traj._parameters.keys()]
    comments = [traj.f_get(p).v_comment for p in params]
    defaults = [traj.f_get(p).f_get() for p in params]
    has_range = [traj.f_get(p).f_has_range() for p in params]
    ranges = [np.unique(np.array(traj.f_get(p).f_get_range())) if has else []
              for p, has in zip(params, has_range)]
    table = [params, defaults, ranges, comments]

    print "\n\033[4m\033[94mParameters\033[0m\033[0m\n"
    print_table(table)


def show_results(traj):
    runlist = traj.f_get_run_names()[:]
    names = [key.split('.')[-1] for key in traj.f_get_results().keys()]
    names = np.unique(np.array(names))

    def _get_results(name):
        return [traj.f_get_from_runs(name)[run] for run in runlist]

    results = [_get_results(name) for name in names]
    comments = [result[0].v_comment for result in results]

    def _get_values(name):
        return [traj.f_get_from_runs(name)[run].f_get(name) for run in runlist]

    values = [_get_values(name) for name in names]

    table = [names, comments, values]

    print "\n\033[4m\033[94mResults\033[0m\033[0m\n"
    print_table(table)


traj = load_traj('corr_T.hdf5')

show_params(traj)

# show_results(traj)

fig, ax = plt.subplots(nrows=1, ncols=1)
fig.tight_layout()

for corr in [0., .02, .04, .06, 0.08, .1]:
    evaluate_repetition('T', 'EW',
                        fix_param=['corr'],
                        fix_value=[corr],
                        ax=ax)

# evaluate_repetition('T', 'Norm_estimate',
#                     fix_param=['corr'],
#                     fix_value=[0.08],
#                     color='r',
#                     ax=ax)

ax.legend()
plt.show()

