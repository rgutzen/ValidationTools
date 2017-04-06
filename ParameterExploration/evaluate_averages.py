import numpy as np
from os import listdir
from os.path import expanduser
from h5py import File
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='ticks', palette='Set2', context='paper')


def _print_table(table):
    col_width = [max(len(str(x)) for x in col)
                 for col in table]

    for i in range(len(table[0])):
        for j, column in enumerate(table):
            print "{:{}}\t\t".format(column[i], col_width[j]),
        print ""
    return None


def _show_params():
    u_values = {}
    param_names = [str(key) for key in f['Parameters'].keys()]
    for param in param_names:
        u_values[param] = np.unique(_get_arrays(param=param, fixed={})[0])
    print 'START PARAMETERS:\n'
    _print_table([param_names, [u_values[name] for name in param_names]])

    print '\nRESULT PARAMETERS:\n'
    for name in [str(key) for key in f['Results'].keys()]:
        print name + ', ',
    return None


def _get_arrays(param=None, result=None, fixed=default_params):
    param_names = [str(key) for key in f['Parameters'].keys()]
    result_names = [str(key) for key in f['Results'].keys()]
    fixed_names = [str(key) for key in fixed.keys()]

    # create mask
    N = f['Parameters/{}'.format(param_names[0])].__len__()
    mask = np.ones(N).astype(bool)
    for fix in fixed_names:
        param_array = f['Parameters/{}'.format(fix)].__array__()
        index_array = np.where(param_array == fixed[fix])[0]
        new_mask = np.array([x in index_array for x in range(N)])
        mask *= new_mask

    # parameter
    if param is not None:
        assert param in param_names
        assert param not in fixed_names
        param = f['Parameters/{}'.format(param)].__array__()[mask]

    # result
    if result is not None:
        assert result in result_names
        result = (f['Results/{}/avg'.format(result)].__array__()[mask],
                  f['Results/{}/var'.format(result)].__array__()[mask])

    return param, result


def plot(x, z, y=None, fixed=default_params, return_fixed=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 20))

    ax.set_xlabel(x)
    ax.set_ylabel(z)

    assert type(x) == str and type(z) == str
    fixed_edit = fixed
    del fixed_edit[x]
    x, z = _get_arrays(x, z, fixed_edit)

    fix_str = ''
    for fix in fixed_edit:
        fix_str += "{}:\t{}\n".format(fix, fixed_edit[fix])

    ax.plot(x, z[0])
    sz = np.sqrt(z[1])
    ax.fill_between(x, z[0]+sz, z[0]-sz, alpha=.5)
    ax.set_title(fix_str)

    if return_fixed:
        return ax, fixed_edit
    else:
        return ax


home_path = expanduser('~')
path = home_path + '/Sciebo/ParameterExploration_results/average/'
filename = 'avg_scanned_params_5scans.h5'

f = File(path + filename, 'r')

default_params = {'A_size': 5, 'T': 2000, 'bkgr_corr': 0, 'corr': 0.1}


if __name__ == '__main__':

    _show_params()

    plot('T', 'EW_max')

    plt.show()
