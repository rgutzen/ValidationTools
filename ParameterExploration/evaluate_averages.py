import numpy as np
from os import listdir
from os.path import expanduser
from h5py import File
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set(style='ticks', palette='Set2', context='talk')

home_path = expanduser('~')
path = home_path + '/Sciebo/ParameterExploration_results/average/'
filename = 'avg_scanned_params_30scans.h5'

f = File(path + filename, 'r')

default_params = {'A_size': 5, 'T': 2000, 'bkgr_corr': 0, 'corr': 0.1}

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


def plot(x, z, y=None, fixed=default_params, return_fixed=False,
         family=None, legend='side', ax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 20))

    assert type(x) == str and type(z) == str
    x_name = x
    z_name = z
    ax.set_xlabel(x_name)
    ax.set_ylabel(z_name)

    # exclude x value from fixed values and get x
    fixed_edit = fixed
    del fixed_edit[x_name]
    x, __ = _get_arrays(param=x_name, fixed=fixed_edit)

    # get family members or else assign single default to family
    if family is not None:
        family_name = family
        family = np.unique(_get_arrays(param=family, fixed={})[0])
        del fixed[family_name]
    else:
        family_name = fixed.keys()[0]
        family = [fixed[family_name]]

    # plot z(x) for each family member
    for family_member in family:
        fixed_edit_temp = fixed_edit
        fixed_edit_temp[family_name] = family_member
        __, z = _get_arrays(result=z_name, fixed=fixed_edit_temp)

        assert len(x) == len(z[0])
        labelstr = "{} = {}".format(family_name, family_member)
        curr_handle, = ax.plot(x, z[0], lw=2, label=labelstr)
        sz = np.sqrt(z[1])
        ax.fill_between(x, z[0] + sz, z[0] - sz, alpha=.3)

        if len(family)-1 and legend == 'side':
            if family_member == family[0]:
                ax.text(1.02, 1, '{}'.format(family_name),
                        transform=ax.transAxes)
            ax.text(1.02*x[-1], z[0][-1],
                    '{}'.format(family_member),
                    fontsize=14, fontweight='bold',
                    color=curr_handle.get_color())

    # show family legend
    if len(family)-1 and legend == 'convetional':
        ax.legend(handletextpad=0, fontsize=14)
        del fixed_edit[family_name]

    # show fixed values
    fix_name_str = ''
    fix_value_str = ''
    for fix in fixed_edit:
        fix_str += "\n{}\t{}".format(fixed_edit[fix], fix).expandtabs()

    ax.text(.8, 1.01, fix_str, fontsize=14, transform=ax.transAxes,
            horizontalalignment='right')
    ax.grid()
    ax.text(.01, 1.01,
            'std estimated on base N = {}'
            .format(f['Results'].attrs['base_num']),
            fontsize=12,
            transform=ax.transAxes)
    sns.despine()

    if return_fixed:
        return ax, fixed_edit
    else:
        return ax


if __name__ == '__main__':

    _show_params()

    plot(x='T', z='EW_max', family='A_size',
         fixed={'A_size': 5, 'T': 2000, 'bkgr_corr': 0, 'corr': 0.4})

    plt.show()
