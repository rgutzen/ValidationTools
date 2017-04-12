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
filename = 'avg_scanned_params_50scans.h5'

f = File(path + filename, 'r')

default_params = {'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2}
# corr = 0.2 with A_size=5 is equivalent to a mean pairwise correlation
# coefficient of ~0.35 within the assembly.


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


def _fix_corrcoef(cc, A):
    # remove contribution of autocorrelation in calculation of
    # mean correlation coefficent and its variance
    assert len(A) == len(cc[0])
    cc_new = np.zeros((2, len(cc[0])))
    for i, (cc_mean, cc_var) in enumerate(zip(cc[0], cc[1])):
        cc_new[0][i] = (cc[0][i] * A[i]**2 - A[i]) / (A[i]**2-A[i])
        cc_new[1][i] = 0
    return cc_new


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
        result_name = result
        result = (f['Results/{}/avg'.format(result)].__array__()[mask],
                  f['Results/{}/var'.format(result)].__array__()[mask])
        if result_name == "Corrcoef":
            if 'A_size' in fixed:
                A = np.ones_like(result[0]) * fixed['A_size']
            else:
                A = _get_arrays(param='A_size', fixed=fixed)[0]
            result = _fix_corrcoef(result, A)

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

    # get family members or else assign single default to family
    if family is not None:
        family_name = family
        family = np.unique(_get_arrays(param=family, fixed={})[0])
        if family_name not in fixed:
            fixed[family_name] = family[0]
    else:
        family_name = fixed.keys()[0]
        family = [fixed[family_name]]

    # exclude x value from fixed values and get x

    if x_name in fixed:
        del fixed[x_name]
    x, __ = _get_arrays(param=x_name, fixed=fixed)


    # plot z(x) for each family member
    for family_member in family:
        fixed[family_name] = family_member
        __, z = _get_arrays(result=z_name, fixed=fixed)

        assert len(x) == len(z[0])
        labelstr = "{} = {}".format(family_name, family_member)
        curr_handle, = ax.plot(x, z[0], lw=2, label=labelstr)
        sz = np.sqrt(z[1])
        ax.fill_between(x, z[0] + sz, z[0] - sz, alpha=.3)

        if len(family)-1 and legend == 'side':
            if family_member == family[0]:
                ax.text(1.02, .99, '{}'.format(family_name),
                        transform=ax.transAxes)
            ax.text(1.02*x[-1], z[0][-1],
                    '{}'.format(family_member),
                    fontsize=14, fontweight='bold',
                    color=curr_handle.get_color())

    if len(family)-1:
        del fixed[family_name]

    # show family legend
    if len(family)-1 and legend == 'conventional':
        ax.legend(handletextpad=0, fontsize=14)

    # show fixed values
    fix_name_str = ''
    fix_value_str = ''
    for fix in fixed:
        fix_name_str += "\n{}".format(fix)
        fix_value_str += "\n{}".format(fixed[fix])

    ax.text(.9, 1.01, fix_name_str, fontsize=14, transform=ax.transAxes,
            horizontalalignment='right')
    ax.text(1, 1.01, fix_value_str, fontsize=14, transform=ax.transAxes,
            horizontalalignment='right')

    ax.text(.05, 1.01,
            'mean & std estimated on base N = {}'
            .format(f['Results'].attrs['base_num']),
            fontsize=12,
            transform=ax.transAxes)

    ax.grid()
    ax.set_xlim(x[0], x[-1])
    sns.despine()

    if return_fixed:
        return fixed, ax
    else:
        return ax


if __name__ == '__main__':

    _show_params()

    ## What are resonable default parameters?
    # plot(x='T', z='EW_max', family='corr',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    ## What is the mean real pairwise correlation in the assembly?
    # plot(x='A_size', z='Corrcoef', family='corr',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})
    # plot(x='corr', z='Corrcoef', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    ## Is the EW an appropriate estimator for the assemblysize?
    # plot(x='A_size', z='EW_max', family='corr',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    ## How well can the assembly be detected without reference?
    # count = "SCREE_count"
    # ax1=plot(x='corr', z=count, family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})
    # ax2=plot(x='corr', z=count, family='bkgr_corr', legend='conventional',
    #      fixed={'A_size': 6, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})
    # ax3=plot(x='bkgr_corr', z=count, family='T',
    #      fixed={'A_size': 10, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    ## How well can the assembly be detect by comparison to HPP?
    # ax1=plot(x='corr', z='D_KL', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    # plot(x='corr', z='D_KS', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})
    # plot(x='corr', z='D_KS_p', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    # plot(x='corr', z='MWU', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})
    # plot(x='corr', z='MWU_p', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    ## How well can the assembly be characterized?
    # plot(x='corr', z='Redundancy', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})
    # plot(x='corr', z='EW_max', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    # plot(x='corr', z='Norm_exact', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})
    # plot(x='corr', z='Norm_estimate', family='A_size',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    # plot(x='A_size', z='min_n_exact', family='corr',
    #      fixed={'A_size': 5, 'T': 10000, 'bkgr_corr': 0, 'corr': 0.2})

    plt.show()
