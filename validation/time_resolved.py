from elephant.spike_train_correlation import cch
from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain
from quantities import ms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from simplejson import load as jsonload
from time import time


def _multiple_cch(binned_sts, rescale=False, **kwargs):
    N = len(binned_sts)
    B = binned_sts[0].num_bins*2-1
    cch_mat = np.zeros((N,N,B))
    max_cc = 0
    for i in range(N):
        for j in np.arange(i+1,N):
            cch_mat[i,j] = cch(binned_sts[i],
                               binned_sts[j],
                               **kwargs)[0].reshape((B,))
            if rescale:
                max_cc = max([max_cc, max(abs(cch_mat[i,j]))])
    if rescale:
        print "Rescaled by factor {}".format(max_cc)
        return cch_mat/float(max_cc)
    else:
        return cch_mat


def _alpha(color_inst, a):
    return [el + (1.-el)*a for el in np.array(color_inst)]


def bin_spiketrains(st_list, binsize=2*ms):
    t_lims = [(st.t_start, st.t_stop) for st in st_list]
    tmin = min(t_lims, key=lambda f: f[0])[0]
    tmax = max(t_lims, key=lambda f: f[1])[1]

    bsts = [BinnedSpikeTrain(st, binsize=binsize, t_start=tmin, t_stop=tmax)
            for st in st_list]

    return bsts


def eval_cch(bsts, threshold=.0, plot=True, title="Set0", save_idx=True,
             **kwargs):

    N = len(bsts)
    B = bsts[0].num_bins * 2 - 1

    CCH = _multiple_cch(bsts, rescale=True, **kwargs)

    if plot:
        fig = plt.figure()
        palette = sns.color_palette('coolwarm')[::]
        ax = fig.gca(projection='3d')
        ax.set_xlabel('Neuron #1')
        ax.set_xlim3d(0, N)
        ax.set_ylabel('Tau')
        ax.set_ylim3d(0, B)
        ax.set_zlabel('Neuron #2')
        ax.set_zlim3d(0, N)
    else:
        ax = None

    tau = np.arange(B)
    sig_cc = []
    print "\n#### new set ####\n"
    for i in range(N):
        for j in np.arange(i+1, N):
            for ti in range(B-2):
                # cc = CCH[i,j,ti]
                cc = (CCH[i,j,ti] + 1) / 2.
                if cc < 0:
                    print "({} {}): Index Error {} -> 0".format(i,j,cc)
                    cc = 0.
                elif cc > 1:
                    print "({} {}): Index Error {} -> 1".format(i,j,cc)
                    cc = 1.
                if cc > threshold:
                    # print "{} {}".format(i,j)
                    sig_cc += [(i,j)]
                    if plot:
                        color_i = int(np.ceil(cc*len(palette))-1)
                        color = _alpha(palette[color_i], 1.-cc)
                        ax.plot([j,j], tau[ti:ti+2], [cc*2. + i]*2, c=color)

    print "=> {} significant entries".format(len(sig_cc))

    if save_idx:
        idfile = open('sig_cc_idx_.txt'.format(title), 'w')
        for item in sig_cc:
            idfile.write("{} \n".format(item))
        idfile.close()

    return CCH, ax


def time_resolved_cc_distribution(binned_spiketrains=None, CCHs=None,
                                  ax=plt.gca(), bins=100, **kwargs):

    if binned_spiketrains is not None:
        N = len(binned_spiketrains)
        B = binned_spiketrains[0].num_bins * 2 - 1
        CCHs = np.empty(N**2 * B)
        for i in range(N):
            for j in np.arange(i + 1, N):
                idx = int(round((i*(N-.5*(i+1))+j-i-1))*B)
                CCHs[idx:idx+B] = cch(binned_spiketrains[i],
                                      binned_spiketrains[j],
                                      **kwargs)[0].reshape((B,))
    elif CCHs is None:
        return None

    ct, edges = np.histogram(CCHs, bins=bins, density=True)
    dx = np.diff(edges)[0]
    xvalues = edges[:-1] + dx/2.
    ax.plot(xvalues, ct)
    return ax, edges


if __name__ == '__main__':

    f = open('/home/robin/Projects/polychrony_data/data/polychronousspiekdata_new.txt', 'r')
    spikedata = jsonload(f)
    f.close()

    def dict_to_neo(spiketraindict, t_stop=10000 * ms):
        sts = []
        for idx_str in spiketraindict.keys():
            sts += [SpikeTrain(spiketraindict[idx_str], units='ms',
                               t_stop=t_stop, idx=idx_str)]
        return sts

    spiketrains = [dict_to_neo(data) for data in spikedata]

    start_time = time()

    # set = 0
    # bsts = bin_spiketrains(spiketrains[set], binsize=2*ms)

    # use cross_corr_coef=True for firing rate correction
    # CCH, ax = eval_cch(bsts, threshold=.8, border_correction=False,
    #                    cross_corr_coef=True)
    # ax.set_title('Set {}'.format(set))


    ## Generalized Correlation distributions
    bsts_lists = [bin_spiketrains(sts, binsize=2*ms) for sts in spiketrains]
    fig, ax = plt.subplots(1, 1)
    edges = np.linspace(-.25, .25, 200)
    for bsts in bsts_lists:
        time_resolved_cc_distribution(binned_spiketrains=bsts, ax=ax,
                                      bins=edges, cross_corr_coef=True)

    print("--- %s seconds ---" % (time() - start_time))

    plt.show()
