import numpy as np
import matplotlib.pyplot as plt
import imp
from elephant.spike_train_surrogates import *
from elephant.statistics import mean_firing_rate, cv, isi
from quantities import Hz, ms
from scipy.linalg import eigh, norm
from neo.core import SpikeTrain

statistics_path = './validation/dist.py'
dist = imp.load_source('*', statistics_path)

matrix_analysis_path = './validation/matrix.py'
matstat = imp.load_source('*', matrix_analysis_path)

test_data_path = './validation/test_data.py'
testdata = imp.load_source('*', test_data_path)

viziphant_path = '../INM6/Tasks/viziphant/plots/generic.py'
vizi = imp.load_source('*', viziphant_path)

N = 100
sample_ccs = np.eye(N)
for i in range(N):
    for j in np.arange(i+1,N):
        cc = np.random.uniform(-0.1,1)
        sample_ccs[i,j] = cc
        sample_ccs[j,i] = cc


def vine(d, rand_ccs):
    P = np.zeros((d,d))          # storing partial correlations
    S = np.eye(d)
    rand_ccs = np.triu(rand_ccs,1).flatten()
    np.random.shuffle(rand_ccs)
    for k in range(d):
        for i in np.arange(k+1, d):
            P[k,i] = rand_ccs[k*d + i - 0.5 * (k+1)*(k+2)]
            p = P[k,i]
            for l in np.arange(k-1,0,-1): # converting partial correlation to raw correlation
                p = p * np.sqrt((1.-P[l,i]**2)*(1.-P[l,k]**2)) + P[l,i]*P[l,k]
            if not np.isfinite(p):
                p = 0
            S[k,i] = p
            S[i,k] = p
    # permuting the variables to make the distribution permutation-invariant
    # permutation = np.random.permutation(d)
    # S = S[permutation][permutation]
    return S

N = 100
rate = 10*Hz
T = 100000*ms
binsize = 2*ms

spiketrain_list1 = testdata.test_data(size=N,
                                      corr=[.01,.02],
                                      t_stop=T,
                                      rate=rate,
                                      assembly_sizes=[5,5],
                                      method="CPP",
                                      bkgr_corr=0.0,
                                      shuffle=False)

spiketrain_list2 = testdata.test_data(size=N,
                                      corr=.5,
                                      t_stop=T,
                                      rate=rate,
                                      assembly_sizes=[2,2,2,2,2],
                                      method="CPP",
                                      bkgr_corr=0.0,
                                      shuffle=False)


def calc_ccs(sts, binsize=2*ms):
    corr_matrix = matstat.corr_matrix(sts, binsize=binsize)

    ccs = corr_matrix.flatten()

    ccs = np.delete(ccs, np.where(ccs == 1)[0])

    return corr_matrix, ccs


def number_of_sync_spikes(cc, spikenum, B):
    if not hasattr(spikenum, "__len__"):
        spikenum = [spikenum, spikenum]

    mean_spikenum = [0, 0]
    for i, sn in enumerate(spikenum):
        mean_spikenum[i] = sn / float(B)

    norm_factor = 1.
    for N, m in zip(spikenum, mean_spikenum):
        norm_factor *= (N * (1. - 2.*m) + B*m**2)

    async_term = spikenum[0] * mean_spikenum[1] \
               + spikenum[1] * mean_spikenum[0] \
               - B * mean_spikenum[0] * mean_spikenum[1]

    sync_term = 1. - 3. * mean_spikenum[0] * mean_spikenum[1]

    return (cc * np.sqrt(norm_factor) + async_term) / sync_term


def sync_spike_matrix(cc_mat, spikenums, binnum):
    N = len(spikenums)
    SSM = np.eye(N) * spikenums

    for k in range(N):
        for j in np.arange(k+1, N):
            nss = number_of_sync_spikes(cc_mat[k, j],
                                        [spikenums[k], spikenums[j]],
                                        binnum)
            nss = np.round(nss)
            SSM[k,j] = nss
            SSM[j,k] = nss

    return SSM


def minHOC(SSM, spikenums, B, T=None):
    N = len(spikenums)
    st_mat = np.zeros((N,B))
    binsum = np.sum(st_mat, axis=0)
    for k in range(N):
        for j in np.arange(k+1, N):
            chance_sync = np.dot(st_mat[k], st_mat[j])
            if chance_sync:
                print "chance sync trains {},{}: {}".format(k,j,chance_sync)
            missing_sync_spikes = int(SSM[k, j] - chance_sync)
            if missing_sync_spikes > 0:
                min_idx = np.argsort(binsum)[:missing_sync_spikes]
                st_mat[k,min_idx] = 1
                st_mat[j,min_idx] = 1

            binsum = np.sum(st_mat, axis=0)

        if sum(st_mat[k]) < spikenums[k]:
            print 'add {} extra sync spike'  \
                  .format(spikenums[k] - sum(st_mat[k]))
            min_idx = np.argsort(binsum)
            i = 0
            while sum(st_mat[k]) < spikenums[k]:
                if not st_mat[k][min_idx[i]]:
                    st_mat[k][min_idx[i]] = 1
                i += 1
        else:
            print 'spike overflow {} : {}' \
                  .format(k, sum(st_mat[k]) - spikenums[k])

    if T is None:
        return st_mat
    else:
        binsize = T / float(B)
        spiketrain_list = [[]] * N
        for st_count, st in enumerate(st_mat):
            spike_bins = np.where(st)[0]
            spiketimes = spike_bins * binsize + binsize / 2.
            spiketrain_list[st_count] = SpikeTrain(spiketimes, T)
        return spiketrain_list

cc_mat1, ccs1 = calc_ccs(spiketrain_list1)
cc_mat2, ccs2 = calc_ccs(spiketrain_list2)

spikenums = np.zeros(len(spiketrain_list1))
for i, st in enumerate(spiketrain_list1):
    spikenums[i] = len(st)

B = T/binsize
SSM = sync_spike_matrix(cc_mat1, spikenums, B.magnitude)
print SSM

# minHOC_stlist = minHOC(SSM, spikenums, B.magnitude, T)


art_mat = vine(N, cc_mat1)
# min_mat, ccs_min = calc_ccs(minHOC_stlist)

fig, ax = plt.subplots(nrows=3, ncols=1)
fig.tight_layout()

# dist.show(sample_ccs.flatten(), art_mat.flatten(), bins=200, ax=ax[0])

dist.show(cc_mat1.flatten(), art_mat.flatten(), bins=200, ax=ax[0])
ax[0].set_ylim(0,2.5)

matstat.plot_matrix(cc_mat1, ax=ax[1], remove_autocorr=True)

matstat.plot_matrix(art_mat, ax=ax[2], remove_autocorr=True, sort=True)


fig, ax = plt.subplots(nrows=2, ncols=1)
fig.tight_layout()

vizi.rasterplot(spiketrain_list1, ax=ax[0])
# vizi.rasterplot(minHOC_stlist, ax=ax[1])

plt.show()