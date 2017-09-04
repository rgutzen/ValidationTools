from elephant.spike_train_correlation import cch
from elephant.conversion import BinnedSpikeTrain
from neo import SpikeTrain
from quantities import ms
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import sys
from simplejson import load as jsonload
from time import time
from mpi4py import MPI


def _calculate_multi_cch(sts, binsize=2*ms, maxlag=100, set_name=-1, **kwargs):
    N = len(sts)
    if maxlag is None:
        maxlag = sts[0].t_stop - sts[0].start
        maxlag = int(maxlag.rescale('ms').magnitude)
    B = 2*maxlag + 1
    pairs_idx = np.triu_indices(N, 1)
    pairs = [[i, j] for i, j in zip(pairs_idx[0], pairs_idx[1])]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    Nnodes = comm.Get_size()
    print 'rank', rank, 'and size (Nnodes)', Nnodes
    comm.Barrier()
    if rank == 0:
        split = np.array_split(pairs, Nnodes)
    else:
        split = None
    pair_per_node = int(np.ceil(float(len(pairs)) / Nnodes))
    split_pairs = comm.scatter(split, root=0)

    cch_array = np.zeros((pair_per_node, B))
    max_cc = 0
    for count, (i,j) in enumerate(split_pairs):
        binned_sts = bin_spiketrains([sts[i], sts[j]], binsize=2*ms)
        cch_array[count] = np.squeeze(cch(binned_sts[0],
                                          binned_sts[1],
                                          window=[-maxlag, maxlag],
                                          cross_corr_coef=True,
                                          **kwargs)[0])
        max_cc = max([max_cc, max(cch_array[count])])
    pop_cch = comm.gather(cch_array, root=0)
    pop_max_cc = comm.gather(max_cc, root=0)

    if rank == 0:
        print np.shape(pop_cch)

    np.savez('/home/r.gutzen/pop_cch_results/cch_array_set{}_bin{}ms_lag{}bins.npz'
             .format(set_name, int(binsize.magnitude), maxlag),
             cch_array=pop_cch, split_pairs=split_pairs, max_cc=pop_max_cc,
             binsize=binsize.magnitude)
    return None


def bin_spiketrains(st_list, binsize=2*ms):
    t_lims = [(st.t_start, st.t_stop) for st in st_list]
    tmin = min(t_lims, key=lambda f: f[0])[0]
    tmax = max(t_lims, key=lambda f: f[1])[1]

    return [BinnedSpikeTrain(st, binsize=binsize, t_start=tmin, t_stop=tmax)
            for st in st_list]


def dict_to_neo(spiketraindict, t_stop=10000 * ms):
    sts = []
    for idx_str in spiketraindict.keys():
        sts += [SpikeTrain(spiketraindict[idx_str], units='ms',
                           t_stop=t_stop, idx=idx_str)]
    return sts

# ----- MAIN -----

start_time = time()

argin = sys.argv
try:
    task_id = int(argin[-1])
except ValueError:
    task_id = 0
lag = 100

print 'set: {}\nlag: {}'.format(task_id, lag)

f = open('/home/r.gutzen/Projects/pop_cch/polychrony_data/polychronousspiekdata_new.txt', 'r')
spikedata = jsonload(f)
f.close()

spiketrains = dict_to_neo(spikedata[task_id])

_calculate_multi_cch(spiketrains, binsize=2*ms, maxlag=lag, set_name=task_id)

m, s = divmod(time() - start_time, 60)
h, m = divmod(m, 60)
print 'Computation took %d:%02d:%02d' % (h, m, s)
