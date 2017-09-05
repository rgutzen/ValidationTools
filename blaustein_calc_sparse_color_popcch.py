# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import sys
from time import time
from mpi4py import MPI


def load(filename, rescale=False, return_pairs=True):
    file = np.load(filename)
    if return_pairs:
        if rescale:
            return np.squeeze(file['cch_array']) / file['max_cc'][0], \
                   file['split_pairs']
        else:
            return np.squeeze(file['cch_array']), file['split_pairs']
    else:
        if rescale:
            return np.squeeze(file['cch_array']) / file['max_cc'][0]
        else:
            return np.squeeze(file['cch_array'])


def cch_space(squeezed_cch_array, threshold=.5, save_id=0, **kwargs):
    # save as array with only >threshold value
    # and a second array with their (i,j,t)
    N = len(squeezed_cch_array)
    B = len(squeezed_cch_array[0])
    dim = int(.5 * (1 + np.sqrt(8. * N + 1)))
    pairs = np.triu_indices(dim, 1)
    pair_ids = [[i, j] for i, j in zip(pairs[0], pairs[1])]
    binnums = np.arange(B)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    Nnodes = comm.Get_size()
    print 'rank', rank, 'and size (Nnodes)', Nnodes
    comm.Barrier()
    if rank == 0:
        split = np.array_split(squeezed_cch_array, Nnodes)
    else:
        split = None
    cch_per_node = int(np.ceil(float(len(squeezed_cch_array)) / Nnodes))
    split_cchs = comm.scatter(split, root=0)

    # color_array_inst = np.zeros((cch_per_node, B), dtype=int)
    color_array_inst = np.array([], dtype=int)
    pair_tau_ids = np.array([0,0,0], dtype=int)
    for count, cch in enumerate(split_cchs):
        mask = (cch >= threshold)
        int_color_cch = (cch[mask] - threshold) / (1. - threshold) * 10.
        i,j = pair_ids[count]
        if binnums[mask].size:
            pair_tau_ids = np.vstack((pair_tau_ids,
                                      np.array([(i,j,t) for t in binnums[mask]])))
            color_array_inst = np.append(color_array_inst,
                                         int_color_cch.astype(int))
    color_array = comm.gather(color_array_inst, root=0)
    # pair_tau_ids = comm.gather(pair_tau_ids, root=0)
    pair_tau_ids = pair_tau_ids[1:]

    if rank == 0:
        print np.shape(color_array)

    np.savez('/home/r.gutzen/pop_cch_results/color_array_set{}_th{}.npz'
             .format(save_id, threshold),
             color_array=color_array, pair_tau_ids=pair_tau_ids,
             threshold=threshold, B=B, N=N)
    return None


# ----- MAIN -----

start_time = time()

argin = sys.argv
try:
    task_id = int(argin[1])
except IndexError or ValueError:
    task_id = 0

try:
    threshold = int(argin[2])
except IndexError:
    threshold = 0.5

path = '/home/r.gutzen/pop_cch_results/'
filename = 'cch_array_set{}_bin2ms_lag100bins.npz'.format(task_id)

ccharray = load(path+filename, rescale=True, return_pairs=False)

cch_space(abs(np.squeeze(ccharray)), threshold=threshold, save_id=task_id)

m, s = divmod(time() - start_time, 60)
h, m = divmod(m, 60)
print 'Computation took %d:%02d:%02d' % (h, m, s)
