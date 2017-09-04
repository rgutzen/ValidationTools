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
            return np.squeeze(file['cch_array'])/file['max_cc'][0]
        else:
            return np.squeeze(file['cch_array'])


def cch_space(squeezed_cch_array, threshold=.5, save_id=0, **kwargs):

    B = len(squeezed_cch_array[0])

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

    color_array_inst = np.zeros((cch_per_node, B), dtype=int)
    for count, cch in enumerate(split_cchs):
        mask = (cch >= threshold)
        int_color_cch = (cch[mask] - threshold) * (11./(1. - threshold))
        color_array_inst[count,mask] = int_color_cch.astype(int)
    color_array = comm.gather(color_array_inst, root=0)

    if rank == 0:
        print np.shape(color_array)

    np.savez('/home/r.gutzen/pop_cch_results/color_array_set{}_th{}.npz'
             .format(save_id, threshold),
             color_array=color_array, threshold=threshold)
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

# Saved color arrays only have 0 entries!

m, s = divmod(time() - start_time, 60)
h, m = divmod(m, 60)
print 'Computation took %d:%02d:%02d' % (h, m, s)
