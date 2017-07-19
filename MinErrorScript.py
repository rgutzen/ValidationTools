from validation import matrix, test_data
import matplotlib.pyplot as plt
import numpy as np
from quantities import Hz, ms
import best
import best.plot
from pymc import MCMC
from scipy.misc import comb
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

N = 100
T = 10000*ms
binsize = 2.*ms
B = T/binsize
rate = 10*Hz
A_size = [5,3,3,3]
assembly_corr = [0.2, 0.05,0.2,0.2]
sync_prob_assembly = [test_data.corr_to_sync_prob(cc, size, rate, T, B)
                      for cc, size in zip(assembly_corr, A_size)]

nbr_of_pairs = [size * (size-1) / 2 for size in A_size]
pw_synchrony = [test_data.transform_sync_prob(syprob, size, rate, T, B, A_size_1=2)
                for syprob, size in zip(sync_prob_assembly, A_size)]
pw_sync_list = []
for pws, pairnbr in zip(pw_synchrony, nbr_of_pairs):
    pw_sync_list += ([pws] * pairnbr)

spiketrains_1 = test_data.test_data(size=N,
                                    t_stop=T,
                                    rate=rate,
                                    corr=sync_prob_assembly,
                                    assembly_sizes=A_size,
                                    # corr=[pw_synchrony[0]] * nbr_of_pairs[0]
                                    #      + [pw_synchrony[1]] * nbr_of_pairs[1],
                                    # assembly_sizes=[2] * sum(nbr_of_pairs),
                                    shuffle=False
                                    )

spiketrains_2 = test_data.test_data(size=N,
                                    t_stop=T,
                                    rate=rate,
                                    corr=sync_prob_assembly,
                                    assembly_sizes=A_size,
                                    # corr=pw_sync_list,
                                    # assembly_sizes=[2] * sum(nbr_of_pairs),
                                    shuffle=False,
                                    )

corr_mat = matrix.corr_matrix(spiketrains_1)
__, EVs_1 = np.linalg.eigh(corr_mat)

corr_mat = matrix.corr_matrix(spiketrains_2)
__, EVs_2 = np.linalg.eigh(corr_mat)

angles, _ = matrix.EV_angles(EVs_1, EVs_2, deg=False, mute=True,
                             all_to_all=True)

fig, ax = plt.subplots(1, 1)
pvalue, rangles = matrix.angle_significance(angles.tolist(),
                                            dim=N, abs=True,
                                            ax=ax, res=10**7,
                                            s=1./N**2, sig_level=0.01,
                                            rand_angles=None)

# BESTdata = {'Random':rangles,
#             'Eigen':angles}
# BESTmodel = best.make_model(BESTdata, separate_nu=True)
# M = MCMC(BESTmodel)
# M.sample(iter=110000, burn=10000)
#
# fig = best.plot.make_figure(M)

plt.show()
