from validation import matrix, test_data
import matplotlib.pyplot as plt
import numpy as np
from quantities import Hz, ms
import best
import best.plot
from pymc import MCMC
from scipy.misc import comb
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

N = 100
T = 100000*ms
binsize = 2.*ms
B = T/binsize
rate = 10*Hz
A_size = []
sync_prob_assembly = 0.

# nbr_of_pairs = [size * (size-1) / 2 for size in A_size]
# pw_synchrony = [test_data.transform_sync_prob(syprob, size, rate, T, B, A_size_1=2)
#                 for syprob, size in zip(sync_prob_assembly, A_size)]

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
                                    # corr=[pw_synchrony[0]] * nbr_of_pairs[0]
                                    #    + [pw_synchrony[1]] * nbr_of_pairs[1],
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
pvalues, rangles = matrix.angle_significance(angles.tolist(),
                                             dim=N, abs=True,
                                             ax=ax, res=10**7)

xs = len(np.where(np.array(pvalues) < 2. / (N*(N+1)))[0])
xn = len(np.where(np.array(pvalues) < .05)[0])
print np.array(pvalues)[np.where(np.array(pvalues) < 2. / (N*(N+1)))[0]]
print np.array(pvalues)[np.where(np.array(pvalues) < 1. / N)[0]]

def Prob(s,x,n):
    comb_prob = [s**j * (1.-s)**(n-j) * comb(n, j) for j in range(x)]
    return 1. - sum(comb_prob)

print r"\# $(\theta) = {}, p_{:.2f} = {}$".format(xn, 0.05, Prob(.05, xn, N))
print r"\# $(\theta) = {}, p_{:.4f} = {}$".format(xs, 2./(N*(N-1)), Prob(2./(N*(N-1)), xs, N))

# BESTdata = {'Random':rangles,
#             'Eigen':angles}
# BESTmodel = best.make_model(BESTdata, separate_nu=True)
# M = MCMC(BESTmodel)
# M.sample(iter=110000, burn=10000)
#
# fig = best.plot.make_figure(M)

plt.show()
