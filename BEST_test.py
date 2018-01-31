from __future__ import division
import best.plot
import best
import numpy as np
from pymc import Uniform, Normal, Exponential, NoncentralT, deterministic, Model, MCMC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# This example reproduces Figure 3 of
#
# Kruschke, J. (2012) Bayesian estimation supersedes the t
#     test. Journal of Experimental Psychology: General.
#
# According to the article, the data were generated from t
# distributions of known values.

def make_model(data):
    assert len(data) == 2, 'There must be exactly two data arrays'
    name1, name2 = sorted(data.keys())
    y1 = np.array(data[name1])
    y2 = np.array(data[name2])
    assert y1.ndim == 1
    assert y2.ndim == 1
    y = np.concatenate((y1, y2))

    mu_m = np.mean(y)
    mu_p = 0.000001 * 1 / np.std(y) ** 2

    sigma_low = np.std(y) / 1000
    sigma_high = np.std(y) * 1000

    # the five prior distributions for the parameters in our model
    group1_mean = Normal('group1_mean', mu_m, mu_p)
    group2_mean = Normal('group2_mean', mu_m, mu_p)
    group1_std = Uniform('group1_std', sigma_low, sigma_high)
    group2_std = Uniform('group2_std', sigma_low, sigma_high)
    nu_minus_one = Exponential('nu_minus_one', 1 / 29)

    @deterministic(plot=False)
    def nu(n=nu_minus_one):
        out = n + 1
        return out

    @deterministic(plot=False)
    def lam1(s=group1_std):
        out = 1 / s ** 2
        return out

    @deterministic(plot=False)
    def lam2(s=group2_std):
        out = 1 / s ** 2
        return out

    group1 = NoncentralT(name1, group1_mean, lam1, nu, value=y1,
                         observed=True)
    group2 = NoncentralT(name2, group2_mean, lam2, nu, value=y2,
                         observed=True)
    return Model({'group1': group1,
                  'group2': group2,
                  'group1_mean': group1_mean,
                  'group2_mean': group2_mean,
                  'group1_std': group1_std,
                  'group2_std': group2_std,
                  })

drug = [101,100,102,104,102,97,105,105,98,101,100,123,105,103,100,95,102,106,
        109,102,82,102,100,102,102,101,102,102,103,103,97,97,103,101,97,104,
        96,103,124,101,101,100,101,101,104,100,101]
placebo = [99,101,100,101,102,100,97,101,104,101,102,102,100,105,88,101,100,
           104,100,100,100,101,102,103,97,101,101,100,101,99,101,100,100,
           101,100,99,101,100,102,99,100,99]

data = {'drug':drug,'placebo':placebo}

model = make_model(data)

M = MCMC(model)
M.sample(iter=100, burn=2)


N1 = len(M.get_node("drug").value)
N2 = len(M.get_node("placebo").value)
data_size = [N1, N2]

posterior_mean1 = M.trace('group1_mean')[:]
posterior_mean2 = M.trace('group2_mean')[:]
diff_means = posterior_mean1 - posterior_mean2

posterior_std1 = M.trace('group1_std')[:]
posterior_std2 = M.trace('group2_std')[:]

pooled_var = ((N1 - 1) * posterior_std1 ** 2
           + (N2 - 1) * posterior_std2 ** 2) / (N1 + N2 - 2)

effect_size = diff_means / np.sqrt(pooled_var)

stats = best.calculate_sample_statistics(effect_size)

score_HDI = (stats['hdi_min'], stats['hdi_max'])

score = stats['mode']

print score

# print M.nodes.keys
# print M.get_node('drug').value

# fig = best.plot.make_figure(M)

plt.show()
