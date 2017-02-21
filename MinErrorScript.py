
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
from quantities import Hz, ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef

N = 15
res = 1000

angles = []
similarity = []
for j in range(res):
    vector = 2 * np.random.random_sample((2, N)) - 1
    vector = np.absolute(vector)
    for i in range(2):
        vector[i] /= np.linalg.norm(vector[i])
    angles += [np.arccos(np.dot(vector[0], vector[1]))]
    similarity += [1 - 2 * np.dot(vector[0], vector[1])/np.pi]

plt.figure()
hist, edges = np.histogram(angles, bins=20, density=True)
plt.bar(edges[:-1], hist, np.diff(edges))

hist, edges = np.histogram(similarity, bins=20, density=True)
plt.bar(edges[:-1], hist, np.diff(edges), color='g', alpha=.3)


phi = [np.pi/(res) * i for i in range(res)]
norm = integrate.quad(lambda a: np.sin(a) ** (N - 2), 0, np.pi)[0]
f = [np.sin(2*p) ** (N - 2) / norm for p in phi]
plt.plot(phi, f, color='r')
plt.show()
