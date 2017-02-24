
import numpy as np
import matplotlib.pyplot as plt
from numpy import log
import scipy.integrate as integrate
from elephant.spike_train_generation import homogeneous_poisson_process as HPP
from quantities import Hz, ms
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef

N = 20
res = 1000

angles = []
similarity = []
distance = []
for j in range(res):
    vector = np.random.normal(size=(2, N))
    for i in range(2):
        vector[i] /= np.linalg.norm(vector[i])
    # vector is uniform (relative to surface) distributed on hypersphere
    vector = np.absolute(vector)
    angles += [np.arccos(np.dot(vector[0], vector[1]))]
    distance += [np.dot(vector[0], vector[1])]
    similarity += [1 - 2 * np.dot(vector[0], vector[1])/np.pi]

# plt.figure()
hist, edges = np.histogram(angles, bins=20, density=True)
plt.bar(edges[:-1], hist, np.diff(edges)*.9)

# hist, edges = np.histogram(similarity, bins=20, density=True)
# plt.bar(edges[:-1], hist, np.diff(edges), color='g', alpha=.3)

# hist, edges = np.histogram(distance, bins=20, density=True)
# plt.bar(edges[:-1], hist, np.diff(edges), color='r', alpha=.3)

step = np.pi/float(2*res)
phi = np.array([step*i for i in range(res)])
def f_I_func(p):
    return np.sin(p)**(N-2)

def f_II_func(f):
    f_I_d = np.array([f_I_func(2*p) for p in phi])/integrate.quad(f_I_func, 0, np.pi/2)[0]
    # for n in np.arange(N-1)+1:
    f = np.convolve(f, f_I_d, mode='same')#[::2]
    return f

f_I = np.array([f_I_func(p) for p in phi])/integrate.quad(f_I_func, 0, np.pi/2)[0]
f_II = f_II_func(f_I)
f_II /= (sum(f_II)*step)
f_phi = .25 * f_I + .75 * f_II
plt.plot(np.pi/2-phi, f_phi)

# x = np.array([(i-1)/float(res) for i in range(res)])
# Z1 = - np.log(1-x)
# Z = Z1
# for n in np.arange(N-1)+1:
#     Z = np.convolve(Z, Z1, mode='full')
# plt.plot(np.arccos(x), Z[::N]/(sum(Z[::N])/res), color='k')
# plt.plot(x,Z[:res]/(sum(Z[:res])/res))
# plt.plot([(i+1)/float(res) for i in range(N*res-N+1)],Z/(sum(Z)/res))

# step = np.pi/(2*res)
# phi = [step * (i+1) for i in range(res)]
# norm = integrate.quad(lambda a: np.sin(a) ** (N - 2), 0, np.pi/2)[0]
# f = [(np.sin(p)) ** (N - 2) for p in phi]
# f = [f_it / (sum(f)*step) for f_it in f]
# plt.plot(phi, f, color='r')
plt.show()
