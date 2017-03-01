
import numpy as np
from numpy import pi
import scipy as sc
import matplotlib.pyplot as plt


def generate_rand_angles(N, res, abs=True):
    phi = []
    for j in range(res):
        vector = np.random.normal(size=(2, N))
        for i in range(2):
            vector[i] /= np.linalg.norm(vector[i])
        if abs:
            vector = np.absolute(vector)
        phi += [np.arccos(np.dot(vector[0], vector[1]))]
    return np.array(phi)


def generate_uniform_angles(res):
    return np.random.rand(res) * pi


def projected_rand_angles(N, res):
    """theta = P(phi,alpha): [0,pi]x[0,pi] -> [0,pi/2]"""
    phi_array = generate_rand_angles(N, res, abs=False)
    # alpha_array = generate_uniform_angles(res)
    alpha_array = generate_rand_angles(N, res, abs=False)
    theta = np.zeros_like(phi_array)
    iterator = np.arange(res)
    for phi, alpha, i in zip(phi_array, alpha_array, iterator):

        if phi <= pi/2:

            if (phi/2.         <= alpha < pi/2 - phi/2.) \
            or (pi /2 + phi/2. <= alpha < pi   - phi/2.):
                theta[i] = phi

            elif (0            <= alpha < phi/2)\
            or   (pi/2 - phi/2 <= alpha < pi/2 + phi/2) \
            or   (pi   - phi/2 <= alpha < pi):
                theta[i] = pi - phi - 2*alpha

        elif phi > pi/2:

            if (0              <= alpha < pi/2 - phi/2) \
            or (phi/2          <= alpha < pi   - phi/2) \
            or (pi/2  + phi/2  <= alpha < pi):
                theta[i] = pi - phi - 2*alpha

            elif (pi/2 - phi/2 <= alpha < pi/2) \
            or   (pi   - phi/2 <= alpha < pi/2 + phi/2):
                theta[i] = pi - phi

    return theta

def alt_func(N,res):
    angle = np.array([pi/(2*res) * (i+1) for i in range(res)])
    f = np.array([2/pi for i in range(res)])
    f = np.convolve(f, f, mode="full")[res - 1:]
    f = f / sum(f) * res * 2 / pi
    for n in range(N-2):
        n += 2
        f_n = np.array([np.sin(2*p)**(n-2) for p in angle])
        f_n = f_n/sum(f_n) * res * 2/pi
        f = np.convolve(f, f_n, mode="full")[::2]
        f = f/sum(f) * res * 2/pi
    return angle, f

N = 3
res = 10000

plt.figure()
hist, edges = np.histogram(generate_rand_angles(N, res), bins=20, density=True)
plt.bar(edges[:-1], hist, np.diff(edges)*.9)

# hist, edges = np.histogram(projected_rand_angles(N, res), bins=edges, density=True)
# x_values = edges[:-1]
# plt.plot(x_values, hist)

angle, f = alt_func(N,res)
plt.plot(angle, f, 'r')

plt.show()


