
import numpy as np
from numpy import pi
import imp
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from quantities import Hz, ms
from scipy.linalg import eigh, norm
from matplotlib import rc
rc('text', usetex=True)

test_data_path = 'test_data.py'
testdata = imp.load_source('*', test_data_path)
matrix_analysis_path = 'matrix.py'
matstat = imp.load_source('*', matrix_analysis_path)


def generate_ev_angles(N, space_dim=1, abs=True):
    sts1 = testdata.test_data(N, 0, 10000 * ms, 10 * Hz, method="CPP",
                       assembly_sizes=[],
                       bkgr_corr=0., shuffle=False, shuffle_seed=None)
    sts2 = testdata.test_data(N, 0, 10000 * ms, 10 * Hz, method="CPP",
                       assembly_sizes=[],
                       bkgr_corr=0., shuffle=False, shuffle_seed=None)

    corr_matrix1 = matstat.corr_matrix(sts1)
    corr_matrix2 = matstat.corr_matrix(sts2)

    __, EVs1 = eigh(corr_matrix1)
    __, EVs2 = eigh(corr_matrix2)

    if abs:
        EVs1 = np.absolute(EVs1.T[::-1])
        EVs2 = np.absolute(EVs2.T[::-1])
    else:
        EVs1 = EVs1.T[::-1]
        EVs2 = EVs2.T[::-1]

    if space_dim == 1:
        M = np.dot(EVs1, EVs2.T)
        angles = [np.arccos(np.diag(M))]
    if space_dim == N:
        M = np.dot(EVs1, EVs2.T)
        angles = np.arccos(np.linalg.det(M))
    else:
        angles = np.zeros(N/space_dim)
        for i in np.arange(N)[::space_dim]:
            if i:
                M = np.dot(EVs1[i-space_dim:i], EVs2[i-space_dim:i].T)
                # space_angle = np.arccos(np.sqrt(np.linalg.det(np.dot(M, M.T))))
                space_angle = np.arccos(np.linalg.det(M))
                angles[i/space_dim-1] = space_angle
    return np.array(angles)


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

N = 100
res = 10000

draw_angles = np.array([70.30, 69.30, 56.78])
draw_angles *= pi/180.
color_id = [0, 2, 4]

other_angles = np.array([70.30, 69.30, 56.78, 54.09, 52.68, 53.02, 53.44, 52.49, 53.29, 50.43, 46.04, 54.77, 53.68, 52.02, 51.56, 48.76, 49.63, 50.10, 53.00, 52.22, 47.29, 54.80, 44.02, 50.84, 48.79, 46.75, 56.45, 52.88, 54.34, 55.21, 55.28, 46.28, 54.13, 46.75, 55.23, 54.55, 54.94, 47.70, 53.97, 51.75, 46.90, 50.47, 48.88, 50.00, 51.69, 52.67, 56.32, 45.88, 4.12, 57.02, 50.63, 52.84, 50.22, 45.47, 44.21, 48.92, 49.69, 46.88, 52.83, 48.44, 54.03, 53.16, 50.23, 50.94, 50.12, 51.80, 52.51, 52.57, 53.07, 52.96, 53.21, 51.97, 52.52, 48.38, 57.16, 50.48, 50.69, 47.00, 52.92, 49.59, 54.11, 47.46, 50.24, 49.19, 48.86, 53.78, 46.45, 50.95, 59.04, 53.89, 55.84, 52.20, 59.65, 53.16, 58.54, 63.58, 60.93, 72.00, 70.83, 66.49])
other_angles *= pi/180.

sns.set(style='ticks', palette='Set2', context='poster')
fontsize = 22
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,5))
# edges = np.linspace(0, pi/2, 31*pi)

# hist, __ = np.histogram(other_angles, bins=edges, density=True)
# ev_ang = ax.bar(edges[:-1], hist, np.diff(edges), color='g', edgecolor='w', label='EV angles')

# hist, ___ = np.histogram(generate_rand_angles(N, res), bins=edges, density=True)
# dx = edges[1]-edges[0]
# r_ang = ax.bar(edges[:-1], hist, np.diff(edges), color='0.3', edgecolor='w', alpha=.3)
# ax.plot(edges[:-1]+dx, hist, ls='steps', color='0.3')
dim = 1
ev_angles = np.array([])
for _ in range(100/(N/dim)):
    ev_angles = np.append(ev_angles, generate_ev_angles(N, dim, abs=False))
ev_angles = ev_angles[np.isfinite(ev_angles)]
print ev_angles

hist, edges = np.histogram(generate_rand_angles(N, res, abs=False), bins=40, density=True)
ax.bar(edges[:-1], hist, np.diff(edges)*.9, color=sns.color_palette()[1], edgecolor='w', alpha=1)

hist, _____ = np.histogram(ev_angles, bins=edges, density=True)
ax.bar(edges[:-1], hist, np.diff(edges)*.9, color=sns.color_palette()[0], edgecolor='w', alpha=.6)

lines = []
# for beta, cid in zip(draw_angles, color_id):
#     plt1 = ax.axvline(beta, color=sns.color_palette()[cid+1], linestyle='-', linewidth=4)
#     plt2 = ax.axvline(beta, color=sns.color_palette()[cid], linestyle='--', linewidth=4)
#     lines += [(plt1, plt2)]


# angle_description = [r"$\angle(\lambda^{id}_{net1};\lambda^{id}_{net2})$"
#                      .format(id=i+1, net1='{CPP}', net2='{HPP}')
#                      for i in range(3)]
# plt.legend(lines + [ev_ang, r_ang],
#            angle_description +
#            [r'$10^2$ Eigenvalue angles (CPP-HPP)',
#             r'$10^4$ Random angles (HPP-HPP)'],
#            loc='upper left', fontsize=fontsize)

ax.tick_params('y', labelsize=fontsize-4)
ax.tick_params('x', labelsize=fontsize)
ax.set_xticks(np.array([0, 0.125, .25, .375, .5, .625, .75, .875, 1])*pi)
ax.set_xticklabels(['', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$',
                    r'$\frac{3}{8}\pi$', r'$\frac{\pi}{2}$',
                    r'$\frac{5}{8}\pi$', r'$\frac{3}{4}\pi$',
                    r'$\frac{7}{8}\pi$', r'$\pi$'])
ax.set_xlabel(r'Plane Angle in $\mathtt{R}_+$' + r'$^{{}}$'.format(str(N)), fontsize=fontsize)
ax.set_ylabel('Angle Density', fontweight='bold', fontsize=fontsize)

sns.despine()

A1 = np.array([0,1,0,1])
A2 = np.array([0,1,0,0])
A1 = A1 / np.linalg.norm(A1)
A2 = A2 / np.linalg.norm(A2)

# A3 = np.array([0,0,1,0])

B1 = np.array([0,0,1,1])
B2 = np.array([0,1,0,1])
B1 = B1 / np.linalg.norm(B1)
B2 = B2 / np.linalg.norm(B2)

# B3 = np.array([0,0,1,0])

M = np.dot(np.array([A1,A2]), np.array([B1,B2]).T)
print M
space_angle = np.arccos(np.linalg.det(M))
print space_angle*180./np.pi

plt.show()


