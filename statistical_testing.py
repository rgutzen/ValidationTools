import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def plot_comparison(dist1,dist2):
    binnum = len(dist1)
    bins = np.linspace(-binnum/2., binnum/2., binnum+1)
    print bins
    fig = plt.figure('Distribution Comparison')
    plt.plot(bins[:-1], dist1, color='g', lw=2, label='dist1')
    plt.plot(bins[:-1], dist2, color='y', lw=2, label='dist2')
    D_KS, p_value = st.ks_2samp(dist1, dist2)
    plt.draw()
    # do sth.
    return None

def ks_test(X, Y):
    print st.ks_2samp(X, Y, xlabel='Measured Parameter')
    X = np.array(X)[np.isfinite(X)]
    Y = np.array(Y)[np.isfinite(Y)]
    print st.ks_2samp(X, Y)
    # ToDo: Is this measure more meaningful: with or without including nans?
    fig = plt.figure('KS-Test')
    plt.ylabel('CDF')
    plt.xlabel(xlabel)
    ax = np.empty((2),dtype=object)
    ax[0] = fig.add_subplot(111)
    ax[1] = ax[0].twiny()
    color = ['g', 'r']
    for i, A in enumerate([X, Y]):
        A_sorted = np.sort(A)
        A_sorted = A_sorted[np.isfinite(A_sorted)]
        CDF = np.array(range(len(A))) / float(len(A))
        ax[i].plot(A_sorted, CDF, color=color[i])
        ax[i].set_xticks(A_sorted)
        ax[i].set_xticklabels([''] * len(A_sorted))
        ax[i].tick_params(axis='x', length=15, color=color[i])
        plt.draw()
    return st.ks_2samp(X, Y)

