import constants as c
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as itp
from numeric import density, QuantumForce


def plot3Danim(data, tstart=0., tfinal=None, tstep=10.):

    x = data[:, 1:c.N+1]
    p = data[:, c.N+1:2*c.N+1]
    H = data[:, 2*c.N+1:3*c.N+1]
    t = data[:, 0]
    if tfinal is None:
        tfinal = t[-1]
#    tn = np.arange(tstart, tfinal, step=tstep)
#    interpx1 = itp.interp1d(t, x)
#    interpx2 = itp.interp1d(t, p)
#    interpx3 = itp.interp1d(t, H)
#    x1 = interpx1(tn).T
#    x2 = interpx2(tn).T
#    x3 = interpx3(tn).T

    plt.ion()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    for i in range(len(t)):
        for ax in axes.flatten():
            ax.clear()
        axes[0, 0].plot(x[i], '-s')
        axes[1, 0].plot(p[i], '-s')
        axes[2, 0].plot(H[i], '-s')
        axes[0, 1].plot(density(x[i], c.xmethod), '-s')
        axes[1, 1].plot(QuantumForce(x[i], c.xmethod), '-s')
        axes[0, 0].set_ylabel('x')
        axes[1, 0].set_ylabel('p')
        axes[2, 0].set_ylabel('H')
        axes[0, 1].set_ylabel('rho')
        axes[0, 0].set_ylim(np.array([-5., 5.])+c.xe)
        axes[1, 0].set_ylim(np.array([-5., 5.]))
        axes[0, 1].set_ylim(np.array([0., 5.]))
        axes[0, 0].set_title('time = {0}'.format(t[i]))
        plt.pause(0.001)
    plt.ioff()

    return


def plot3Dtrace(data, particles=[1]):

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 16))

    for i in particles:

        if i < 1 or i > c.N:
            msg = "particle number must be between 1 and {0}.".format(c.N)
            raise ValueError(msg)

        index = i
        axes[0].plot(data[:, 0], data[:, index], '.')

        index = i+c.N
        axes[1].plot(data[:, 0], data[:, index], '.')

        index = i+2*c.N
        axes[2].plot(data[:, 0], data[:, index], '.')

    axes[0].set_ylabel('x')
    axes[1].set_ylabel('p')
    axes[2].set_ylabel('H')
    plt.show()

    return
