from scipy.integrate import cumtrapz
import constants as c
import numpy as np


def Hamiltonian(x, p):

    H = 0.5*c.m*p*p + 0.5*c.m*c.omega*c.omega*(x-c.xe)*(x-c.xe)

    return H


def density(x, method="density"):

    if method == "density":
        dx = np.gradient(x)
        rho = 1./(c.N*dx)

    elif method == "uniform":
        y = cumtrapz(c.weights, c._x, initial=0.)
        rho = np.gradient(y, x)

    rho = np.absolute(rho)

    return rho


def QuantumForce(x, method="density"):

    rho = density(x, c.xmethod)
    rhodot = np.gradient(rho, x)
    rhoddot = np.gradient(rhodot, x)

    if c.quantum:
        Q = -1./(4.*c.m)*(rhoddot/rho - 0.5*(rhodot/rho)*(rhodot/rho))
    else:
        Q = 0.*rho

    return Q


def dqdx_hardcode(x):

    ans = []

    for n in range(c.N):
        if n == 0:
            dm2 = dm1 = 0.
            dp1 = 1./(x[n+1] - x[n])
            dp2 = 1./(x[n+2] - x[n+1])
        elif n == 1:
            dm2 = 0.
            dm1 = 1./(x[n  ] - x[n-1])
            dp1 = 1./(x[n+1] - x[n  ])
            dp2 = 1./(x[n+2] - x[n+1])
        elif n == c.N - 2:
            dp2 = 0.
            dm2 = 1./(x[n-1] - x[n-2])
            dm1 = 1./(x[n  ] - x[n-1])
            dp1 = 1./(x[n+1] - x[n  ])
        elif n == c.N - 1:
            dp2 = dp1 = 0.
            dm2 = 1./(x[n-1] - x[n-2])
            dm1 = 1./(x[n  ] - x[n-1])
        else:
            dm2 = 1./(x[n-1] - x[n-2])
            dm1 = 1./(x[n  ] - x[n-1])
            dp1 = 1./(x[n+1] - x[n  ])
            dp2 = 1./(x[n+2] - x[n+1])

        tempp = dp1*dp1*(dp2-2.0*dp1+dm1)
        tempm = dm1*dm1*(dm2-2.0*dm1+dp1)
        ans.append(tempp - tempm)

    ans = np.array(ans)

    return 1./(4.*c.m)*ans


def numericDerivatives(x):

    dVdx = c.m*c.omega*c.omega*(x-c.xe)

    dQdx = np.gradient(QuantumForce(x, c.xmethod), x)

    pdot = -dVdx - dQdx

    return pdot


def equation_of_motion(t, coordinates):

    x = coordinates[:c.N]
    p = coordinates[c.N:]

    # first derivatives w.r.t. t
    xdot = p/c.m - c.gamma*(x-c.xe)

    pdot = numericDerivatives(x)

    return np.concatenate((xdot, pdot), axis=0)
