from scipy.stats import norm
import numpy as np

# mass
m = 1.0
omega = 1.0
gamma = 2.0

# no of particles
N = 50

# quantum force
quantum = True

# RK4 parameters
rtol = 1e-07
atol = 1e-08
maxstep = 1.
ts = 0.
tf = 15.0

# numerical differentiation
dtol = 1e-08
dmethod = "stencil"

# equilibrium
xe = 8.

# initial params for gaussian
eps = 1e-5
sigma = 0.5
mu = xe
xmin = mu - 5.0*sigma
xmax = mu + 5.0*sigma
xmethod = "uniform"

if xmethod == "uniform":
    # if uniform spread along x-axis then assign weight according to normal
    _x = np.linspace(xmin, xmax, N)
    weights = norm.pdf(_x, loc=mu, scale=sigma)
elif xmethod == "density":
    # if uniform spread in density then each has equal weight
    weights = 1./N*np.ones(N)
