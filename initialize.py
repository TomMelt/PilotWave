import constants as c
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm


def getx(y):
    """return x for I_0^x(pdf(x)) dx - y = 0.

    :y - float
    :returns: x - float

    """
    x = opt.brentq(
            lambda x: norm.cdf(x, loc=c.mu, scale=c.sigma) - y,
            c.xmin,
            c.xmax,
            )
    return x


def getInitialConditions(method="density"):
    """return an array of x positions based on normal distribution

    :returns: x and p - np.array of floats as a tuple

    """
    if method == "density":

        y = np.linspace(c.eps, 1.0-c.eps, c.N)
        x = [getx(i) for i in y]
        x = np.array(x)

    if method == "uniform":

        x = np.linspace(c.xmin, c.xmax, c.N)

    p = np.zeros(c.N)

    return x, p
