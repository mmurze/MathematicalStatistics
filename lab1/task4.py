import numpy as np
import math as m
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import laplace
from scipy.stats import poisson
from scipy.stats import uniform
from statsmodels.distributions.empirical_distribution import ECDF


def normal_rvs_cdf(size, start, end):
    norm_rv = norm(loc = 0, scale = 1)
    distr = norm_rv.rvs(size = size)
    distr.sort()
    x = np.linspace(start, end, size)
    cdf = norm_rv.cdf(x)
    return [distr, cdf]

def poisson_rvs_cdf(size, start, end):
    poisson_rv = poisson(10)
    distr = poisson_rv.rvs(size = size)
    distr.sort()
    x = np.linspace(start, end, size)
    cdf = poisson_rv.cdf(x, mu = 10)
    return [distr, cdf]

def cauchy_rvs_cdf(size, start, end):
    distr = cauchy.rvs(size=size, loc = 0, scale = 1)
    distr.sort()
    return [distr, "Cauchy"]

def laplace_rvs_cdf(size, start, end):
    distr = laplace.rvs(size=size, loc = 0, scale = 1/m.sqrt(2))
    distr.sort()
    return [distr, "Laplace"]

def uniform_rvs_cdf(size, start, end):
    distr = uniform.rvs(size=size, loc = -m.sqrt(3), scale = 2 * m.sqrt(3))
    distr.sort()
    return [distr, "Uniform"]

def ECDF_continius(sizes : list, rvs):
    norm_rv = norm(loc = 0, scale = 1)
    sample = norm_rv.rvs(100)
    x = np.linspace(-4,4,100)
    cdf = norm_rv.cdf(x)
    plt.plot(x, cdf, label='theoretical CDF')

    # для построения ECDF используем библиотеку statsmodels
    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf = ECDF(sample)
    plt.step(ecdf.x, ecdf.y, label='ECDF')

    plt.ylabel('$f(x)$')
    plt.xlabel('$x$')
    plt.legend(loc='upper left')