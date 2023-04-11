import numpy as np
import math as m
import matplotlib.pyplot as plt
import seaborn as sns

from math import gamma
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import laplace
from scipy.stats import poisson
from scipy.stats import uniform
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns

FOLDER_FOR_SAVE = "graphics/"

def normal_rvs_cdf(size, x):
    norm_rv = norm(loc = 0, scale = 1)
    sample = norm_rv.rvs(size = size)
    cdf = norm_rv.cdf(x)
    return [sample, cdf]

def poisson_rvs_cdf(size, x):
    poisson_rv = poisson(10)
    sample = poisson_rv.rvs(size = size)
    cdf = poisson_rv.cdf(x)
    return [sample, cdf]

def cauchy_rvs_cdf(size, x):
    cauchy_rv = cauchy(loc = 0, scale = 1)
    sample = cauchy_rv.rvs(size = size)
    cdf = cauchy_rv.cdf(x)
    return [sample, cdf]

def laplace_rvs_cdf(size, x):
    laplace_rv = laplace(loc = 0, scale = 1/m.sqrt(2))
    sample = laplace_rv.rvs(size = size)
    cdf = laplace_rv.cdf(x)
    return [sample, cdf]

def uniform_rvs_cdf(size, x):
    uniform_rv = uniform(loc = -m.sqrt(3), scale = 2 * m.sqrt(3))
    sample = uniform_rv.rvs(size = size)
    cdf = uniform_rv.cdf(x)
    return [sample, cdf]

def ECDF_continius(sizes : list, rvs_cdf, left, right, title):
    for size in sizes:
        x = np.linspace(left, right, size)
        sample, cdf = rvs_cdf(size, x)

        plt.plot(x, cdf, label='theoretical CDF')

        # для построения ECDF используем библиотеку statsmodels
        # from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(sample)
        plt.step(x, ecdf(x), label='ECDF')
        plt.title(title + " n = " + str(size))
        plt.ylabel('$f(x)$')
        plt.xlabel('$x$')
        plt.legend(loc = "lower right")
        plt.savefig(FOLDER_FOR_SAVE+"4__" +title +str(size) +".png")
        plt.clf()


def do_task4(sizes):
    ECDF_continius(sizes, normal_rvs_cdf, -4, 4, "Normal")
    ECDF_continius(sizes, poisson_rvs_cdf, 6, 14, "Poisson")
    ECDF_continius(sizes, cauchy_rvs_cdf, -4, 4, "Cauchy")
    ECDF_continius(sizes, laplace_rvs_cdf, -4, 4, "Laplace")
    ECDF_continius(sizes, uniform_rvs_cdf, -4, 4, "Uniform")

# sizes = [20, 60, 100]
# do_task4(sizes)


def normal_kde_pdf(size, x):
    pdf = norm.pdf(x, loc = 0, scale = 1)
    sample = norm.rvs(size = size, loc = 0, scale = 1)
    return [sample, pdf]

def poisson_kde_pdf(size, x):
    pdf = [10 ** float(y) * np.exp(-10) / gamma(y + 1) for y in x]
    # pdf = poisson.pmf(x, 10)  
    sample = poisson.rvs(10, size = size)
    return [sample, pdf]

def cauchy_kde_pdf(size, x):
    pdf = cauchy.pdf(x)
    sample = cauchy.rvs(size = size)
    return [sample, pdf]

def laplace_kde_pdf(size, x):
    pdf = laplace.pdf(x, loc = 0, scale = 1/m.sqrt(2))
    sample = laplace.rvs(size = size, loc = 0, scale = 1/m.sqrt(2))
    return [sample, pdf]

def uniform_kde_pdf(size, x):
    pdf = uniform.pdf(x, loc = -m.sqrt(3), scale = 2 * m.sqrt(3))
    sample = uniform.rvs(size = size, loc = -m.sqrt(3), scale = 2 * m.sqrt(3))
    return [sample, pdf]

def KDE(sizes : list, kde_pdf, left, right, title):
    h = [0.5, 1, 2]
    plt.subplots(3, 3, figsize = (10,10))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    z=1
    for k in range(len(sizes)):
        x = np.linspace(left, right, sizes[k])
        sample, pdf = kde_pdf(sizes[k], x)

        for i in range(len(h)):
            plt.subplot(3, 3, z)
            plt.plot(x, pdf, "r--", linewidth=2)
            sns.kdeplot(data=sample, bw_method='silverman', bw_adjust = h[i],
                            fill = True, common_norm=True,linewidth=1, label='kde')

            plt.title(title + " n = " + str(sizes[k]) + "; h = " + str(h[i]))
            plt.ylabel('$f(x)$')
            plt.xlabel('$x$')
            plt.legend(loc = "upper left")
            plt.xlim(left, right)
            z +=1
            
    # plt.savefig(FOLDER_FOR_SAVE+"4_2_" +title +".png")
    plt.show()

def do_task42(sizes):
    KDE(sizes, poisson_kde_pdf, 6, 14, "Poisson")
    KDE(sizes, normal_kde_pdf, -4, 4, "Normal")
    KDE(sizes, cauchy_kde_pdf, -4, 4, "Cauchy")
    KDE(sizes, laplace_kde_pdf, -4, 4, "Laplace")
    KDE(sizes, uniform_kde_pdf, -4, 4, "Uniform")

sizes = [20, 60, 100]
do_task42(sizes)
