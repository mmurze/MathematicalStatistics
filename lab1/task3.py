import numpy as np
import math as m
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import laplace
from scipy.stats import poisson
from scipy.stats import uniform

def emissions(x, size):
    count = 0
    q1, q3 = np.quantile(x, [0.25, 0.75])
    min = q1 - 3/2*(q3-q1)
    max = q1 + 3/2*(q3-q1)
    for i in range(size):
        if x[i] < min or x[i] > max:
            count += 1
    count /= size
    return count

def normal_rvs(size):
    distr = norm.rvs(size = size, loc = 0, scale = 1)
    distr.sort()
    return [distr, "Normal"]

def poisson_rvs(size):
    distr = poisson.rvs(10, size = size)
    distr.sort()
    return [distr, "Poisson"]

def cauchy_rvs(size):
    distr = cauchy.rvs(size=size, loc = 0, scale = 1)
    distr.sort()
    return [distr, "Cauchy"]

def laplace_rvs(size):
    distr = laplace.rvs(size=size, loc = 0, scale = 1/m.sqrt(2))
    distr.sort()
    return [distr, "Laplace"]

def uniform_rvs(size):
    distr = uniform.rvs(size=size, loc = -m.sqrt(3), scale = 2 * m.sqrt(3))
    distr.sort()
    return [distr, "Uniform"]

def plot_boxplot_Tukey(sizes : list, rvs_and_name):
    repeats = 1000
    result, count =  [], 0
    for size in sizes:
        result.append(rvs_and_name(size)[0])
        for i in range(repeats):
            distr = rvs_and_name(size)[0]
            count = emissions(distr, size)
        count /= (repeats*size)
        print(rvs_and_name(0)[1] + " " + str(size) + " -> " + str(np.around(count,  decimals = 6)) + "\n")
    
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    ax.boxplot(result, vert = 0)
    ax.set_yticklabels(sizes)
    plt.title(rvs_and_name(0)[1])
    plt.show()

def do_task3(sizes):
    plot_boxplot_Tukey(sizes, normal_rvs)
    plot_boxplot_Tukey(sizes, poisson_rvs)
    plot_boxplot_Tukey(sizes, cauchy_rvs)
    plot_boxplot_Tukey(sizes, laplace_rvs)
    plot_boxplot_Tukey(sizes, uniform_rvs)

sizes = [20, 100]
do_task3(sizes)
