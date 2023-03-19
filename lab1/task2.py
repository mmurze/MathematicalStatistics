import numpy as np
import math as m
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import laplace
from scipy.stats import poisson
from scipy.stats import uniform

FOLDER_FOR_SAVE = "graphics/"

def mean(x):
    return np.mean(x)

def median(x):
    return np.median(x)

def zR(x):
    return (x[0] + x[len(x) - 1]) / 2

def zP(x, i):
    return np.quantile(x, i)

def zQ(x):
    return (zP(x, 0.25) + zP(x, 0.75)) / 2

def zTR(x):
    r = int(len(x) / 4)
    sum = 0
    for i in range(r + 1, len(x) - r + 1):
        sum += x[i]
    return sum / (len(x) - 2 * r)

def dispersion(x):
    return np.std(x) ** 2

<<<<<<< HEAD
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
=======
>>>>>>> 017c78d8a43a6212dfb39e195c81b1478aed92c8

def print_table_rows(E, D, E_name, D_name):
    strTmp = E_name + " & " + str(E[0])
    for e in range(1, len(E)):
        strTmp += " & " + str(E[e])
    strTmp += " \\\\"
    print(strTmp)
    print("\\hline")

    strTmp = D_name + " & " + str(D[0])
    for d in range(1, len(D)):
        strTmp += " & " + str(D[d])
    strTmp += " \\\\"
    print(strTmp)
    print("\\hline")

    # strTmp = "E(z) \pm \sqrt{D(z)}"
    # for i in range(len(E)):
    #     strTmp += " & [" + str(round(E[i] - m.sqrt(D[i]), 6)) + ";"
    # strTmp += " \\\\"
    # print(strTmp)

    # strTmp = ""
    # for i in range(len(E)):
    #     strTmp += " & " + str(round(E[i] + m.sqrt(D[i]), 6)) + "]"
    # strTmp += " \\\\"
    # print(strTmp)
    # print("\\hline")
    return

<<<<<<< HEAD
def print_table(sizes : list, rvs_and_name):
    repeats = 1000
=======

def print_table_normal(sizes : list, repeats : int):
>>>>>>> 017c78d8a43a6212dfb39e195c81b1478aed92c8
    for size in sizes:
        means, meds, zRs, zQs, zTRs = [], [], [], [], []
        table = [means, meds, zRs, zQs, zTRs]
        E, D = [], []
        for i in range(repeats):
<<<<<<< HEAD
            distr = rvs_and_name(size)[0]
=======
            distr = norm.rvs(size = size, loc = 0, scale = 1)
            distr.sort()
>>>>>>> 017c78d8a43a6212dfb39e195c81b1478aed92c8
            means.append(mean(distr))
            meds.append(median(distr))
            zRs.append(zR(distr))
            zQs.append(zQ(distr))
            zTRs.append(zTR(distr))
        for column in table:
            E.append(np.around(mean(column), 6))
            D.append(np.around(dispersion(column), 6))
        print_table_rows(E, D, rvs_and_name(0)[1] + " E(z) " + str(size), rvs_and_name(0)[1] + " D(z) " + str(size))
    return

<<<<<<< HEAD
def do_task2(sizes):
    print_table(sizes, normal_rvs)
    print_table(sizes, poisson_rvs)
    print_table(sizes, cauchy_rvs)
    print_table(sizes, laplace_rvs)
    print_table(sizes, uniform_rvs)

sizes = [10, 100, 1000]
do_task2(sizes)
=======
def print_table_cauchy(sizes : list, repeats : int):
    for size in sizes:
        means, meds, zRs, zQs, zTRs = [], [], [], [], []
        table = [means, meds, zRs, zQs, zTRs]
        E, D = [], []
        for i in range(repeats):
            distr = cauchy.rvs(size = size, loc = 0, scale = 1)
            distr.sort()
            means.append(mean(distr))
            meds.append(median(distr))
            zRs.append(zR(distr))
            zQs.append(zQ(distr))
            zTRs.append(zTR(distr))
        for column in table:
            E.append(round(mean(column), 6))
            D.append(round(dispersion(column), 6))
        #print("size: " + str(size))
        print_table_rows(E, D, "cauchy E(z) " + str(size), "cauchy D(z) " + str(size))
    return

def print_table_poisson(sizes : list, repeats : int):
    for size in sizes:
        means, meds, zRs, zQs, zTRs = [], [], [], [], []
        table = [means, meds, zRs, zQs, zTRs]
        E, D = [], []
        for i in range(repeats):
            distr = poisson.rvs(10, size = size)
            distr.sort()
            means.append(mean(distr))
            meds.append(median(distr))
            zRs.append(zR(distr))
            zQs.append(zQ(distr))
            zTRs.append(zTR(distr))
        for column in table:
            E.append(round(mean(column), 6))
            D.append(round(dispersion(column), 6))
        #print("size: " + str(size))
        print_table_rows(E, D, "poisson E(z) " + str(size), "poisson D(z) " + str(size))
    return

def print_table_laplace(sizes : list, repeats : int):
    for size in sizes:
        means, meds, zRs, zQs, zTRs = [], [], [], [], []
        table = [means, meds, zRs, zQs, zTRs]
        E, D = [], []
        for i in range(repeats):
            distr = laplace.rvs(size = size, loc = 0, scale = 1/m.sqrt(2))
            distr.sort()
            means.append(mean(distr))
            meds.append(median(distr))
            zRs.append(zR(distr))
            zQs.append(zQ(distr))
            zTRs.append(zTR(distr))
        for column in table:
            E.append(round(mean(column), 6))
            D.append(round(dispersion(column), 6))
        #print("size: " + str(size))
        print_table_rows(E, D, "laplace E(z) " + str(size), "laplace D(z) " + str(size))
    return

def print_table_uniform(sizes : list, repeats : int):
    for size in sizes:
        means, meds, zRs, zQs, zTRs = [], [], [], [], []
        table = [means, meds, zRs, zQs, zTRs]
        E, D = [], []
        for i in range(repeats):
            distr = uniform.rvs(size = size, loc = -m.sqrt(3), scale = 2 * m.sqrt(3))
            distr.sort()
            means.append(mean(distr))
            meds.append(median(distr))
            zRs.append(zR(distr))
            zQs.append(zQ(distr))
            zTRs.append(zTR(distr))
        for column in table:
            E.append(round(mean(column), 6))
            D.append(round(dispersion(column), 6))
        #print("size: " + str(size))
        print_table_rows(E, D, "uniform E(z) " + str(size), "uniform D(z) " + str(size))
    return

def do_task2(sizes:list):
    print_table_uniform(sizes, 1000)
    print_table_laplace(sizes, 1000)
    print_table_normal(sizes, 1000)
    print_table_cauchy(sizes, 1000)
    print_table_poisson(sizes, 1000)


do_task2([10, 100, 1000])
>>>>>>> 017c78d8a43a6212dfb39e195c81b1478aed92c8
