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

    strTmp = "E(z) \pm \sqrt{D(z)}"
    for i in range(len(E)):
        strTmp += " & [" + str(round(E[i] - m.sqrt(D[i]), 6)) + ";"
    strTmp += " \\\\"
    print(strTmp)

    strTmp = ""
    for i in range(len(E)):
        strTmp += " & " + str(round(E[i] + m.sqrt(D[i]), 6)) + "]"
    strTmp += " \\\\"
    print(strTmp)
    print("\\hline")
    return

def print_table_normal(sizes : list, repeats : int):
    for size in sizes:
        means, meds, zRs, zQs, zTRs = [], [], [], [], []
        table = [means, meds, zRs, zQs, zTRs]
        E, D = [], []
        for i in range(repeats):
            distr = norm.rvs(size = size)
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
        print_table_rows(E, D, "Normal E(z) " + str(size), "Normal D(z) " + str(size))
    return
print_table_normal([10, 50], 1000)