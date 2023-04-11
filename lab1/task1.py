import numpy as np
import matplotlib.pyplot as plt
import math as m

from scipy.stats import norm
from scipy.stats import cauchy
from scipy.stats import laplace
from scipy.stats import poisson
from scipy.stats import uniform

FOLDER_FOR_SAVE = "graphics/"

def plot_poisson(sizes:list, x_name : str, y_name : str):
    for size in sizes:
        fig, ax = plt.subplots(1,1)

        hist = poisson.rvs(10, size = size)
        n = poisson(10)                                                 
        
        x = np.arange(poisson.ppf(0.01, 10), poisson.ppf(0.99, 10))     
        y = n.pmf(x)                                                    
        
        ax.plot(x, y, 'k', lw  = 2)
        ax.hist(hist, density=True, bins='auto', histtype='stepfilled', alpha=0.5)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title("size: " + str(size))

        # plt.savefig(FOLDER_FOR_SAVE + "poisson" + str(size) + ".png")
        plt.show()

def plot_norm(sizes:list, x_name : str, y_name : str):
    for size in sizes:
        fig, ax = plt.subplots(1,1)
        
        n = norm(loc = 0, scale = 1)                       
       
        x = np.linspace(n.ppf(0.01), n.ppf(0.99), 100)     
        y = n.pdf(x)                                       
        
        ax.plot(x, y, 'k', lw  = 2)
        ax.hist(n.rvs(size = size), density=True, bins='auto', histtype='stepfilled', alpha=0.5)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title("size: " + str(size))

        plt.savefig(FOLDER_FOR_SAVE + "norm" +str(size)+".png")

def plot_cauchy(sizes:list, x_name : str, y_name : str):
    for size in sizes:
        fig, ax = plt.subplots(1,1)
        
        n = cauchy(loc = 0, scale = 1)                     
       
        x = np.linspace(n.ppf(0.01), n.ppf(0.99), 100)     
        y = n.pdf(x)                                       
        
        ax.plot(x, y, 'k', lw  = 2)
        ax.hist(n.rvs(size = size), density=True, histtype='stepfilled', alpha=0.5)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title("size: " + str(size))

        plt.savefig(FOLDER_FOR_SAVE + "cauchy" +str(size)+".png")

def plot_laplace(sizes:list, x_name : str, y_name : str):
    for size in sizes:
        fig, ax = plt.subplots(1,1)
        
        n = laplace(loc = 0, scale = 1/m.sqrt(2))                                       
       
        x = np.linspace(n.ppf(0.01), n.ppf(0.99), 100)     
        y = n.pdf(x)                                       
        
        ax.plot(x, y, 'k', lw  = 2)
        ax.hist(n.rvs(size = size), density=True, histtype='stepfilled', alpha=0.5)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title("size: " + str(size))

        plt.savefig(FOLDER_FOR_SAVE + "laplace" +str(size)+".png")

def plot_uniform(sizes:list, x_name : str, y_name : str):
    for size in sizes:
        fig, ax = plt.subplots(1,1)
        
        n = uniform(loc = -m.sqrt(3), scale = 2 * m.sqrt(3))                                       
       
        x = np.linspace(n.ppf(0.01), n.ppf(0.99), 100)     
        y = n.pdf(x)                                       
        
        ax.plot(x, y, 'k', lw  = 2)
        ax.hist(n.rvs(size = size), density=True, histtype='stepfilled', alpha=0.5)

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title("size: " + str(size))

        plt.savefig(FOLDER_FOR_SAVE + "uniform" +str(size)+".png")

def do_task1(sizes:list):
    plot_uniform(sizes, "uniformNumbers", "Density")
    plot_laplace(sizes, "laplaceNumbers", "Density")
    plot_norm(sizes, "normalNumbers", "Density")
    plot_cauchy(sizes, "cauchyNumbers", "Density")
    plot_poisson(sizes, "poissonNumbers", "Density")

do_task1([10, 50, 1000])
