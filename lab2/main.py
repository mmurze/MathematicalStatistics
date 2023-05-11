import os
import numpy as np
import scipy
import scipy.stats as st
import scipy.optimize as opt
import matplotlib.pyplot as plt
from typing import *


def generate_normal_points(correlation_coeff: float, size: int) -> np.ndarray:
    return st.multivariate_normal(mean=[0, 0], cov=[[1, correlation_coeff], [correlation_coeff, 1]]).rvs(size)


def corrcoef_pearson(data: np.ndarray) -> float:
    r, p = st.pearsonr(data.T[0], data.T[1])
    return r


def corrcoef_spearman(data: np.ndarray) -> float:
    r, p = st.spearmanr(data.T[0], data.T[1])
    return r


def corrcoef_quadrant(data: np.ndarray) -> float:
    n = len(data)
    n1 = 0; n2 = 0; n3 = 0; n4 = 0

    x_mean = np.median(data.T[0])
    y_mean = np.median(data.T[1])
    for elem in data:
        if elem[0] > x_mean and elem[1] > y_mean:
            n1 += 1
        if elem[0] < x_mean and elem[1] > y_mean:
            n2 += 1
        if elem[0] < x_mean and elem[1] < y_mean:
            n3 += 1
        if elem[0] > x_mean and elem[1] < y_mean:
            n4 += 1

    return ((n1 + n3) - (n2 + n4)) / n


def draw_data(data: np.ndarray) -> None:
    plt.scatter(data.T[0], data.T[1])
    plt.show()


def write_corrcoefs(distribution: Callable[[float, int], np.ndarray], name: str, size: int,
                    rhos: List[float], repeats=1000, folder=".", ext="tex"):
    filename = f"{folder}/corrcoefs_{name}_{size}.{ext}"

    if not os.path.isdir(folder):
        os.makedirs(folder)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\\begin{tabular}{| c | c | c | c |} \\hline\n")
        for rho in rhos:
            file.write(f"$\\rho = {rho}$ & $r$ & $r_S$ & $r_Q$ \\\\ \\hline \n")
            pearson = np.array([])
            spearman = np.array([])
            quadrant = np.array([])
            for _ in range(repeats):
                data = distribution(rho, size)
                pearson = np.append(pearson, corrcoef_pearson(data))
                spearman = np.append(spearman, corrcoef_spearman(data))
                quadrant = np.append(quadrant, corrcoef_quadrant(data))

            p_mean = np.round(np.mean(pearson), 4)
            s_mean = np.round(np.mean(spearman), 4)
            q_mean = np.round(np.mean(quadrant), 4)
            p_mean2 = np.round(np.mean([x ** 2 for x in pearson]), 4)
            s_mean2 = np.round(np.mean([x ** 2 for x in spearman]), 4)
            q_mean2 = np.round(np.mean([x ** 2 for x in quadrant]), 4)
            p_var = np.round(np.var(pearson), 4)
            s_var = np.round(np.var(spearman), 4)
            q_var = np.round(np.var(quadrant), 4)

            file.write(f"$E(z)$ & {p_mean} & {s_mean} & {q_mean} \\\\ \n")
            file.write(f"$E(z^2)$ & {p_mean2} & {s_mean2} & {q_mean2} \\\\ \n")
            file.write(f"$D(z)$ & {p_var} & {s_var} & {q_var} \\\\ \\hline \n")
        file.write("\\end{tabular}")


def write_corrcoefs_for_mix(name: str, sizes: List[int],
                            repeats=1000, folder=".", ext="tex"):
    def distribution(size):
        a = st.multivariate_normal(mean=[0, 0], cov=[[1, 0.9], [0.9, 1]]).rvs(size)
        b = st.multivariate_normal(mean=[0, 0], cov=[[10, -9], [-9, 10]]).rvs(size)
        return 0.9 * a + 0.1 * b

    filename = f"{folder}/corrcoefs_{name}.{ext}"

    if not os.path.isdir(folder):
        os.makedirs(folder)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\\begin{tabular}{| c | c | c | c |} \\hline\n")
        for size in sizes:
            file.write(f"size = {size} & $r$ & $r_S$ & $r_Q$ \\\\ \\hline \n")
            pearson = np.array([])
            spearman = np.array([])
            quadrant = np.array([])
            for _ in range(repeats):
                data = distribution(size)
                pearson = np.append(pearson, corrcoef_pearson(data))
                spearman = np.append(spearman, corrcoef_spearman(data))
                quadrant = np.append(quadrant, corrcoef_quadrant(data))

            p_mean = np.round(np.mean(pearson), 4)
            s_mean = np.round(np.mean(spearman), 4)
            q_mean = np.round(np.mean(quadrant), 4)
            p_mean2 = np.round(np.mean([x ** 2 for x in pearson]), 4)
            s_mean2 = np.round(np.mean([x ** 2 for x in spearman]), 4)
            q_mean2 = np.round(np.mean([x ** 2 for x in quadrant]), 4)
            p_var = np.round(np.var(pearson), 4)
            s_var = np.round(np.var(spearman), 4)
            q_var = np.round(np.var(quadrant), 4)

            file.write(f"$E(z)$ & {p_mean} & {s_mean} & {q_mean} \\\\ \n")
            file.write(f"$E(z^2)$ & {p_mean2} & {s_mean2} & {q_mean2} \\\\ \n")
            file.write(f"$D(z)$ & {p_var} & {s_var} & {q_var} \\\\ \\hline \n")
        file.write("\\end{tabular}")


def draw_ellipse(distribution: Callable[[float, int], np.ndarray], size: int,
                    rho: float, folder=".", ext="pdf"):
    plt.clf()
    data = distribution(rho, size)
    x = np.linspace(np.min(data.T[0]) - 1, np.max(data.T[0]) + 1, 1000)
    y = np.linspace(np.min(data.T[1]) - 1, np.max(data.T[1]) + 1, 1000)
    x, y = np.meshgrid(x, y)
    corrcoef = corrcoef_pearson(data)
    x_mean = np.mean(data.T[0])
    y_mean = np.mean(data.T[1])
    sigma_x2 = np.var(data.T[0])
    sigma_y2 = np.var(data.T[0])
    plt.contour(x, y, ((x - x_mean) ** 2 / sigma_x2 -
                       2 * corrcoef * (x - x_mean) * (y - y_mean) / (np.sqrt(sigma_x2) * np.sqrt(sigma_y2)) +
                       (y - y_mean) ** 2 / sigma_y2), [1])
    plt.scatter(data.T[0], data.T[1])
    plt.title(f"n = {size}, $\\rho$ = {rho}")

    if not os.path.isdir(folder):
        os.makedirs(folder)

    plt.savefig(f"{folder}/ellipse_normal_{rho}_{size}.{ext}")


def draw_ellipses(distribution: Callable[[float, int], np.ndarray], sizes: List[int],
                    rhos: List[float], folder=".", ext="pdf"):
    for rho in rhos:
        for size in sizes:
            draw_ellipse(distribution, size, rho, folder, ext)


def generate_regression_data() -> np.ndarray:
    errors = st.norm.rvs(loc=0, scale=1, size=20)
    x = np.linspace(-1.8, 2.0, 20)
    return np.array([[x, 2 + 2 * x + error] for (x, error) in zip(x, errors)])


def ls_linear_regression(data):
    res = st.linregress(data.T)
    return res.intercept, res.slope


def lad_linear_regression(data):
    def criteria(b: List[float]) -> float:
        return np.sum([np.abs(y - b[0] - b[1] * x) for (x, y) in data])
    res = opt.minimize(criteria, [0, 1])
    return res.x[0], res.x[1]


def regression(make_fliers: bool = False):
    data = generate_regression_data()
    if make_fliers:
        data.T[1][0] += 10
        data.T[1][len(data) - 1] -= 10
    b0_ls, b1_ls = ls_linear_regression(data)
    b0_lad, b1_lad = lad_linear_regression(data)
    x = np.linspace(np.min(data.T[0]), np.max(data.T[0]), 100)

    title = "Линейная регрессия"
    if make_fliers:
        title += " с возмущениями"
        
    plt.clf()
    plt.title(title)
    plt.scatter(data.T[0], data.T[1], color="k")
    plt.plot(x, 2 * x + 2)
    plt.plot(x, b0_ls + b1_ls * x)
    plt.plot(x, b0_lad + b1_lad * x)
    plt.legend(["Выборка", "Модель", f"МНК, $\\beta*^ \\approx {np.round(b0_ls, 2)}$, $\\beta_1 \\approx {np.round(b1_ls, 2)}$",
                f"МНМ, $\\beta_0 \\approx {np.round(b0_lad, 2)}$, $\\beta_1 \\approx {np.round(b1_lad, 2)}$"])
               #loc="upper left")
    plt.grid()
    plt.show()


def draw_regression(folder=".", ext="pdf"):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    regression()
    plt.savefig(f"{folder}/regression.{ext}")
    regression(make_fliers=True)
    plt.savefig(f"{folder}/regression_fliers.{ext}")

def chi_2_experiment(distribution: Callable[[int], np.ndarray], name, size, folder=".", ext="tex"):
    n = size
    data = distribution(size)
    k = np.round(1.72 * (n ** (1 / 3))).astype(int)
    points = np.linspace(-1.1, 1.1, k - 1)

    ni = np.array([])
    n_sum = 0
    for point in points:
        ni_cur = 0
        for x in data:
            if x < point:
                ni_cur += 1
        ni = np.append(ni, ni_cur - n_sum)
        n_sum = ni_cur
    ni = np.append(ni, n - n_sum).astype(int)

    pi = np.array([])
    p_sum = 0
    mean = np.mean(data)
    std = np.std(data)
    for point in points:
        p_cur = st.norm.cdf(point, loc=mean, scale=std)
        pi = np.append(pi, p_cur - p_sum)
        p_sum = p_cur
    pi = np.append(pi, 1 - p_sum)

    # calculations
    npi = n * pi
    ni_minus_npi = ni - npi
    res = (ni_minus_npi ** 2) / npi

    filename = f"{folder}/chi_{name}_{size}.{ext}"

    if not os.path.isdir(folder):
        os.makedirs(folder)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\\begin{tabular}{| c | c | c | c | c | c | c |} \\hline\n")
        file.write("$i$ & limits & $n_i$ & $p_i$ & $np_i$ & $n_i - np_i$ & $\\frac{(n_i-np_i)^2}{np_i} $ \\\\ \\hline \n")
        res_sum = np.sum(res)
        for (i, ni, pi, npi, ni_minus_npi, res) in zip(range(1, k + 1), ni, pi, npi, ni_minus_npi, res):
            a = np.round(points[i - 2], 2) if i - 1 > 0 else 'inf'
            b = np.round(points[i - 1], 2) if i - 1 < len(points) else 'inf'
            file.write(f"{i} & [{a}, {b}] & {ni} & {np.round(pi, 4)} & {np.round(npi, 2)} & {np.round(ni_minus_npi, 2)} & {np.round(res, 2)} \\\\ \\hline \n")
        file.write(f"$\\Sigma$ & - & {n} & 1 & {n} & 0 & {np.round(res_sum, 2)} \\\\ \\hline \n")
        file.write("\\end{tabular}")


def student_interval_for_mean(data: np.ndarray, alpha: float = 0.05):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    t = st.t.ppf(1 - 0.5 * alpha, df=n - 1)
    a = mean - std * t / np.sqrt(n - 1)
    b = mean + std * t / np.sqrt(n - 1)
    return a, b


def chi2_interval_for_std(data: np.ndarray, alpha: float = 0.05):
    n = len(data)
    std = np.std(data)
    chi_a = st.chi2.ppf(1 - 0.5 * alpha, df=n - 1)
    chi_b = st.chi2.ppf(0.5 * alpha, df=n - 1)
    a = std * np.sqrt(n) / np.sqrt(chi_a)
    b = std * np.sqrt(n) / np.sqrt(chi_b)
    return a, b


def asymp_interval_for_mean(data: np.ndarray, alpha: float = 0.05):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    u = st.norm.ppf(1 - 0.5 * alpha)
    a = mean - std * u / np.sqrt(n)
    b = mean + std * u / np.sqrt(n)
    return a, b


def asymp_interval_for_std(data: np.ndarray, alpha: float = 0.05):
    n = len(data)
    std = np.std(data)
    u = st.norm.ppf(1 - 0.5 * alpha)
    e = st.kurtosis(data)
    a = std / np.sqrt((1 + u * np.sqrt((e + 2) / n)))
    b = std / np.sqrt((1 - u * np.sqrt((e + 2) / n)))
    return a, b


def draw_hist(data: np.ndarray, folder: str = ".", ext: str = "pdf"):
    size = len(data)
    q25, q75 = np.percentile(data, [25, 75])
    bin_width = 2 * (q75 - q25) * len(data) ** (-1 / 3)
    bins = round((data.max() - data.min()) / bin_width)

    x = np.linspace(data.min(), data.max(), 1000)

    plt.clf()
    plt.hist(data, density=1, bins=bins, label="Generated data", edgecolor='blue', alpha = 0.3, linewidth=1.0)
    plt.title(f"$n = {size}$")
    plt.xlabel(f"Normal numbers")
    plt.legend()

    if not os.path.isdir(folder):
        os.makedirs(folder)
    plt.savefig(f"{folder}/hist_{size}.{ext}")


def draw_intervals(intervals: List[Tuple[float, float]], legend: List[str], title: str, folder: str = ".", ext: str = "pdf"):
    count = len(intervals)

    plt.clf()
    for (interval, label, k) in zip(intervals, legend, range(1, count + 1)):
        plt.plot(interval, [k, k], label=label, marker="o")
    plt.title(title)
    plt.ylim(0, count + 1)
    plt.legend()

    if not os.path.isdir(folder):
        os.makedirs(folder)

    plt.savefig(f"{folder}/{title.lower()}_intervals.{ext}")


def make_intervals(sizes: List[int]):
    mean_int = []
    std_int = []
    asymp_mean_int = []
    asymp_std_int = []
    for size in sizes:
        data = st.norm.rvs(loc=0, scale=1, size=size)
        draw_hist(data, folder="./figures/hists")
        mean_int.append(student_interval_for_mean(data))
        std_int.append(chi2_interval_for_std(data))
        asymp_mean_int.append(asymp_interval_for_mean(data))
        asymp_std_int.append(asymp_interval_for_std(data))

    draw_intervals(mean_int, [f"$n = {size}$" for size in sizes], "Mean", folder="./figures/intervals")
    draw_intervals(std_int, [f"$n = {size}$" for size in sizes], "Std", folder="./figures/intervals")
    draw_intervals(asymp_mean_int, [f"$n = {size}$" for size in sizes], "Mean", folder="./figures/asymp_intervals")
    draw_intervals(asymp_std_int, [f"$n = {size}$" for size in sizes], "Std", folder="./figures/asymp_intervals")


np.random.seed(737146741)
write_corrcoefs(generate_normal_points, "normal", 20, [0, 0.5, 0.9], folder="./figures/corrcoefs")
write_corrcoefs(generate_normal_points, "normal", 60, [0, 0.5, 0.9], folder="./figures/corrcoefs")
write_corrcoefs(generate_normal_points, "normal", 100, [0, 0.5, 0.9], folder="./figures/corrcoefs")
write_corrcoefs_for_mix("mix", [20, 60, 100], folder="./figures/corrcoefs")
draw_ellipses(generate_normal_points, [20, 60, 100], [0, 0.5, 0.9], folder="./figures/ellipses")
draw_regression(folder="./figures/regression")

data = st.norm.rvs(loc=0, scale=1, size=100)
print(f"mean {np.mean(data)}, std {np.std(data)}")

chi_2_experiment(lambda size: st.norm.rvs(loc=0, scale=1, size=size), "normal", 100, folder="./figures/chi")
chi_2_experiment(lambda size: st.laplace.rvs(loc=0, scale=1, size=size), "laplace", 20, folder="./figures/chi")
chi_2_experiment(lambda size: st.uniform.rvs(loc=-np.sqrt(3), scale=2 * np.sqrt(3), size=size), "uniform", 20, folder="./figures/chi")

make_intervals([20, 100])