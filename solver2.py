import pandas as pd
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import argparse
from typing import Tuple, Union

# Helper functions
def L1(X: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    val = X.flatten()
    m = len(y)
    c = np.zeros(m + d + 1)
    c[3:] = 1
    A_ub = np.zeros((2 * m, m + d + 1))
    b_ub = np.zeros(2 * m)
    for i in range(d, -1, -1):
        A_ub[:m, d-i] = np.power(val, i)
        A_ub[m:, d-i] = -np.power(val, i)
    A_ub[:m, d+1:] = -np.identity(m)
    A_ub[m:, d+1:] = -np.identity(m)
    b_ub[:m] = y
    b_ub[m:] = -y
    bounds = [(None, None)] * 3 + [(0, None)] * m
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    return result.x[:d+1]

def L2(X: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    val = X.flatten()
    m = len(y)
    A_ub = np.zeros((m, d + 1))
    for i in range(d, -1, -1):
        A_ub[:, d-i] = np.power(val, i)
    result, _, _, _ = lstsq(A_ub, y)
    return result

def find_root(a: float, b: float, c: float) -> Union[Tuple[None, None], Tuple[float, Union[float, None]]]:
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None, None
    elif discriminant == 0:
        root = -b / (2*a)
        return root, None
    else:
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        return root1, root2

def calculateEmpiricalError(results: np.ndarray, t: float, y: np.ndarray) -> float:
    results = np.array(results)
    hypothesis_results = np.sign(results-t)
    return np.mean(hypothesis_results != y)

def find_optimal_t(X: np.ndarray, y: np.ndarray, a_1: float, b_1: float, c_1: float, a_2: float, b_2: float, c_2: float) -> Tuple[float, float]:
    p_1_pairs = [(a_1 * x*x + b_1*x + c_1, y_val) for x, y_val in zip(X, y)]
    p_2_pairs = [(a_2 * x*x + b_2*x + c_2, y_val) for x, y_val in zip(X, y)]
    p_1_pairs_sorted = sorted(p_1_pairs, key=lambda pair: pair[0])
    p_2_pairs_sorted = sorted(p_2_pairs, key=lambda pair: pair[0])
    p_1_results_sorted = [pair[0] for pair in p_1_pairs_sorted]
    y_sorted_1 = [pair[1] for pair in p_1_pairs_sorted]
    p_2_results_sorted = [pair[0] for pair in p_2_pairs_sorted]
    y_sorted_2 = [pair[1] for pair in p_2_pairs_sorted]

    t_1_opt = 0
    t_2_opt = 0
    curr_emp_err_1 = calculateEmpiricalError(p_1_results_sorted, t_1_opt, y_sorted_1)
    curr_emp_err_2 = calculateEmpiricalError(p_2_results_sorted, t_2_opt, y_sorted_2)

    for i in range(1, len(X)):
        left = max(-1.0, p_1_results_sorted[i-1])
        right = min(1.0, p_1_results_sorted[i])
        if left <= right:
            curr_t_1 = (right + left) / 2.0
            next_emp_err_1 = calculateEmpiricalError(p_1_results_sorted, curr_t_1, y_sorted_1)
            if next_emp_err_1 < curr_emp_err_1:
                t_1_opt = curr_t_1
                curr_emp_err_1 = next_emp_err_1

    for i in range(1, len(X)):
        left = max(-1.0, p_2_results_sorted[i-1])
        right = min(1.0, p_2_results_sorted[i])
        if left <= right:
            curr_t_2 = (right + left) / 2.0
            next_emp_err_2 = calculateEmpiricalError(p_2_results_sorted, curr_t_2, y_sorted_2)
            if next_emp_err_2 < curr_emp_err_2:
                t_2_opt = curr_t_2
                curr_emp_err_2 = next_emp_err_2
    return t_1_opt, t_2_opt

def main(datafile: str) -> None:
    data = pd.read_csv(datafile, header=None).to_numpy()
    points, sigma, r = data[0, 0].astype('int64'), data[0, 1], data[0, 2]
    y = data[1:, 0]
    X = data[1:, 1]

    a_1, b_1, c_1 = L1(X, y, 2)
    a_2, b_2, c_2 = L2(X, y, 2)
    print(f"L1 coefficient: {a_1}, {b_1}, {c_1}")
    print(f"L2 coefficient: {a_2}, {b_2}, {c_2}")

    L1_root_1, L1_root_2 = find_root(a_1, b_1, c_1)
    L2_root_1, L2_root_2 = find_root(a_2, b_2, c_2)

    t_1, t_2 = find_optimal_t(X, y, a_1, b_1, c_1, a_2, b_2, c_2)
    print(f"t1={t_1}, t2={t_2}")

    L1_root_1, L1_root_2 = find_root(a_1, b_1, c_1-t_1)
    L2_root_1, L2_root_2 = find_root(a_2, b_2, c_2-t_2)
    print(f"L1 roots: {L1_root_1}, {L1_root_2}")
    print(f"L2 roots: {L2_root_1}, {L2_root_2}")

    err_1 = norm.cdf(-1.0364333894937898) - norm.cdf(L1_root_2) + norm.cdf(0.6744897501960817)-0.5 + 1.0-norm.cdf(L1_root_1)
    err_2 = norm.cdf(-1.0364333894937898) - norm.cdf(L2_root_2) + norm.cdf(0.6744897501960817)-0.5 + 1.0-norm.cdf(L2_root_1)
    print(f"err1={err_1}, err2={err_2}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process polynomial regression.')
    parser.add_argument('datafile', type=str, help='Path to the CSV data file.')
    args = parser.parse_args()
    main(args.datafile)
