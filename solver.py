import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import argparse
from typing import Tuple

import re #for pattern matching

def extract_variables_from_filename(filename: str) -> Tuple[str, float, float, float]:
    """
    Extracts variables encoded in a filename.

    The function parses a given filename to extract variables based on predefined patterns.
    This is used to obtain `flip_type`, `left`, `right`, and `opt` values from filenames.

    Args:
        filename (str): The path to the file with the encoded variables in its name.

    Returns:
        Tuple[str, float, float, float]: A tuple containing:
            - flip_type (str): The type of flip, either 'asymmetric' or another type.
            - left (float): The left value, significant when flip_type is 'asymmetric'. Otherwise, 0.0.
            - right (float): The right value, significant when flip_type is 'asymmetric'. Otherwise, 0.0.
            - opt (float): The opt value, significant when flip_type is not 'asymmetric'. Otherwise, 0.0.

    Raises:
        ValueError: If the filename does not match any of the expected patterns.
    """

    # Extract the basename without directory structure
    basename = filename.split('/')[-1]

    # Check and extract the values
    if "asymmetric" in basename:
        # For the 'asymmetric' case containing 'left' and 'right'
        pattern = r"data_asymmetric_(\d+(\.\d+)?_\d+(\.\d+)?).csv"
        match = re.match(pattern, basename)
        if match:
            flip_type = "asymmetric"
            left, right = map(float, match.group(1).split('_'))
            return flip_type, left, right, 0.0  # 'opt' is 0.0, as it's irrelevant
    else:
        # For other cases containing 'opt'
        pattern = r"data_(\w+)_(\d+(\.\d+)?).csv"
        match = re.match(pattern, basename)
        if match:
            flip_type = match.group(1)
            opt = float(match.group(2))
            return flip_type, 0.0, 0.0, opt  # 'left' and 'right' are 0.0, as they're irrelevant

    raise ValueError("Filename does not match the expected pattern")

def L1_polynomial_regression(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    val = X.flatten()
    m = len(y)
    c = np.zeros(m + 2)
    c[2:] = 1
    A_ub = np.zeros((2 * m, m + 2))
    b_ub = np.zeros(2 * m)
    for i in range(m):
        A_ub[i, 0] = val[i]
        A_ub[m+i, 0] = -val[i]
        A_ub[i, 1] = 1
        A_ub[m+i, 1] = -1
        A_ub[i, i+2] = -1
        A_ub[m+i, i+2] = -1
        b_ub[i] = y[i]
        b_ub[m+i] = -y[i]
    bounds = [(None, None), (None, None)] + [(0, None)] * m
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    a_1, b_1 = result.x[0], result.x[1]
    return a_1, b_1

def L2_polynomial_regression(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    X = X[:, np.newaxis]
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((X, ones_column))
    XtX = np.matmul(X.T, X)
    Xty = np.matmul(X.T, y)
    w = np.linalg.solve(XtX, Xty)
    a_2 = w[0]
    b_2 = w[1]
    return a_2, b_2

def main(datafile: str) -> None:
    data = pd.read_csv(datafile, header=None)
    data = data.to_numpy()
    points, sigma, r = data[0, 0].astype('int64'), data[0, 1], data[0, 2]
    y = data[1:, 0]
    X = data[1:, 1]

    a_1, b_1 = L1_polynomial_regression(X, y)
    a_2, b_2 = L2_polynomial_regression(X, y)
    print("L1 coefficient:", a_1, b_1)
    print("L2 coefficient:", a_2, b_2)

    x = np.arange(-2.0, 2.0, 0.01)
    p = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
    y_1 = a_1 * x + b_1
    y_2 = a_2 * x + b_2

    plt.figure(figsize=(8, 6))
    plt.plot(x, p, color="blue", label="Normal distribution N(0.0,"+str(sigma) +")")
    plt.plot(x, np.sign(a_1 * x + b_1), color="red", label="h1")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('L1 and L2 Regression Lines with Training Data')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform L1 and L2 regression.')
    parser.add_argument('datafile', type=str, help='The path to the data CSV file.')
    args = parser.parse_args()
    main(args.datafile)
