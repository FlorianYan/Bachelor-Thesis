import argparse
import csv
import os
import numpy as np
import scipy.stats as stats

def find_x_for_probability(probability: float) -> float:
    """
    Computes the x-value for a given probability in a standard normal distribution.

    Parameters:
    probability (float): The probability level for which to find the corresponding x-value.

    Returns:
    float: The x-value corresponding to the given probability level.
    """

    x = stats.norm.ppf(1 - probability)
    return x

def validate_split(left: float, right: float, opt: float) -> None:
    """
    Validates that the sum of the left and right split probabilities equals the opt value.

    Parameters:
    left (float): The left split probability.
    right (float): The right split probability.
    opt (float): The total probability value that left and right should sum up to.

    Raises:
    ValueError: If the sum of left and right does not equal opt.

    Returns:
    None
    """
    if (left + right) != opt:
        raise ValueError("Left and right split values must add up to opt.")

def main(n: int, opt: float, flip_type: str, flip_symmetry: str, left: float = 0, right: float = 0) -> None:
    sigma = 1  # sigma is always 1
    corrupted = opt > 0

    if flip_type == 'one_side':
        r = find_x_for_probability(opt)
        range_check = lambda x: x > r
    elif flip_type == 'both_sides':
        if flip_symmetry == 'symmetric':
            r = find_x_for_probability(opt / 2)
            range_check = lambda x: abs(x) > abs(r)
        elif flip_symmetry == 'asymmetric':
            validate_split(left, right, opt)
            r_left = find_x_for_probability(left)
            r_right = find_x_for_probability(right)
            range_check = lambda x: x < -r_left or x > r_right
    else:
        raise ValueError("Invalid flip_type. Must be 'one_side' or 'both_sides'.")

    print(f'Calculated r: {find_x_for_probability(opt)} for one_side or (r_left: {r_left}, r_right: {r_right}) for both_sides.')

    # Generate the output file name based on provided options
    if flip_type == 'asymmetric':
        output_file = f"data_{flip_type}_{int(left*100)}%_{int(right*100)}%.csv"
    else:
        output_file = f"data_{flip_type}_{int(opt*100)}%.csv"

    file_path = os.path.join('data', output_file)

    with open(file_path, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

	# write information about the distribution into the .csv file
        csv_writer.writerow([n / 10, sigma, r])

        for _ in range(n):
            x = np.random.normal(loc=0.0, scale=sigma)
            y = -1 if x < 0 else 1
            if corrupted and range_check(x):
                y *= -1

            csv_writer.writerow([y, x])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate normally distributed data according to N(0,1) and write to CSV.")

    parser.add_argument("-o", type=float, required=True, help="Noise level opt.")
    parser.add_argument("-n", type=int, default=10000, help="Number of samples.")
    parser.add_argument("--out_dir", type=str, default="data/", help="Output CSV directory.")
    parser.add_argument("--flip_type", type=str, choices=["one_side", "both_sides"], required=True, help="Flip type: 'one_side' or 'both_sides'.")
    parser.add_argument("--flip_symmetry", type=str, choices=["symmetric", "asymmetric"], help="Flip symmetry for 'both_sides': 'symmetric' or 'asymmetric'.")
    parser.add_argument("--left", type=float, default=0, help="Left split for asymmetric flipping.")
    parser.add_argument("--right", type=float, default=0, help="Right split for asymmetric flipping.")

    args = parser.parse_args()

    main(args.n, args.opt, args.flip_type, args.flip_symmetry, args.left, args.right)
