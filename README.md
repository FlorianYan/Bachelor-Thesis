# Efficient Algorithms for Agnostically Learning Halfspaces Under Distributional Assumptions

This repository contains code for benchmarking two algorithms $L_1$ and $L_2$—described in the bachelor thesis titled **"Efficient Algorithms for Agnostically Learning Halfspaces Under Distributional Assumptions."** These algorithms are evaluated using synthetic data generated from a distribution, where each dataset consists of 1D points drawn from the standard normal distribution.

## Overview

The goal of this repository is to benchmark two algorithms for agnostic learning of halfspaces. Specifically, the two algorithms are:

- **$L_1$ algorithm**: A method for solving the learning problem using the $L_1$ norm.
- **$L_2$ algorithm**: A method for solving the learning problem using the $L_2$ norm.

The repository includes scripts to generate synthetic data, apply these algorithms, and store the results in CSV format for evaluation. This is a small part of the overall thesis, which is mainly theoretical in nature.

## Repository Structure

To better organize the files, here’s a recommended directory structure:
```
/Efficient-Halfspace-Learning
├── data/
│   ├── dataset1.csv
│   ├── dataset2.csv
│   ├── ...
├── src/
│   ├── data_generator.py
│   ├── solver.py
│   ├── utils.py
├── results/
│   ├── L1_results.csv
│   ├── L2_results.csv
│   ├── ...
├── README.md
└── requirements.txt
```

### Directory Breakdown:

- `data/`: Contains the generated datasets in CSV format (e.g., `dataset1.csv`, `dataset2.csv`, etc.).
- `src/`: Contains the Python scripts.
  - `data_generator.py`: Script for generating synthetic datasets with 1D points sampled from the standard normal distribution and labeled based on their sign.
  - `solver.py`: Script implementing both the $L_1$ and $L_2$ algorithms for learning the halfspaces.
  - `utils.py`: (optional) Any helper functions used in the scripts.
- `results/`: Directory where output files (e.g., algorithm results) are saved, like `L1_results.csv` and `L2_results.csv`.
- `requirements.txt`: List of required Python packages for running the code (e.g., `numpy`, `pandas`, `matplotlib`).

## Getting Started

To run the code and replicate the experiments from the thesis, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/Efficient-Halfspace-Learning.git
cd Efficient-Halfspace-Learning
```

### 2. Install required dependencies:

You can create a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Generate datasets::

To generate a dataset, you can run the data_generator.py script. By default, it will generate 10,000 points (you can change the number of points by passing the --num_points argument).
The datasets, which were used in the thesis are found under `data/`

```bash
python src/data_generator.py --num_points 10000 --output_file data/dataset1.csv
```
This will generate a CSV file (dataset1.csv) in the data/ directory.

### 4. Run the algorithms:

Once you have generated the datasets, you can apply the $L_1$ and $L_2$ algorithms. The results will be saved in the results/ directory.

To run the $L_1$ solver:
```bash
python src/solver.py --algorithm L1 --input_file data/dataset1.csv --output_file results/L1_results.csv
```

To run the $L_2$ solver:
```bash
python src/solver.py --algorithm L2 --input_file data/dataset1.csv --output_file results/L2_results.csv
```

### 5. Explore Results:

The results of each algorithm will be saved in the results/ directory as CSV files. You can analyze these results or visualize them as needed.

### License:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
