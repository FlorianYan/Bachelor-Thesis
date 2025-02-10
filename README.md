# Efficient Algorithms for Agnostically Learning Halfspaces Under Distributional Assumptions

This repository contains code for benchmarking the $L_1$ and $L_2$ algorithms described in the bachelor thesis titled **"Efficient Algorithms for Agnostically Learning Halfspaces Under Distributional Assumptions."** These algorithms are evaluated using synthetic data generated from the standard normal distribution.

## Overview

The goal of this repository is to benchmark two algorithms for agnostic learning of halfspaces. Specifically, the two algorithms are:

- **$L_1$ algorithm**: An algorithm based on $L_1$-norm polynomial regression for learning halfspaces.
- **$L_2$ algorithm**: An algorithm based on $L_2$-norm polynomial regression for learning halfspaces.

The repository includes scripts to generate synthetic data, apply these algorithms, and store the results in CSV format for evaluation. This is a small part of the overall thesis, which is mainly theoretical in nature.

## Repository Structure

```
/Bachelor-Thesis
├── data/
│   ├── data__0%.csv
│   ├── data_5%.csv
│   ├── ...
├── src/
│   ├── data_generator.py
│   ├── solver.py
├── results/
│   ├── data_generator.py
├── README.md
└── requirements.txt
```

### Directory Breakdown:

- `data/`: Contains the generated datasets in CSV format (e.g., `dataset1.csv`, `dataset2.csv`, etc.).
- `src/`: Contains the Python scripts.
  - `data_generator.py`: Script for generating synthetic datasets with 1D points sampled from the standard normal distribution and then labeled.
  - `solver.py`: Script implementing both the $L_1$ and $L_2$ algorithms for learning the halfspaces.
- `results/`: Directory where algorithm results are saved for each dataset. 
- `requirements.txt`: List of required Python packages for running the code.

## Getting Started

To run the code and replicate the experiments from the thesis, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/Efficient-Halfspace-Learning.git
cd Bachelor-Thesis
```

### 2. Install required dependencies:

You can create a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Generate datasets::

To generate a dataset, you can run the data_generator.py script. 
The datasets, which were used in the thesis are found under `data/`
For example, to generate 10 000 datapoints with opt=0.1 and labels flipped on ones side, run the following command

```bash
python src/data_generator.py -o 0.1 --flip_type one_side 
```
This will generate a CSV file in the `data/` directory.

### 4. Run the algorithms:

Once you have generated the datasets, you can apply the $L_1$ and $L_2$ algorithms and compare their performance.

`.ipynb` solvers were used during the actual benchmarking of the thesis, the `.py` scripts for streamlined usage are currently under construction.
