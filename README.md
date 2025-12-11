# Electrum Benchmarking Repo

This repository contains benchmarking and visualization tools for evaluating custom molecular fingerprints designed for transition metal complexes. These fingerprints encode ligand environments and metal identity, enabling classification and regression tasks such as predicting coordination numbers, oxidation states, or quantum mechanical properties.

## Project Overview

The core goal is to benchmark and visualize four types of fingerprints:
- `electrum`: metal-aware molecular fingerprints
- `electrum_ligands`: ligand-only fingerprints
- `electrum_atomic`: fingerprints derived from atom-level features

These are tested across datasets for classification (e.g., coordination number, oxidation state) and regression (e.g., HOMO/LUMO energies, dipole moment).

## Directory Structure

```plaintext
├── 00-walkthrough.ipynb         # Optional exploration notebook
├── 01-benchmark.py              # MLP classification benchmark
├── 02-benchmark-regression.py   # MLP regression benchmark
├── 03-benchmark-knn.py          # k-NN classification benchmark
├── 04-tmaps.py                  # TMAP visualization script
├── 05-dataset-analysis.py       # Statistics on datasets
├── 06-composition-plots.py      # Plots for statistics
├── electrum*.py                 # Fingerprint generators
├── datasets/                    # Input CSVs (coordination, oxidation, QM data)
├── figures/tmaps/               # Output TMAP visualizations
├── results/                     # CSVs of benchmark results
├── LICENSE
├── README.md
```

## Installation

You can set up the environment using:

```bash
pip install -r requirements.txt
```

Dependencies include:
- `scikit-learn`
- `joblib`
- `pandas`, `numpy`
- `matplotlib`
- `tmap` and `faerun` for visualizations

## How to Run Benchmarks

### 1. Classification with MLP

```bash
python 01-benchmark.py --file datasets/oxidationstate_46k.csv
```

This benchmarks multiple fingerprint types and sizes (256, 512, 1024 bits) on a classification task, and compares to a scrambled label baseline. Output is saved to:

```
results/oxidationstate_46k_benchmark.csv
```

### 2. Regression with MLP

```bash
python 02-benchmark-regression.py --file datasets/tmQMg.csv
```

This performs multi-target regression on ~20 QM properties using MLPs, with cross-validation and error reporting per target:

```
results/tmQMg_regression_benchmark.csv
```

### 3. Classification with k-NN

```bash
python 03-benchmark-knn.py --file datasets/oxidationstate_46k.csv
```

Uses a 5-NN classifier with Manhattan distance:

```
results/oxidationstate_46k_knn_benchmark.csv
```

## 4. Visualizing Fingerprints with TMAP

Generate both static and interactive 2D layouts of fingerprint similarity using:

```bash
python 04-tmaps.py
```

Outputs:
- `figures/tmaps/coordnumber_tmap.png` — color-coded by coordination class
- `figures/tmaps/oxidationstate_tmap.html` — interactive map with CSD links

Please note that TMAP only works on python version 3.8 or lower. So to re-generate the TMAPs you will need to create a new virtual environment with python 3.8 and install the required dependencies.

## 5. Walkthrough Notebook

An optional Jupyter notebook (`00-walkthrough.ipynb`) provides a guided exploration of the code, including:

- Overview of fingerprint generation
- Effect of fingerprint parameters (`radius`, `n_bits`)
- Batch processing of multiple complexes
- Debugging tips for common issues
- Example machine learning workflow with feature importances

## Installation of ELECTRUM

We have also created a package called `electrum-fp` that provides the fingerprinting functionality. You can install it via pip:

```bash
pip install electrum-fp
```

## Usage

```python
from electrum_fp.electrum import calculate_fingerprint

ligands = "Cc1c(C)c(C)c(c2cccc3cccnc32)c1C.Cl.Cl"        # Ligand SMILES
metal = "Rh"                                             # Corresponding metal

fps = calculate_fingerprint(ligands, metal, radius=2, n_bits=512)
print(fps) 
``` 

## Contributing

We welcome contributions from the community! If you develop new fingerprint functions, add tutorials, improve documentation, or create new benchmark scripts, please consider submitting a pull request. Contributions that help others understand how you used, extended, or applied ELECTRUM are especially appreciated.

Before contributing, please check the existing code and notebooks for style and structure to keep the repository consistent.

## Citation & License

CCDC have reviewed the data and approved the publication. 

If you use this code in a publication, please cite as follows: 

```
@Article{D5DD00145E,
author ="Orsi, Markus and Frei, Angelo",
title  ="ELECTRUM: an electron configuration-based universal metal fingerprint for transition metal compounds",
journal  ="Digital Discovery",
year  ="2025",
volume  ="4",
issue  ="12",
pages  ="3567-3577",
publisher  ="RSC",
doi  ="10.1039/D5DD00145E"
}
```

This project is released under the MIT License.
