import numpy as np
import pandas as pd
import argparse
import os
import multiprocessing
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from electrum import calculate_fingerprints as fp_electrum
from electrum_ligands import calculate_fingerprints as fp_ligands
from electrum_atomic import calculate_fingerprints as fp_atomic

def calculate_regression_metrics(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }

def run_cv_regression(X, y, groups, n_splits=3, seed=42):
    model = MLPRegressor(hidden_layer_sizes=(512, 256, 128, 64, 32), max_iter=1000, random_state=seed)
    bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results = []
    for train_idx, test_idx in kf.split(X, bins, groups):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        metrics = calculate_regression_metrics(y[test_idx], y_pred)
        results.append(metrics)

    return pd.DataFrame(results)

def evaluate_fingerprint_target(fp_func, X, y_true, groups, name, n_bits, target):
    result = {}

    # True labels
    df_true = run_cv_regression(X, y_true, groups)
    mean = df_true.mean()
    std = df_true.std()
    result[f'{name}_{n_bits}_{target}'] = [f'{m:.3f} ± {s:.3f}' for m, s in zip(mean, std)]

    # Scrambled labels
    #y_scrambled = np.random.permutation(y_true)
    #df_scrambled = run_cv_regression(X, y_scrambled, groups)
    #mean_s = df_scrambled.mean()
    #std_s = df_scrambled.std()
    #result[f'{name}_{n_bits}_{target}_scrambled'] = [f'{m:.3f} ± {s:.3f}' for m, s in zip(mean_s, std_s)]

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to input CSV file')
    args = parser.parse_args()
    np.random.seed(42)

    df = pd.read_csv(args.file)

    fingerprints = {
        'electrum': fp_electrum,
        'ligands': fp_ligands,
        'atomic': fp_atomic,
    }

    bit_sizes = [1024]

    regression_targets = [
        "tzvp_lumo_energy", "tzvp_homo_energy", "tzvp_homo_lumo_gap", "homo_lumo_gap_delta",
        "tzvp_electronic_energy", "electronic_energy_delta", "tzvp_dispersion_energy",
        "dispersion_energy_delta", "enthalpy_energy", "enthalpy_energy_correction",
        "gibbs_energy", "gibbs_energy_correction", "zpe_correction", "heat_capacity",
        "entropy", "tzvp_dipole_moment", "dipole_moment_delta", "polarisability",
        "lowest_vibrational_frequency", "highest_vibrational_frequency"
    ]

    summary = {}

    for name, fp_func in fingerprints.items():
        for n_bits in bit_sizes:
            print(f'Processing fingerprint: {name} ({n_bits} bits)')
            X = np.array(fp_func(df['LigandSmiles'], df['Metal'], radius=2, n_bits=n_bits))
            groups = df['LigandSmiles']

            tasks = []
            for target in regression_targets:
                print(f'  Queued target: {target}')
                y_true = df[target].to_numpy()
                tasks.append(delayed(evaluate_fingerprint_target)(
                    fp_func, X, y_true, groups, name, n_bits, target
                ))

            n_jobs = min(14, multiprocessing.cpu_count())
            results = Parallel(n_jobs=n_jobs)(tasks)
            for res in results:
                summary.update(res)

    summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['r2', 'mae', 'mse'])
    summary_df.index.name = 'Fingerprint_Target'

    base = os.path.splitext(os.path.basename(args.file))[0]
    out_path = f'results/{base}_regression_benchmark.csv'
    os.makedirs('results', exist_ok=True)
    summary_df.to_csv(out_path)
    print(f'Saved: {out_path}')