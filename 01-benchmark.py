import numpy as np
import pandas as pd
import argparse
import os

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

from electrum import calculate_fingerprints as fp_electrum
from electrum_ligands import calculate_fingerprints as fp_ligands
from electrum_atomic import calculate_fingerprints as fp_atomic

def calculate_metrics(y_true, y_pred, y_true_onehot, y_pred_onehot):
    return {
        'roc_auc_ovr_macro': roc_auc_score(y_true_onehot, y_pred_onehot, average='macro', multi_class='ovr'),
        'roc_auc_ovr_weighted': roc_auc_score(y_true_onehot, y_pred_onehot, average='weighted', multi_class='ovr'),
        'prc_auc_macro': average_precision_score(y_true_onehot, y_pred_onehot, average='macro'),
        'prc_auc_weighted': average_precision_score(y_true_onehot, y_pred_onehot, average='weighted'),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }

def run_cv(X, y, groups, n_splits=3, seed=42):
    model = MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64, 32), max_iter=1000, random_state=seed)
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lb = LabelBinarizer().fit(y)

    results = []
    for train_idx, test_idx in kf.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        y_true_1hot = lb.transform(y[test_idx])
        y_pred_1hot = lb.transform(y_pred)

        metrics = calculate_metrics(y[test_idx], y_pred, y_true_1hot, y_pred_1hot)
        results.append(metrics)

    return pd.DataFrame(results)

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

    bit_sizes = [256, 512, 1024]
    summary = {}

    for name, fp_func in fingerprints.items():
        for n_bits in bit_sizes:
            print(f'Processing fingerprint: {name} ({n_bits} bits)')

            X = np.array(fp_func(df['LigandSmiles'], df['Metal'], radius=2, n_bits=n_bits))
            y_true = np.array(df['classification'])
            groups = df['LigandSmiles']

            # True labels
            df_true = run_cv(X, y_true, groups)
            mean = df_true.mean() * 100
            std = df_true.std() * 100
            summary[f'{name}_{n_bits}'] = [f'{m:.1f} ± {s:.2f}' for m, s in zip(mean, std)]

            # Scrambled labels
            y_scrambled = np.random.permutation(y_true)
            df_scrambled = run_cv(X, y_scrambled, groups)
            mean_s = df_scrambled.mean() * 100
            std_s = df_scrambled.std() * 100
            summary[f'{name}_{n_bits}_scrambled'] = [f'{m:.1f} ± {s:.2f}' for m, s in zip(mean_s, std_s)]

    summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=mean.index)
    summary_df.index.name = 'Fingerprint'

    base = os.path.splitext(os.path.basename(args.file))[0]
    out_path = f'results/{base}_benchmark.csv'
    summary_df.to_csv(out_path)
    print(f'Saved: {out_path}')