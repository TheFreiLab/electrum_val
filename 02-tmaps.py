import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tmap as tm
from electrum import calculate_fingerprints
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = 'Menlo'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update({'font.size': 10})


def compute_layout(fps, path, knn=500):
    if os.path.exists(path):
        return joblib.load(path)

    knn_search = NearestNeighbors(n_neighbors=knn, metric='manhattan')
    knn_search.fit(fps)

    edge_list = []
    for i in range(len(fps)):
        dists, idxs = knn_search.kneighbors(fps[i].reshape(1, -1))
        for j in range(knn):
            edge_list.append([i, idxs[0, j], dists[0, j]])

    cfg = tm.LayoutConfiguration()
    cfg.node_size = 1 / 30
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 5
    cfg.k = 15
    cfg.sl_scaling_type = tm.RelativeToAvgLength

    x, y, s, t, _ = tm.layout_from_edge_list(len(fps), edge_list, cfg)
    layout = {'x': list(x), 'y': list(y), 's': list(s), 't': list(t)}
    joblib.dump(layout, path)
    return layout


def plot_tmap(x, y, s, t, color_labels, legend_values, legend_labels, cmap, out_path, title, linewidth=0.01):
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(x, y, s=0.5, c=color_labels, cmap=cmap)

    for src, tgt in zip(s, t):
        plt.plot([x[src], x[tgt]], [y[src], y[tgt]], 'k-', linewidth=linewidth)

    ax = plt.gca()
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    normed_colors = scatter.cmap(scatter.norm(legend_values))
    handles = [Patch(color=normed_colors[i], label=label) for i, label in enumerate(legend_labels)]
    ax.legend(handles=handles, title=title, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.savefig(out_path, dpi=1200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    os.makedirs('figures/tmaps', exist_ok=True)

    # --- Oxidation State ---
    df_os = pd.read_csv('datasets/oxidationstate_46k.csv', dtype={'oxidation_states': str})
    df_os = df_os.sort_values('oxidation_states').drop_duplicates(subset='smiles').sample(frac=1, random_state=42).reset_index(drop=True)

    fps_os = np.array(calculate_fingerprints(df_os['LigandSmiles'], df_os['Metal'], radius=2, n_bits=512))
    layout_os = compute_layout(fps_os, 'figures/tmaps/oxidationstate_layout.pkl')

    classes_os = df_os['classification']
    oxidation_state_labels = df_os['oxidation_states'].astype(str).unique()
    class_order = sorted(df_os['classification'].unique())
    oxidation_label_map = dict(zip(class_order, sorted(df_os['oxidation_states'].astype(str).unique(), key=lambda x: int(x))))

    legend_vals = class_order
    legend_labels = [f'+{oxidation_label_map[c]}' if oxidation_label_map[c] != '0' else '0' for c in class_order]

    plot_tmap(
        x=layout_os['x'],
        y=layout_os['y'],
        s=layout_os['s'],
        t=layout_os['t'],
        color_labels=classes_os,
        legend_values=legend_vals,
        legend_labels=legend_labels,
        cmap='turbo',
        out_path='figures/tmaps/oxidationstate_tmap.png',
        title='Oxidation State'
    )

    # --- Coordination Number ---
    df_cn = pd.read_csv('datasets/coordnumber.csv')
    df_cn = df_cn.groupby('classification').apply(lambda x: x.sample(n=2000, replace=True, random_state=42)).reset_index(drop=True)
    df_cn = df_cn.sample(frac=1, random_state=42).reset_index(drop=True)

    fps_cn = np.array(calculate_fingerprints(df_cn['LigandSmiles'], df_cn['Metal'], radius=2, n_bits=512))
    layout_cn = compute_layout(fps_cn, 'figures/tmaps/coordnumber_layout.pkl')

    class_vals_cn = df_cn['classification']
    class_unique = sorted(df_cn['classification'].unique())
    legend_labels_cn = [str(int(val)) for val in class_unique]

    plot_tmap(
        x=layout_cn['x'],
        y=layout_cn['y'],
        s=layout_cn['s'],
        t=layout_cn['t'],
        color_labels=class_vals_cn,
        legend_values=class_unique,
        legend_labels=legend_labels_cn,
        cmap='turbo',
        out_path='figures/tmaps/coordnumber_tmap.png',
        title='Coordination Number',
        linewidth=0.1
    )
