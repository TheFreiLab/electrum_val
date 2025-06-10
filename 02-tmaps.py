import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from sklearn.neighbors import NearestNeighbors

import tmap as tm
from electrum import calculate_fingerprints

def prepare_tmap_layout(fps: np.ndarray, layout_path: str, knn: int = 500) -> dict:
    if os.path.exists(layout_path):
        return joblib.load(layout_path)

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
    joblib.dump(layout, layout_path)
    return layout

def plot_tmap(layout: dict, labels, legend_labels, cmap, out_path: str, legend_title: str):
    x, y = layout['x'], layout['y']

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(x, y, s=0.5, c=labels, cmap=cmap)

    for src, tgt in zip(layout['s'], layout['t']):
        plt.plot([x[src], x[tgt]], [y[src], y[tgt]], 'k-', linewidth=0.01)

    ax = plt.gca()
    ax.spines[['top', 'right', 'left', 'bottom']] = [False] * 4
    ax.set_xticks([])
    ax.set_yticks([])

    unique_vals = np.unique(labels)
    normed_colors = scatter.cmap(scatter.norm(unique_vals))
    handles = [Patch(color=normed_colors[i], label=str(label)) for i, label in enumerate(legend_labels)]
    ax.legend(handles=handles, title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.savefig(out_path, dpi=1200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    os.makedirs('figures/tmaps', exist_ok=True)

    # Oxidation state TMAP
    df_os = pd.read_csv('datasets/oxidationstate_46k.csv', dtype={'oxidation_states': str})
    df_os = df_os.sort_values('oxidation_states').drop_duplicates(subset='smiles').reset_index(drop=True)
    fps_os = np.array(calculate_fingerprints(df_os['LigandSmiles'], df_os['Metal'], radius=2, n_bits=512))

    layout_os = prepare_tmap_layout(fps_os, 'figures/tmaps/oxidationstate_layout.pkl')
    plot_tmap(layout_os, df_os['oxidation_states_classification'], legend_labels=['+1', '+2', '+3', '+4', '+5', '+6', '0'],
              cmap='turbo', out_path='figures/tmaps/oxidationstate_tmap.png', legend_title='Oxidation State')

    # Coordination number TMAP
    df_cn = pd.read_csv('datasets/coordnumber.csv')
    df_cn = df_cn.groupby('bondorder').apply(lambda x: x.sample(n=2000, replace=True, random_state=42)).reset_index(drop=True)
    fps_cn = np.array(calculate_fingerprints(df_cn['LigandSmiles'], df_cn['Metal'], radius=2, n_bits=512))

    layout_cn = prepare_tmap_layout(fps_cn, 'figures/tmaps/coordnumber_layout.pkl')
    plot_tmap(layout_cn, df_cn['bondorder'], legend_labels=np.unique(df_cn['bondorder']),
              cmap='turbo', out_path='figures/tmaps/coordnumber_tmap.png', legend_title='Coordination Number')