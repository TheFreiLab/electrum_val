import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Paths 
BASE = Path(__file__).resolve().parent
RES = BASE / 'results' / 'composition'
DATA = BASE / 'datasets'
PLOTS = BASE / 'figures' / 'composition'
PLOTS.mkdir(parents=True, exist_ok=True)

# Helper functions for plotting

def _save(fig, name):
    out = PLOTS / name
    fig.tight_layout()
    fig.savefig(out, dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out}')

def _bar_from_series(s: pd.Series, title: str, xlabel: str, fname: str, max_items=None, rotate=45):
    s = s.copy()
    s.index = s.index.astype(str)
    s = s.sort_values(ascending=False)
    if max_items is not None:
        s = s.head(max_items)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(s)), s.values)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, rotation=rotate, ha='right')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    _save(fig, fname)

def _heatmap(df: pd.DataFrame, title: str, fname: str):
    df_plot = df.copy()
    df_plot.index = df_plot.index.astype(str)
    df_plot.columns = df_plot.columns.astype(str)
    data = df_plot.values.astype(float)
    data[data == 0] = np.nan
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="black")

    fig = plt.figure(figsize=(max(6, 0.35*len(df_plot.columns)+2),
                              max(6, 0.35*len(df_plot.index)+2)))
    ax = fig.gca()
    im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap=cmap)

    ax.set_xticks(np.arange(df_plot.shape[1]))
    ax.set_yticks(np.arange(df_plot.shape[0]))
    ax.set_xticklabels(df_plot.columns, rotation=90)
    ax.set_yticklabels(df_plot.index)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('count')
    _save(fig, fname)

# Load composition output files

cn_counts = pd.read_csv(RES / 'cn_counts.csv', index_col=0).squeeze('columns')
os_counts = pd.read_csv(RES / 'os_counts.csv', index_col=0).squeeze('columns')

metal_counts_cn = pd.read_csv(RES / 'metal_counts_cn.csv', index_col=0).squeeze('columns')
metal_counts_os = pd.read_csv(RES / 'metal_counts_os.csv', index_col=0).squeeze('columns')
tmq_metal_counts = pd.read_csv(RES / 'tmqmg_metal_counts.csv', index_col=0).squeeze('columns')

cn_by_metal = pd.read_csv(RES / 'cn_by_metal.csv', index_col=0)
os_by_metal = pd.read_csv(RES / 'os_by_metal.csv', index_col=0)

df_tmq = pd.read_csv(DATA / 'tmQMg.csv') # tmQMg raw for property distributions

# Plot distributions

_bar_from_series(cn_counts.iloc[:, 0] if isinstance(cn_counts, pd.DataFrame) else cn_counts,
                 'Coordination number distribution', 'coordination number',
                 'cn_distribution_bar.png', max_items=None, rotate=0)

_bar_from_series(os_counts.iloc[:, 0] if isinstance(os_counts, pd.DataFrame) else os_counts,
                 'Oxidation state distribution', 'oxidation state',
                 'os_distribution_bar.png', max_items=None, rotate=0)

_bar_from_series(metal_counts_cn.iloc[:, 0] if isinstance(metal_counts_cn, pd.DataFrame) else metal_counts_cn,
                 'Metal distribution (CN dataset)', 'metal',
                 'cn_metal_distribution_bar_top20.png', max_items=20, rotate=90)

_bar_from_series(metal_counts_os.iloc[:, 0] if isinstance(metal_counts_os, pd.DataFrame) else metal_counts_os,
                 'Metal distribution (OS dataset)', 'metal',
                 'os_metal_distribution_bar_top20.png', max_items=20, rotate=90)

_bar_from_series(tmq_metal_counts.iloc[:, 0] if isinstance(tmq_metal_counts, pd.DataFrame) else tmq_metal_counts,
                 'Metal distribution (tmQMg dataset)', 'metal',
                 'tmqmg_metal_distribution_bar_top20.png', max_items=20, rotate=90)

# Plot heatmaps

TOP_N_HEATMAP = 20 # To keep heatmaps readable

def _topN_heat(df_counts: pd.Series, cross_tab: pd.DataFrame, N: int):
    if isinstance(df_counts, pd.DataFrame):
        s = df_counts.iloc[:,0]
    else:
        s = df_counts
    top = s.sort_values(ascending=False).head(N).index.astype(str)
    df_h = cross_tab.copy()
    df_h = df_h.loc[df_h.index.astype(str).isin(top)]
    df_h = df_h.loc[df_h.sum(axis=1).sort_values(ascending=False).index]
    try:
        cols_sorted = sorted(df_h.columns, key=lambda x: float(x))
        df_h = df_h[cols_sorted]
    except Exception:
        pass
    return df_h

cn_heat = _topN_heat(metal_counts_cn, cn_by_metal, TOP_N_HEATMAP)
_heatmap(cn_heat, f'Metal x Coordination number (top {min(TOP_N_HEATMAP, cn_heat.shape[0])} metals)', 'cn_metal_cn_heatmap.png')

os_heat = _topN_heat(metal_counts_os, os_by_metal, TOP_N_HEATMAP)
_heatmap(os_heat, f'Metal x Oxidation state (top {min(TOP_N_HEATMAP, os_heat.shape[0])} metals)', 'os_metal_os_heatmap.png')