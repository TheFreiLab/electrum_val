import pandas as pd
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent
DATA = BASE / 'datasets'
OUTDIR = BASE / 'results' / 'composition'
OUTDIR.mkdir(parents=True, exist_ok=True)

def write_csv(df_or_series, filename, **kwargs):
    path = OUTDIR / filename
    if hasattr(df_or_series, 'to_csv'):
        df_or_series.to_csv(path, **kwargs)
    else:
        pd.DataFrame(df_or_series).to_csv(path, **kwargs)
    print(f'[wrote] {path} ({"exists" if path.exists() else "missing"})')

# CN dataset
df_cn = pd.read_csv(DATA / 'coordnumber.csv')
assert {'Metal', 'classification', 'LigandSmiles'}.issubset(df_cn.columns), df_cn.columns.tolist()
print(f'[CN] rows={len(df_cn)}, metals={df_cn["Metal"].nunique()}, CNclasses={df_cn["classification"].nunique()}')

cn_counts = df_cn['classification'].value_counts().sort_index()
write_csv(cn_counts, 'cn_counts.csv', header=['count'])
metal_counts_cn = df_cn['Metal'].value_counts()
write_csv(metal_counts_cn, 'metal_counts_cn.csv', header=['count'])
cn_by_metal = pd.crosstab(df_cn['Metal'], df_cn['classification'])
write_csv(cn_by_metal, 'cn_by_metal.csv')
write_csv(pd.DataFrame({
    'n_rows': [len(df_cn)],
    'n_metals': [df_cn['Metal'].nunique()],
    'n_coordination_classes': [df_cn['classification'].nunique()],
    'top5_metals': [', '.join(metal_counts_cn.head(5).index.astype(str))],
    'top5_cn': [', '.join(cn_counts.sort_values(ascending=False).head(5).index.astype(str))]
}), 'cn_overview.csv', index=False)

# OS dataset
df_os = pd.read_csv(DATA / 'oxidationstate_46k.csv')
assert {'Metal', 'oxidation_states', 'LigandSmiles'}.issubset(df_os.columns), df_os.columns.tolist()
print(f'[OS] rows={len(df_os)}, metals={df_os["Metal"].nunique()}, oxStates={df_os["oxidation_states"].nunique()}')

os_counts = df_os['oxidation_states'].value_counts().sort_index()
write_csv(os_counts, 'os_counts.csv', header=['count'])
metal_counts_os = df_os['Metal'].value_counts()
write_csv(metal_counts_os, 'metal_counts_os.csv', header=['count'])
os_by_metal = pd.crosstab(df_os['Metal'], df_os['oxidation_states'])
write_csv(os_by_metal, 'os_by_metal.csv')
write_csv(pd.DataFrame({
    'n_rows': [len(df_os)],
    'n_metals': [df_os['Metal'].nunique()],
    'n_ox_states': [df_os['oxidation_states'].nunique()],
    'top5_metals': [', '.join(metal_counts_os.head(5).index.astype(str))],
    'top5_ox': [', '.join(os_counts.sort_values(ascending=False).head(5).index.astype(str))]
}), 'os_overview.csv', index=False)

# tmQMg dataset 
df_tmq = pd.read_csv(DATA / 'tmQMg.csv')
assert {'Metal', 'LigandSmiles'}.issubset(df_tmq.columns), df_tmq.columns.tolist()
exclude = {'Metal', 'LigandSmiles'}
prop_cols = [c for c in df_tmq.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_tmq[c])]
if not prop_cols:
    raise ValueError('No numeric property columns detected in tmQMg.csv')
print(f'[tmQMg] rows={len(df_tmq)}, metals={df_tmq["Metal"].nunique()}, props={len(prop_cols)}')

write_csv(pd.DataFrame({
    'n_rows': [len(df_tmq)],
    'n_metals': [df_tmq['Metal'].nunique()],
    'n_properties': [len(prop_cols)],
    'properties': [', '.join(prop_cols)]
}), 'tmqmg_overview.csv', index=False)

summary = []
for col in prop_cols:
    s = df_tmq[col].dropna()
    summary.append({
        'property': col,
        'count': int(s.shape[0]),
        'mean': float(s.mean()),
        'std': float(s.std()),
        'min': float(s.min()),
        'q25': float(s.quantile(0.25)),
        'median': float(s.median()),
        'q75': float(s.quantile(0.75)),
        'max': float(s.max()),
    })
write_csv(pd.DataFrame(summary), 'tmqmg_props_summary.csv', index=False)

metal_counts_tmq = df_tmq['Metal'].value_counts()
write_csv(metal_counts_tmq, 'tmqmg_metal_counts.csv', header=['count'])
per_metal = df_tmq.groupby('Metal')[prop_cols].mean().reset_index()
write_csv(per_metal, 'tmqmg_props_by_metal.csv', index=False)

# Final listing
files = sorted(OUTDIR.glob('*'))
print(f'\n[done] Wrote {len(files)} files to {OUTDIR}')
for f in files:
    print(' -', f.name)