#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Healthy-liver-only preprocessing and QC pipeline for single-cell RNA-seq data.

What this script does:
- Loads Healthy AnnData files into a list (`adatas`) and standardizes metadata.
- Performs QC metrics computation and basic filtering (genes/cells, MT content).
- Visualizes QC distributions per sample (Seaborn facet KDEs).
- Removes outliers using MAD-based thresholds.
- Harmonizes DX labels for healthy samples.
- Detects doublets per sample using scVI + SOLO and filters them out.
- Concatenates per-sample AnnData objects into a single `adata` and writes a checkpoint.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation as mad

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

# ----------------------------------------------------------------------
# Load Healthy samples
# ----------------------------------------------------------------------
adatas = [x for x in os.listdir('/home/mdiaz/sc_liver_data/healthy_h5ad_adata') if x.endswith('.h5ad')]

def load_it(adata):
    """Load one Healthy AnnData file and standardize metadata fields."""
    samp = adata.split('_')[0]
    dx = adata.split('_')[1]
    adata = sc.read_h5ad('/home/mdiaz/sc_liver_data/healthy_h5ad_adata/' + adata)
    adata.obs['Patient'] = samp
    adata.obs['DX'] = dx
    adata.obs['Sample'] = adata.obs['Patient'] + '_' + adata.obs['DX']
    adata.obs.index = adata.obs.index + '-' + samp + '_' + dx
    return adata

adatas = [load_it(ad) for ad in adatas]
for adata in adatas:
    adata.var_names_make_unique()
    print(f"Unique variable names for {adata.obs['Sample'][0]}: {adata.var_names.is_unique}")

# ----------------------------------------------------------------------
# QC
# ----------------------------------------------------------------------
def qc(adata):
    """Basic QC filtering and mitochondrial metrics calculation."""
    sc.pp.filter_cells(adata, min_genes=200)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars="mt", inplace=True, percent_top=[20], log1p=True)
    remove = ['total_counts_mt', 'log1p_total_counts_mt']
    adata.obs = adata.obs[[x for x in adata.obs.columns if x not in remove]]
    return adata

adatas = [qc(ad) for ad in adatas]
df = pd.concat(x.obs for x in adatas).sort_values('Sample')

# ----------------------------------------------------------------------
# Outlier removal
# ----------------------------------------------------------------------
def mad_outlier(adata, metric, nmads, upper_only=False):
    M = adata.obs[metric]
    if not upper_only:
        return (M < np.median(M) - nmads * mad(M)) | (M > np.median(M) + nmads * mad(M))
    return (M > np.median(M) + nmads * mad(M))

def pp(adata):
    adata = adata[adata.obs.pct_counts_mt < 25]
    bool_vector = (
        mad_outlier(adata, 'log1p_total_counts', 5) +
        mad_outlier(adata, 'log1p_n_genes_by_counts', 5) +
        mad_outlier(adata, 'pct_counts_in_top_20_genes', 5) +
        mad_outlier(adata, 'pct_counts_mt', 3, upper_only=True)
    )
    adata = adata[~bool_vector]
    adata.uns['cells_removed'] = int(np.sum(bool_vector))
    return adata

adatas = [pp(ad) for ad in adatas]
for adata in adatas:
    print(f"Remaining cells: {len(adata)}, Removed cells: {adata.uns['cells_removed']}")

# ----------------------------------------------------------------------
# Harmonize DX labels
# ----------------------------------------------------------------------
for adata in adatas:
    adata.obs['DX'] = adata.obs['DX'].replace(['non-tumor', 'PBC001', 'PBC005'], 'healthy')

df['DX'] = df['DX'].replace(['non-tumor', 'PBC001', 'PBC005'], 'healthy')
print("DX labels in df:", df['DX'].unique())

# ----------------------------------------------------------------------
# Doublet detection using scVI + SOLO
# ----------------------------------------------------------------------
for i, adata in enumerate(adatas):
    print(f"Processing dataset {i+1}")
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False, flavor='seurat_v3')
    sc.pp.pca(adata, n_comps=50, mask_var="highly_variable")
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)
    print(f"Shape after filtering: {adata.shape}")

    if hasattr(adata.X, "tocsr"):
        adata.X = adata.X.tocsr()

    scvi.model.SCVI.setup_anndata(adata)
    vae = scvi.model.SCVI(adata)
    vae.train()

    solo = scvi.external.SOLO.from_scvi_model(vae)
    solo.train()

    df_pred = solo.predict()
    df_pred['prediction'] = solo.predict(soft=False)
    df_pred['dif'] = df_pred.doublet - df_pred.singlet
    doublets = df_pred[(df_pred.prediction == 'doublet') & (df_pred.dif > 0.9)]
    print(f"Doublets detected: {doublets.shape[0]}")

    adata.obs['doublet'] = adata.obs.index.isin(doublets.index)
    adata = adata[~adata.obs['doublet']]
    adatas[i] = adata
    print(f"Remaining cells after doublet removal: {adata.shape}")

# ----------------------------------------------------------------------
# Concatenate and save
# ----------------------------------------------------------------------
adata = sc.concat(adatas, join='outer')
print(adata)
adata.write('/home/mdiaz/HCC_project/healthy_adata/0C_doub_remov.h5ad')
