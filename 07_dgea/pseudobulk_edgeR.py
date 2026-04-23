#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pseudobulk_edgeR.py
===================
Unified pseudobulk differential expression analysis using edgeR (via rpy2).

Combines two analyses:
  1. Hepatocytes: HCC tumor vs. Healthy (after removing myeloid contamination)
  2. Malignant cells: HCC tumor cells vs. Healthy hepatocytes

Both analyses remove contaminated Leiden clusters (17, 59, 23, 46, 13, 20)
from the "Hepatocytes" group before running edgeR.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import rpy2.rinterface_lib.callbacks
import anndata2ri
import logging
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
anndata2ri.activate()
pandas2ri.activate()

sc.settings.verbosity = 0
matplotlib.use('Agg')

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────

INPUT_H5AD = os.getenv('INPUT_H5AD', '/home/mdiaz/HCC_project/MERGED_adata/scvi_integrated.h5ad')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/home/mdiaz/HCC_project/MERGED_analysis/DGE_csv_files')
FIGURES_DIR = os.getenv('FIGURES_DIR', '/home/mdiaz/manuscript_revision/new_figures')

# Contaminated Leiden clusters (MARCO > 30%): 17, 59, 23, 46, 13, 20
CONTAMINATED_LEIDEN_CLUSTERS = {'17', '59', '23', '46', '13', '20'}
NUM_CELLS_PER_DONOR = 10
REPLICATES_PER_PATIENT = 3
RANDOM_SEED = 42

# Output filenames
HEP_OUT_CSV = os.path.join(OUTPUT_DIR, 'DGE_hepatocytes_CLEAN_v2.csv')
HEP_OUT_CSV_FULL = os.path.join(OUTPUT_DIR, 'DGE_hepatocytes_CLEAN_v2_allgenes.csv')
MC_OUT_CSV = os.path.join(OUTPUT_DIR, 'Malignant_vs_Healthy_Hepatocytes_CLEAN.csv')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Plot styling
matplotlib.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ───────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

def remove_myeloid_contamination(adata, leiden_col='leiden_r3'):
    """
    Remove contaminated Leiden clusters from Hepatocytes.

    Reclassifies hepatocytes in CONTAMINATED_LEIDEN_CLUSTERS as 'Myeloid_contaminant'.
    Returns modified adata and count of removed cells.
    """
    before = (adata.obs['CellType_harmonized'] == 'Hepatocytes').sum()
    contaminated_mask = (
        (adata.obs['CellType_harmonized'] == 'Hepatocytes') &
        (adata.obs[leiden_col].astype(str).isin(CONTAMINATED_LEIDEN_CLUSTERS))
    )

    ct_col = adata.obs['CellType_harmonized']
    if hasattr(ct_col, 'cat'):
        if 'Myeloid_contaminant' not in ct_col.cat.categories:
            adata.obs['CellType_harmonized'] = ct_col.cat.add_categories(['Myeloid_contaminant'])

    adata.obs.loc[contaminated_mask, 'CellType_harmonized'] = 'Myeloid_contaminant'
    after = (adata.obs['CellType_harmonized'] == 'Hepatocytes').sum()

    return adata, before - after


def aggregate_and_filter(adata_sub, donor_key='Sample', condition_key='label',
                         n_reps=REPLICATES_PER_PATIENT, min_cells=NUM_CELLS_PER_DONOR,
                         seed=RANDOM_SEED):
    """
    Generate pseudobulk by aggregating counts per donor and replicate.

    Returns (counts_df, meta_df): gene counts (genes x samples) and metadata.
    """
    random.seed(seed)
    np.random.seed(seed)

    size_by_donor = adata_sub.obs.groupby(donor_key).size()
    donors_to_drop = [d for d, n in size_by_donor.items() if n <= min_cells]
    if donors_to_drop:
        print(f"        Donors dropped (< {min_cells} cells): {donors_to_drop}")

    rows = []
    for donor in adata_sub.obs[donor_key].unique():
        if donor in donors_to_drop:
            continue
        adata_donor = adata_sub[adata_sub.obs[donor_key] == donor]
        indices = list(adata_donor.obs_names)
        random.shuffle(indices)
        split_idx = np.array_split(np.array(indices), n_reps)
        for rep_i, rep_idx in enumerate(split_idx):
            if len(rep_idx) == 0:
                continue
            adata_rep = adata_donor[rep_idx]
            X_sum = np.asarray(adata_rep.X.sum(axis=0)).flatten()
            row = {'pb_id': f"{donor}_rep{rep_i}",
                   donor_key: donor,
                   condition_key: adata_rep.obs[condition_key].iloc[0],
                   'n_cells': len(rep_idx)}
            row.update(dict(zip(adata_sub.var_names, X_sum)))
            rows.append(row)

    df = pd.DataFrame(rows).set_index('pb_id')
    counts_df = df[adata_sub.var_names]
    meta_df = df.drop(columns=adata_sub.var_names)
    meta_df['pb_id'] = meta_df.index
    return counts_df.T, meta_df


def run_edgeR(counts_mat, meta, contrast_str="Tumor - Healthy"):
    """
    Run edgeR analysis via rpy2.

    Parameters
    ----------
    counts_mat : DataFrame
        Gene counts (genes x samples)
    meta : DataFrame
        Metadata with 'label' column and 'pb_id' index
    contrast_str : str
        Contrast formula (e.g., "Tumor - Healthy")

    Returns
    -------
    DataFrame
        edgeR results with columns: logFC, PValue, FDR, gene_symbol, etc.
    """
    counts_r = counts_mat.astype(float)
    meta_r = meta[['label', 'pb_id']].copy()
    meta_r.index = meta_r['pb_id']

    r.assign('counts_py', counts_r)
    r.assign('meta_py', meta_r)

    r_code = f"""
library(edgeR)

counts  <- as.matrix(counts_py)
meta    <- meta_py

stopifnot(all(colnames(counts) == meta$pb_id))

group  <- factor(meta$label, levels = c('Healthy', 'Tumor'))

y <- DGEList(counts = counts, group = group)
keep <- filterByExpr(y)
y <- y[keep, , keep.lib.sizes = FALSE]
cat(sprintf("Genes after filterByExpr: %d\\n", nrow(y)))

y <- calcNormFactors(y)

design <- model.matrix(~ 0 + group)
colnames(design) <- levels(group)

y <- estimateDisp(y, design = design)
fit <- glmQLFit(y, design)

contrast <- makeContrasts({contrast_str}, levels = design)
qlf  <- glmQLFTest(fit, contrast = contrast)
tt   <- topTags(qlf, n = Inf)$table
tt$gene_symbol <- rownames(tt)
"""

    r(r_code)
    tt_py = r['tt']
    if not isinstance(tt_py, pd.DataFrame):
        tt_py = pd.DataFrame(tt_py)

    return tt_py


def plot_barplot(tt_py, out_path, title="DEGs"):
    """Generate top20 up/down barplot."""
    tt_indexed = tt_py.copy()
    tt_indexed.index = tt_indexed['gene_symbol']

    top20 = pd.concat([
        tt_indexed.nlargest(20, 'logFC'),
        tt_indexed.nsmallest(20, 'logFC')
    ])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top20['logFC']]
    ax.bar(range(len(top20)), top20['logFC'], color=colors)
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels(top20.index, rotation=90, fontsize=10, fontweight='bold')
    ax.set_xlabel("Gene", fontsize=14, fontweight='bold')
    ax.set_ylabel("Log₂ Fold Change", fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles = [mpatches.Patch(color='#d62728', label='Upregulated'),
               mpatches.Patch(color='#1f77b4', label='Downregulated')]
    leg = ax.legend(handles=handles, frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_volcano(tt_py, out_path, title="Volcano plot", genes_highlight=None):
    """Generate volcano plot."""
    LOG2FC_THR = 2.0
    PADJ_THR = 0.05

    df_v = tt_py.copy()
    df_v['neglog10_padj'] = -np.log10(df_v['FDR'].clip(lower=1e-300))
    df_v['color'] = 'grey'
    df_v.loc[(df_v['logFC'] >= LOG2FC_THR) & (df_v['FDR'] <= PADJ_THR), 'color'] = 'red'
    df_v.loc[(df_v['logFC'] <= -LOG2FC_THR) & (df_v['FDR'] <= PADJ_THR), 'color'] = 'blue'

    fig, ax = plt.subplots(figsize=(9, 8))
    label_map = {'red': 'Up', 'blue': 'Down', 'grey': 'NS'}
    for col, grp in df_v.groupby('color', observed=True):
        ax.scatter(grp['logFC'], grp['neglog10_padj'],
                   c=col, s=12, alpha=0.6, linewidths=0, label=label_map[col])

    # Label top genes
    to_label = pd.concat([
        df_v[df_v['color'] == 'red'].nlargest(10, 'logFC'),
        df_v[df_v['color'] == 'blue'].nsmallest(10, 'logFC'),
    ])

    # Highlight genes of interest
    if genes_highlight:
        for g in genes_highlight:
            row = df_v[df_v['gene_symbol'] == g]
            if len(row) > 0:
                ax.scatter(row['logFC'], row['neglog10_padj'],
                           c='orange', s=80, zorder=5, linewidths=0.5, edgecolors='black')
                ax.annotate(g, (row['logFC'].values[0], row['neglog10_padj'].values[0]),
                            fontsize=9, fontweight='bold', color='darkorange',
                            xytext=(4, 2), textcoords='offset points')

    for _, row in to_label.iterrows():
        g = row['gene_symbol']
        if not genes_highlight or g not in genes_highlight:
            ax.annotate(g, (row['logFC'], row['neglog10_padj']),
                        fontsize=8, fontweight='bold',
                        xytext=(3, 2), textcoords='offset points', alpha=0.9)

    ax.axvline(-LOG2FC_THR, color='navy', linewidth=0.8, linestyle='--')
    ax.axvline(LOG2FC_THR, color='darkred', linewidth=0.8, linestyle='--')
    ax.axhline(-np.log10(PADJ_THR), color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("log₂ Fold Change", fontsize=13, fontweight='bold')
    ax.set_ylabel("-log₁₀(FDR)", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    leg = ax.legend(fontsize=10, frameon=False)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ───────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ───────────────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("PSEUDOBULK EDGER ANALYSIS: Hepatocytes and Malignant cells")
    print("="*80)

    # ========= LOAD DATA =========
    print(f"\n[1/4] Loading {INPUT_H5AD} ...")
    adata = sc.read_h5ad(INPUT_H5AD)
    print(f"      Shape: {adata.shape}")

    # Filter QC-pass cells
    if 'QC_pass' in adata.obs.columns:
        n_before = adata.n_obs
        adata = adata[adata.obs['QC_pass'].astype(bool)].copy()
        print(f"      QC_pass filter: {n_before} → {adata.n_obs} cells")

    # Rename 'Hep' -> 'Hepatocytes'
    adata.obs['CellType_harmonized'] = adata.obs['CellType_harmonized'].replace({'Hep': 'Hepatocytes'})

    # Remove myeloid contamination
    adata, n_removed = remove_myeloid_contamination(adata)
    print(f"      Contaminated hepatocytes removed: {n_removed}")

    # Map DX to condition labels
    dx_map = {
        'adjacent tissue': 'HCC_adj',
        'tumor tissue': 'Tumor',
        'healthy': 'Healthy',
        'Healthy': 'Healthy',
    }
    adata.obs['label'] = adata.obs['DX'].astype(str).map(dx_map).fillna('Unknown')

    # Use counts layer
    if 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()
        print("      Using layer 'counts' (raw UMIs).")
    else:
        print("      WARN: layer 'counts' not found; using X.")

    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()

    # ========= HEPATOCYTE ANALYSIS =========
    print("\n[2/4] HEPATOCYTE ANALYSIS: Tumor vs. Healthy")
    print("       (Myeloid contamination removed)")

    adata_hep = adata[
        (adata.obs['CellType_harmonized'] == 'Hepatocytes') &
        (adata.obs['label'].isin(['Tumor', 'Healthy']))
    ].copy()

    print(f"       Subset shape: {adata_hep.shape}")
    print(f"       Conditions: {dict(adata_hep.obs['label'].value_counts())}")

    counts_hep, meta_hep = aggregate_and_filter(adata_hep)
    print(f"       Pseudobulk: {counts_hep.shape[1]} samples, {counts_hep.shape[0]} genes")

    tt_hep = run_edgeR(counts_hep, meta_hep, contrast_str="Tumor - Healthy")
    tt_hep = tt_hep.rename(columns={
        'logFC': 'log2fc',
        'PValue': 'pval',
        'FDR': 'padj',
    })
    tt_hep['cell_type'] = 'Hepatocytes'

    # Save hepatocyte results
    tt_hep[['gene_symbol', 'pval', 'padj', 'log2fc', 'cell_type']].to_csv(HEP_OUT_CSV, index=False)
    tt_hep.to_csv(HEP_OUT_CSV_FULL, index=True)
    print(f"       Saved: {HEP_OUT_CSV}")

    # Plot hepatocyte figures
    plot_barplot(tt_hep, os.path.join(FIGURES_DIR, 'C1_BarPlot_Hepatocytes_CLEAN.png'),
                 title="Hepatocyte DEGs during HCC (corrected — myeloid contamination removed)")
    plot_volcano(tt_hep, os.path.join(FIGURES_DIR, 'C1_Volcano_Hepatocytes_CLEAN.png'),
                 title="Hepatocyte DE — corrected analysis (C1)",
                 genes_highlight=['MARCO', 'LILRA5', 'RAC2', 'ADGRE5'])
    print(f"       Figures saved to {FIGURES_DIR}")

    # ========= MALIGNANT CELL ANALYSIS =========
    print("\n[3/4] MALIGNANT CELL ANALYSIS: Malignant vs. Healthy Hepatocytes")
    print("       (Myeloid contamination removed from Hepatocytes reference)")

    adata_mc = adata[
        (adata.obs['CellType_harmonized'].isin(['Malignant_cells', 'Hepatocytes'])) &
        (adata.obs['label'].isin(['Tumor', 'Healthy']))
    ].copy()

    # Exclude Healthy Hepatocytes that are contaminated (reclassified as 'Myeloid_contaminant')
    # Only keep Hepatocytes and Malignant_cells
    adata_mc = adata_mc[
        adata_mc.obs['CellType_harmonized'].isin(['Malignant_cells', 'Hepatocytes'])
    ].copy()

    print(f"       Subset shape: {adata_mc.shape}")
    print(f"       Cell types: {dict(adata_mc.obs['CellType_harmonized'].value_counts())}")
    print(f"       Conditions: {dict(adata_mc.obs['label'].value_counts())}")

    counts_mc, meta_mc = aggregate_and_filter(adata_mc)
    print(f"       Pseudobulk: {counts_mc.shape[1]} samples, {counts_mc.shape[0]} genes")

    tt_mc = run_edgeR(counts_mc, meta_mc, contrast_str="Tumor - Healthy")
    tt_mc = tt_mc.rename(columns={
        'logFC': 'log2fc',
        'PValue': 'pval',
        'FDR': 'padj',
    })
    tt_mc['cell_type'] = 'Malignant_cells'

    # Save malignant results
    tt_mc[['gene_symbol', 'pval', 'padj', 'log2fc', 'cell_type']].to_csv(MC_OUT_CSV, index=False)
    print(f"       Saved: {MC_OUT_CSV}")

    # ========= SUMMARY =========
    print("\n[4/4] SUMMARY")
    print(f"      ✓ Hepatocyte DEGs: {HEP_OUT_CSV}")
    print(f"      ✓ Malignant cell DEGs: {MC_OUT_CSV}")
    print(f"      ✓ Figures: {FIGURES_DIR}")
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
