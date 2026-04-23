#!/usr/bin/env python3
"""
CSC (Cancer Stem Cell) Subpopulation Analysis in HCC Malignant Cells
=====================================================================
Opción C - L2 reviewer comment resolution

Literature basis:
- Lin et al. (2024) Journal of Cancer 15(4):1093
- Ding et al. (2025) Translational Cancer Research 14(10):6803-6813

Author: Miguel Ángel Díaz-Campos (INMEGEN)
Date: 2026-04-19
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
INPUT_H5AD = os.getenv('INPUT_H5AD', '/home/mdiaz/HCC_project/MERGED_adata/scvi_integrated.h5ad')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/home/mdiaz/manuscript_revision/new_figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=300, facecolor='white', frameon=False)

print("\n" + "="*65)
print("CSC SUBPOPULATION ANALYSIS — HCC Malignant Cells")
print("="*65)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading integrated dataset...")
adata = sc.read_h5ad(DATA_PATH)
print(f"    Full dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"    Columns in obs: {list(adata.obs.columns[:10])}")

# Identify the cell type column
ct_col = 'CellType_harmonized' if 'CellType_harmonized' in adata.obs.columns else \
         'CellType' if 'CellType' in adata.obs.columns else \
         [c for c in adata.obs.columns if 'cell' in c.lower() and 'type' in c.lower()][0]
print(f"    Using cell type column: '{ct_col}'")
print(f"    Cell types found: {sorted(adata.obs[ct_col].unique())}")

# Subset to Malignant cells
malignant_mask = adata.obs[ct_col].str.lower().str.contains('malignan', na=False)
adata_mal = adata[malignant_mask].copy()
print(f"\n    Malignant cells: {adata_mal.n_obs:,}")

# ─────────────────────────────────────────────────────────────────────────────
# POINT 1 — CSC SIGNATURE SCORING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] POINT 1 — Defining and scoring CSC signature...")

# Full CSC gene signature (combined from literature)
csc_genes_all = {
    # Canonical HCC CSC markers (Lin et al. 2024, Ding et al. 2025)
    'canonical':  ['CD44', 'EPCAM', 'PROM1', 'ALDH1A1', 'CD24', 'CD13'],
    # Ding et al. 2025 — 17-gene CNV-based CSC signature
    'ding2025':   ['HAMP', 'GPC3', 'DNAJC6', 'NT5DC2', 'UBD', 'ATAD2',
                   'LAMC1', 'GABRE', 'LRRC1', 'MUC13', 'STK39', 'SDS',
                   'PPP1R1A', 'TRIM22', 'FGFR2', 'SPINK1', 'IGF2BP2'],
    # Lin et al. 2024 — developmental / prognostic genes
    'lin2024':    ['HSPB1', 'ADH4', 'FTH1', 'APCS'],
    # Stemness pathways
    'stemness':   ['MYC', 'CCND1', 'AXIN2', 'GLI1', 'GLI2', 'PTCH1',
                   'NANOG', 'SOX2', 'KLF4'],
}

# Filter to genes present in dataset
var_names = set(adata_mal.var_names)

gene_groups_present = {}
for grp, genes in csc_genes_all.items():
    present = [g for g in genes if g in var_names]
    missing = [g for g in genes if g not in var_names]
    gene_groups_present[grp] = present
    print(f"    {grp}: {len(present)}/{len(genes)} present | missing: {missing}")

# Flat list for combined signature
csc_signature = []
for genes in gene_groups_present.values():
    csc_signature.extend(genes)
csc_signature = list(set(csc_signature))
print(f"\n    Combined CSC signature: {len(csc_signature)} genes")
print(f"    Genes: {csc_signature}")

# Score cells
sc.tl.score_genes(adata_mal, csc_signature, score_name='CSC_score', use_raw=False)
print(f"    CSC_score range: [{adata_mal.obs['CSC_score'].min():.3f}, {adata_mal.obs['CSC_score'].max():.3f}]")

# Individual signature scores for granularity
for grp, genes in gene_groups_present.items():
    if len(genes) >= 2:
        sc.tl.score_genes(adata_mal, genes, score_name=f'CSC_{grp}_score', use_raw=False)
        print(f"    CSC_{grp}_score computed")

# Classify CSC-high / CSC-low (top 30% = CSC-high)
threshold = np.percentile(adata_mal.obs['CSC_score'], 70)
adata_mal.obs['CSC_class'] = pd.Categorical(
    np.where(adata_mal.obs['CSC_score'] >= threshold, 'CSC-high', 'CSC-low'),
    categories=['CSC-high', 'CSC-low']
)
n_high = (adata_mal.obs['CSC_class'] == 'CSC-high').sum()
n_low  = (adata_mal.obs['CSC_class'] == 'CSC-low').sum()
print(f"\n    CSC-high cells: {n_high:,} ({100*n_high/adata_mal.n_obs:.1f}%)")
print(f"    CSC-low cells:  {n_low:,} ({100*n_low/adata_mal.n_obs:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# POINT 2 — SUBCLUSTERING + UMAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] POINT 2 — Subclustering malignant cells...")

# Recompute neighborhood graph on malignant cells only
if 'X_scVI' in adata_mal.obsm:
    sc.pp.neighbors(adata_mal, use_rep='X_scVI', n_neighbors=15)
    print("    Using scVI latent space for neighbors")
elif 'X_pca' in adata_mal.obsm:
    sc.pp.neighbors(adata_mal, use_rep='X_pca', n_neighbors=15)
    print("    Using PCA for neighbors")
else:
    sc.pp.neighbors(adata_mal, n_neighbors=15)
    print("    Using default for neighbors")

sc.tl.umap(adata_mal, min_dist=0.3)
sc.tl.leiden(adata_mal, resolution=0.8, key_added='leiden_csc')
n_clusters = adata_mal.obs['leiden_csc'].nunique()
print(f"    Leiden clustering: {n_clusters} subclusters at resolution 0.8")

# UMAP Figure — 3 panels: CSC score | CSC class | leiden subcluster
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: CSC score
sc.pl.umap(adata_mal, color='CSC_score', ax=axes[0], show=False,
           color_map='RdBu_r', title='CSC Score', frameon=False)

# Panel B: CSC class
palette_csc = {'CSC-high': '#B2182B', 'CSC-low': '#4393C3'}
sc.pl.umap(adata_mal, color='CSC_class', ax=axes[1], show=False,
           palette=palette_csc, title='CSC Classification', frameon=False,
           legend_loc='on data', legend_fontsize=10)

# Panel C: Leiden subclusters
sc.pl.umap(adata_mal, color='leiden_csc', ax=axes[2], show=False,
           title='Leiden Subclusters (res=0.8)', frameon=False,
           legend_loc='on data', legend_fontsize=9)

plt.suptitle('CSC Subpopulation Analysis — Malignant Cells\n'
             'HCC scRNA-seq (scVI-integrated, 159,925 cells)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}Fig_CSC_UMAP.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: Fig_CSC_UMAP.png")

# ─────────────────────────────────────────────────────────────────────────────
# POINT 3 — DIFFERENTIAL EXPRESSION CSC-HIGH vs CSC-LOW
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] POINT 3 — DEG analysis CSC-high vs CSC-low...")

sc.tl.rank_genes_groups(
    adata_mal, groupby='CSC_class',
    groups=['CSC-high'], reference='CSC-low',
    method='wilcoxon', use_raw=False
)

# Export to CSV
degs_df = sc.get.rank_genes_groups_df(adata_mal, group='CSC-high')
degs_df_top20 = degs_df.head(20)
csv_path = f'{SCRIPT_DIR}fig4_degs_data/csc_degs_top20.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
degs_df.to_csv(csv_path, index=False)
print(f"    ✓ Saved DEGs: {csv_path}")
print(f"    Top 5 CSC-high markers: {list(degs_df_top20['names'].head())}")

# Dotplot top CSC markers
top_markers = list(degs_df_top20['names'].head(12))
# Filter to genes actually in the dataset
top_markers = [g for g in top_markers if g in adata_mal.var_names][:10]

if top_markers:
    fig, ax = plt.subplots(figsize=(10, 5))
    sc.pl.dotplot(
        adata_mal, var_names=top_markers,
        groupby='CSC_class',
        ax=ax, show=False,
        color_map='RdBu_r',
        title='Top CSC-high Markers vs CSC-low',
        standard_scale='var'
    )
    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}Fig_CSC_dotplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: Fig_CSC_dotplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# POINT 4 — METABOLIC + IMMUNE PROFILING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] POINT 4 — Metabolic and immune profiling...")

gene_panels = {
    'OXPHOS\n(Metabolic)':  ['NDUFA1', 'COX7C', 'ATP5F1A', 'SDHA', 'FH'],
    'Immune\nSilencing':    ['CD3E', 'GZMK', 'NCAM1', 'HLA-A', 'PDCD1'],
    'Wnt\nStemness':        ['MYC', 'CCND1', 'AXIN2'],
    'Hedgehog\nStemness':   ['GLI1', 'GLI2', 'PTCH1'],
}

# Calculate mean expression per group for each panel
results = {}
for panel_name, genes in gene_panels.items():
    present = [g for g in genes if g in adata_mal.var_names]
    if not present:
        print(f"    WARNING: No genes from '{panel_name}' found in dataset")
        continue

    means = {}
    for grp in ['CSC-high', 'CSC-low']:
        mask = adata_mal.obs['CSC_class'] == grp
        cells = adata_mal[mask]
        # Get expression values
        if hasattr(cells.X, 'toarray'):
            expr = cells[:, present].X.toarray()
        else:
            expr = cells[:, present].X
        means[grp] = np.mean(expr, axis=0)

    results[panel_name] = {
        'genes': present,
        'CSC-high': means['CSC-high'],
        'CSC-low': means['CSC-low'],
    }

# Build heatmap: log2FC (CSC-high vs CSC-low)
fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 6))
if len(results) == 1:
    axes = [axes]

for ax, (panel_name, data) in zip(axes, results.items()):
    genes  = data['genes']
    hi_exp = data['CSC-high']
    lo_exp = data['CSC-low']

    # log2FC
    fc = np.log2((hi_exp + 1e-9) / (lo_exp + 1e-9))

    # Heatmap matrix: rows=genes, 2 cols (CSC-high, CSC-low)
    mat = np.column_stack([hi_exp, lo_exp])

    # Normalize per gene
    mat_norm = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-9)

    im = ax.imshow(mat_norm, cmap='RdBu_r', aspect='auto',
                   vmin=-2, vmax=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CSC-high', 'CSC-low'], fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=10)
    ax.set_title(panel_name, fontsize=12, fontweight='bold')

    # Add log2FC annotations
    for i, (g, fc_val) in enumerate(zip(genes, fc)):
        ax.text(2.1, i, f'log2FC={fc_val:+.2f}',
                va='center', ha='left', fontsize=8, color='dimgray')

    plt.colorbar(im, ax=ax, fraction=0.046, label='Scaled\nExpression')

plt.suptitle('Metabolic-Immune-Stemness Profile: CSC-high vs CSC-low\n'
             'Malignant Cells — HCC scRNA-seq',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}Fig_CSC_metabolic_immune.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: Fig_CSC_metabolic_immune.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Generating summary statistics...")

summary = {
    'Total malignant cells': adata_mal.n_obs,
    'CSC-high (top 30%)': n_high,
    'CSC-low (bottom 70%)': n_low,
    'CSC score threshold (70th pct)': round(threshold, 4),
    'CSC score mean (CSC-high)': round(adata_mal.obs.loc[adata_mal.obs['CSC_class']=='CSC-high','CSC_score'].mean(), 4),
    'CSC score mean (CSC-low)': round(adata_mal.obs.loc[adata_mal.obs['CSC_class']=='CSC-low','CSC_score'].mean(), 4),
    'Leiden subclusters': n_clusters,
    'CSC signature genes (total)': len(csc_signature),
    'Top DEG (CSC-high)': degs_df_top20['names'].iloc[0] if len(degs_df_top20) > 0 else 'N/A',
}

print("\n    === ANALYSIS SUMMARY ===")
for k, v in summary.items():
    print(f"    {k}: {v}")

# Save summary
summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
summary_df.to_csv(f'{SCRIPT_DIR}fig4_degs_data/csc_analysis_summary.csv')

print("\n" + "="*65)
print("✅ CSC ANALYSIS COMPLETE")
print(f"   Figures saved to: {OUT_DIR}")
print(f"   Data saved to: {SCRIPT_DIR}fig4_degs_data/")
print("="*65 + "\n")
