#!/usr/bin/env python3
"""
Extract DEGs for Figure 4 using scanpy.rank_genes_groups
Exports top genes by celltype and condition (no visualization, just data)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import scanpy as sc
import numpy as np
from pathlib import Path

# Load data
print("[1/4] Loading scvi_integrated.h5ad...")
adata = sc.read_h5ad("/home/mdiaz/HCC_project/MERGED_adata/scvi_integrated.h5ad")
print(f"      Shape: {adata.n_obs} cells × {adata.n_vars} genes")

# Filter QC-pass and identify condition column
if 'QC_pass' in adata.obs.columns:
    adata = adata[adata.obs['QC_pass'].astype(bool)].copy()

# Infer condition column
condition_col = None
for col in ['Condition', 'condition', 'source', 'Source']:
    if col in adata.obs.columns:
        condition_col = col
        break

print(f"      Using condition column: {condition_col}")
print(f"      Conditions: {adata.obs[condition_col].unique()}")
print(f"      Cell types: {adata.obs['CellType'].unique()}")

# Cell types to analyze (use CellType_harmonized)
cell_types = ['Hepatocytes', 'Malignant_cells', 'TAMs', 'TECs']
celltype_col = 'CellType_harmonized'

# Output directory
out_dir = Path("/home/mdiaz/manuscript_revision/manuscript_code/code_not_in_github/merged_analysis/fig4_degs_data")
out_dir.mkdir(parents=True, exist_ok=True)

# For each celltype, extract top genes in each condition
print("\n[2/4] Computing rank_genes_groups by condition...")

degs_data = {}

for ct in cell_types:
    print(f"\n  Processing {ct}...")

    # Subset to this celltype
    adata_ct = adata[adata.obs[celltype_col] == ct].copy()

    if adata_ct.n_obs < 10:
        print(f"    Skipping {ct}: too few cells ({adata_ct.n_obs})")
        continue

    # Rank genes by condition
    try:
        # Use log1p layer if available, otherwise raw counts
        sc.tl.rank_genes_groups(
            adata_ct,
            groupby=condition_col,
            method='wilcoxon',
            n_genes=100,
            use_raw=False
        )

        # Extract results for each condition
        result = adata_ct.uns['rank_genes_groups']
        conditions = result['names'].dtype.names

        for cond in conditions:
            key = f"{ct}_{cond}"
            genes_cond = result['names'][cond][:20]  # Top 20

            degs_data[key] = {
                'celltype': ct,
                'condition': cond,
                'genes': list(genes_cond)
            }
            print(f"    {cond}: extracted {len(genes_cond)} top genes")

    except Exception as e:
        print(f"    Error processing {ct}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Save DEGs to CSV format
print("\n[3/4] Exporting DEG data...")

for key, data in degs_data.items():
    df = pd.DataFrame({
        'gene': data['genes'],
        'celltype': data['celltype'],
        'condition': data['condition']
    })

    filename = out_dir / f"degs_{key}.csv"
    df.to_csv(filename, index=False)
    print(f"  ✓ {filename}")

# Also export metabolic and immune marker gene lists
print("\n[4/4] Exporting functional gene lists...")

metabolic_genes = {
    'OXPHOS_Complex_I': ['NDUFA1', 'NDUFB5', 'NDUFC2'],
    'OXPHOS_Complex_III': ['UQCR10', 'UQCRQ'],
    'OXPHOS_Complex_IV': ['COX7C', 'COX6C'],
    'OXPHOS_Complex_V': ['ATP5F1A', 'ATP5F1B', 'ATP5F1C'],
    'OXPHOS_other': ['SDHA', 'FH', 'MDH2']
}

immune_silencing_genes = {
    'T_cell_markers': ['CD3D', 'CD3E', 'CD8A', 'TRAC'],
    'Cytotoxic': ['GZMK', 'GZMB', 'PRF1'],
    'NK_markers': ['NCAM1', 'NKG7', 'GNLY'],
    'MHC_I': ['HLA-A', 'HLA-B', 'HLA-C', 'B2M'],
    'Co_stimulation': ['ICAM1', 'ITGAM', 'LFA1'],
    'Checkpoint': ['PD1', 'PDCD1', 'LAG3', 'CTLA4']
}

# Save as simple list
mito_df = pd.DataFrame([
    {'category': k, 'gene': g}
    for k, v in metabolic_genes.items()
    for g in v
])
mito_df.to_csv(out_dir / "metabolic_genes.csv", index=False)
print(f"  ✓ {out_dir / 'metabolic_genes.csv'}")

immune_df = pd.DataFrame([
    {'category': k, 'gene': g}
    for k, v in immune_silencing_genes.items()
    for g in v
])
immune_df.to_csv(out_dir / "immune_genes.csv", index=False)
print(f"  ✓ {out_dir / 'immune_genes.csv'}")

print("\n✅ Done! All DEG data exported.")
print(f"   Output directory: {out_dir}")
