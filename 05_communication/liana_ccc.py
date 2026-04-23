#!/usr/bin/env python
# coding: utf-8
"""
05_communication/liana_ccc.py
==============================
LIANA+ consensus cell-cell communication analysis (RRA aggregate of 4 methods:
CellChat, CellPhoneDB, SingleCellSignalR, NATMI) on the SCVI-integrated HCC
AnnData.

Configuration
-------------
All I/O paths are overridden via environment variables:
  INPUT_H5AD  -- path to scvi_integrated.h5ad
  OUTPUT_DIR  -- directory for all output tables and figures
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import liana as li
from liana.method import cellchat, cellphonedb, singlecellsignalr, natmi

try:
    import session_info
except ImportError:
    session_info = None

try:
    from liana.plotting._circle_plot import circle_plot
except ImportError:
    pass

try:
    from plotnine import ggplot, aes, geom_col, coord_flip, theme_bw, labs, theme, element_text
    _HAS_PLOTNINE = True
except ImportError:
    _HAS_PLOTNINE = False

# ----------------------------
# Config block (env-parametrized)
# ----------------------------
INPUT_H5AD = os.getenv(
    "INPUT_H5AD",
    "/home/mdiaz/HCC_project/MERGED_adata/scvi_integrated.h5ad",
)
OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "/home/mdiaz/HCC_project/HCC_analysis/MERGED_analysis/liana_consensus",
)

SOURCE_LABELS_REQ = ['Malignant cells', 'TAMs', 'TECs', 'Hepatocytes']
TARGET_LABELS_REQ = [
    'B cells', 'Basophil', 'CAF', 'Hepatocytes', 'Macrophages',
    'Malignant cells', 'Cholangiocytes', 'Endothelial cells',
    'NK-TR-CD160', 'Neutrophils', 'T cells', 'TAMs', 'TECs', 'CDCs', 'Fibroblasts',
]

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")


def _normalize_log1p(adata: sc.AnnData) -> sc.AnnData:
    """
    Build a CP10k + log1p normalized layer ('log1p') from the counts layer.

    Operates in-place on adata (adds adata.layers['log1p']).
    """
    X_counts = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    if sp.issparse(X_counts):
        Xc = X_counts.tocsr(copy=False)
        lib = np.asarray(Xc.sum(axis=1)).ravel()
        scale_vec = (1e4 / np.clip(lib, 1, None)).astype(np.float32)
        norm = Xc.multiply(scale_vec[:, None]).tocsr(copy=False).astype(np.float32)
        norm.data = np.log1p(norm.data)
        adata.layers['log1p'] = norm
    else:
        lib = X_counts.sum(axis=1)
        norm = X_counts * ((1e4 / np.clip(lib, 1, None))[:, None])
        adata.layers['log1p'] = np.log1p(norm).astype(np.float32)
    return adata


def _split_complex(x: str):
    """Split a complex gene name (e.g. 'B2M_FCGRT') into its subunits."""
    x = str(x)
    for sep in ['_', '+', '&']:
        if sep in x:
            return [p for p in x.split(sep) if p]
    return [x]


def _props_by_group(adata: sc.AnnData, genes, groupby: str,
                    layer_counts: str = 'counts', expr_cutoff: float = 0):
    """
    Compute per-group expression proportions (fraction of cells with expr > cutoff).

    Parameters
    ----------
    adata        : AnnData with a counts layer.
    genes        : list of gene names.
    groupby      : obs column to group by.
    layer_counts : layer name for raw counts.
    expr_cutoff  : threshold above which a cell is considered expressing.

    Returns
    -------
    pd.DataFrame with columns [groupby, 'gene', 'prop'].
    """
    X, is_sp = (adata.layers[layer_counts], True) if layer_counts in getattr(adata, 'layers', {}) else (adata.X, sp.issparse(adata.X))
    obs_group = adata.obs[groupby].astype(str).values
    varnames  = adata.var_names.astype(str)
    idx_map   = {g: np.where(varnames == g)[0][0] for g in genes if g in varnames.values}

    rows = []
    for ct in np.unique(obs_group):
        I = np.where(obs_group == ct)[0]
        if I.size == 0 or len(idx_map) == 0:
            continue
        cols = list(idx_map.values())
        if is_sp:
            sub  = X[I, :][:, cols].tocsr()
            nnz  = (sub > expr_cutoff).sum(axis=0).A1
        else:
            sub  = X[np.ix_(I, cols)]
            nnz  = (sub > expr_cutoff).sum(axis=0)
        prop = nnz.astype(float) / float(I.size)
        rows.extend([(ct, g, p) for g, p in zip(idx_map.keys(), prop)])
    return pd.DataFrame(rows, columns=[groupby, 'gene', 'prop'])


def _annotate_props(res: pd.DataFrame, props_df: pd.DataFrame,
                    side_col: str, ct_col: str, out_col: str,
                    groupby: str) -> pd.DataFrame:
    """
    Add ligand_props or receptor_props column to the liana result dataframe.

    Handles gene complexes by taking the minimum proportion across subunits.
    """
    def _complex_prop(cell_type, comp):
        subs   = _split_complex(comp)
        sub_df = props_df[(props_df[groupby] == cell_type) & (props_df.gene.isin(subs))]
        if sub_df.shape[0] < len(subs):
            return 0.0
        return float(sub_df['prop'].min())

    res[out_col] = pd.to_numeric([
        _complex_prop(ct, comp)
        for ct, comp in zip(res[ct_col].astype(str), res[side_col].astype(str))
    ])
    return res


def run_liana_rra(adata: sc.AnnData) -> sc.AnnData:
    """
    Run the LIANA RRA consensus (4 methods) on a restricted set of source/target
    cell-type pairs derived from SOURCE_LABELS_REQ and TARGET_LABELS_REQ.

    Results are stored in adata.uns['liana_res'].
    """
    obs_labels  = set(adata.obs['CellType_harmonized'].astype(str).unique())
    src = [s for s in SOURCE_LABELS_REQ if s in obs_labels]
    tgt = [t for t in TARGET_LABELS_REQ if t in obs_labels]

    if not src or not tgt:
        print("[WARN] Some requested labels are absent; using only present ones.")
    groupby_pairs = pd.DataFrame(
        [(s, t) for s in src for t in tgt], columns=['source', 'target']
    )

    custom_rra = li.mt.AggregateClass(li.mt.aggregate_meta,
                                      methods=[cellchat, cellphonedb, singlecellsignalr, natmi])
    np.random.seed(0)
    custom_rra(
        adata=adata,
        groupby='CellType_harmonized',
        resource_name='consensus',
        expr_prop=0.1,
        groupby_pairs=groupby_pairs if not groupby_pairs.empty else None,
        n_perms=1000,
        n_jobs=1,
        seed=1337,
        use_raw=False,
        layer='log1p',
        verbose=True,
        key_added='liana_res',
    )
    return adata


def run_liana_rra_all(adata: sc.AnnData) -> sc.AnnData:
    """
    Run LIANA RRA without source/target restriction (all vs all).

    Results are stored in adata.uns['liana_res'].
    """
    custom_rra = li.mt.AggregateClass(li.mt.aggregate_meta,
                                      methods=[cellchat, cellphonedb, singlecellsignalr, natmi])
    np.random.seed(1337)
    custom_rra(
        adata=adata,
        groupby='CellType_harmonized',
        resource_name='consensus',
        expr_prop=0.1,
        groupby_pairs=None,
        n_perms=1000,
        n_jobs=1,
        seed=1337,
        use_raw=False,
        layer=None,
        verbose=True,
        key_added='liana_res',
    )
    return adata


def add_expression_props(adata: sc.AnnData,
                         groupby: str = 'CellType_harmonized',
                         layer_for_props: str = 'counts') -> sc.AnnData:
    """
    Compute and attach ligand_props / receptor_props to adata.uns['liana_res'].

    Parameters
    ----------
    adata          : AnnData with liana_res in uns.
    groupby        : obs column used during LIANA run.
    layer_for_props: layer to use for computing expression proportions.
    """
    res = adata.uns['liana_res'].copy()

    L = 'ligand_complex'   if 'ligand_complex'   in res.columns else 'ligand'
    R = 'receptor_complex' if 'receptor_complex' in res.columns else 'receptor'

    lig_subunits = sorted({s for v in res[L].astype(str).unique() for s in _split_complex(v)})
    rec_subunits = sorted({s for v in res[R].astype(str).unique() for s in _split_complex(v)})
    genes_needed = sorted(set(lig_subunits) | set(rec_subunits))

    props_df = _props_by_group(adata, genes_needed, groupby=groupby,
                               layer_counts=layer_for_props, expr_cutoff=0)

    res = _annotate_props(res, props_df, L, 'source', 'ligand_props',   groupby)
    res = _annotate_props(res, props_df, R, 'target', 'receptor_props', groupby)

    adata.uns['liana_res'] = res
    return adata


def filter_and_score(adata: sc.AnnData,
                     prop_cutoff: float = 0.10) -> pd.DataFrame:
    """
    Filter liana_res by expression proportions and compute composite scores.

    Parameters
    ----------
    adata       : AnnData with liana_res in uns.
    prop_cutoff : minimum ligand and receptor expression proportion.

    Returns
    -------
    Filtered DataFrame with consensus_mag and consensus_spec columns added.
    """
    res = adata.uns['liana_res'].copy()
    for c in ['magnitude_rank', 'specificity_rank', 'rank']:
        if c in res.columns:
            res[c] = pd.to_numeric(res[c], errors='coerce')

    res_f = res[(res['ligand_props'] >= prop_cutoff) &
                (res['receptor_props'] >= prop_cutoff)].copy()

    if 'magnitude_rank' not in res_f.columns:
        res_f['magnitude_rank'] = res_f['rank']
    if 'specificity_rank' not in res_f.columns:
        res_f['specificity_rank'] = res_f['rank']

    res_f['consensus_mag']  = 1.0 - res_f['magnitude_rank']
    res_f['consensus_spec'] = 1.0 - res_f['specificity_rank']

    adata.uns['liana_res'] = res_f
    print(f"[filter_and_score] Rows after filtering: {res_f.shape[0]}")
    return res_f


def run_qa(adata: sc.AnnData) -> dict:
    """
    Validate liana_res columns and write a QA summary JSON.

    Raises AssertionError if required columns are missing or ranks are out of [0,1].

    Returns
    -------
    dict with summary statistics.
    """
    df = adata.uns['liana_res'].copy()

    need_cols = ['source', 'target', 'ligand_props', 'receptor_props',
                 'magnitude_rank', 'specificity_rank']
    missing   = [c for c in need_cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    assert any(c in df.columns for c in ['ligand', 'ligand_complex']), \
        "Missing ligand/ligand_complex"
    assert any(c in df.columns for c in ['receptor', 'receptor_complex']), \
        "Missing receptor/receptor_complex"

    def _check_01(col):
        v  = pd.to_numeric(df[col], errors='coerce').dropna()
        mn, mx = float(v.min()), float(v.max())
        assert 0.0 - 1e-9 <= mn and mx <= 1.0 + 1e-9, \
            f"{col} out of [0,1]: min={mn}, max={mx}"

    for col in ['ligand_props', 'receptor_props', 'magnitude_rank', 'specificity_rank']:
        _check_01(col)

    summary = {
        'rows':                    int(df.shape[0]),
        'magnitude_rank_min':      float(pd.to_numeric(df['magnitude_rank']).min()),
        'magnitude_rank_max':      float(pd.to_numeric(df['magnitude_rank']).max()),
        'specificity_rank_min':    float(pd.to_numeric(df['specificity_rank']).min()),
        'specificity_rank_max':    float(pd.to_numeric(df['specificity_rank']).max()),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "QA_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("QA OK"); print(summary)
    return summary


def plot_heatmaps(adata: sc.AnnData, df: pd.DataFrame) -> None:
    """
    Save interaction-count and weighted-sum heatmaps (source x target).

    Parameters
    ----------
    adata : AnnData with CellType_harmonized.
    df    : filtered liana_res DataFrame with 'score' and 'spec' columns.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    ct_all = (
        list(adata.obs['CellType_harmonized'].cat.categories.astype(str))
        if pd.api.types.is_categorical_dtype(adata.obs['CellType_harmonized'])
        else sorted(adata.obs['CellType_harmonized'].astype(str).unique())
    )

    df2 = df.copy()
    if 'score' not in df2.columns:
        df2['score'] = 1 - pd.to_numeric(df2['magnitude_rank'], errors='coerce')
    if 'spec' not in df2.columns:
        df2['spec']  = 1 - pd.to_numeric(df2['specificity_rank'], errors='coerce')
    df2['w'] = df2['score'] * df2['spec']

    mat_counts = (df2.groupby(['source', 'target']).size()
                  .unstack(fill_value=0)
                  .reindex(index=ct_all, columns=ct_all, fill_value=0))
    mat_sum = (df2.groupby(['source', 'target'])['w'].sum()
               .unstack(fill_value=0)
               .reindex(index=ct_all, columns=ct_all, fill_value=0))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    im0 = ax[0].imshow(mat_counts.values, aspect='auto')
    ax[0].set_title('Number of interactions')
    ax[0].set_yticks(range(mat_counts.shape[0]))
    ax[0].set_yticklabels(mat_counts.index, fontsize=8)
    ax[0].set_xticks(range(mat_counts.shape[1]))
    ax[0].set_xticklabels(mat_counts.columns, rotation=90, fontsize=8)
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(mat_sum.values, aspect='auto')
    ax[1].set_title('Magnitude x Specificity rank')
    ax[1].set_yticks(range(mat_sum.shape[0]))
    ax[1].set_yticklabels(mat_sum.index, fontsize=8)
    ax[1].set_xticks(range(mat_sum.shape[1]))
    ax[1].set_xticklabels(mat_sum.columns, rotation=90, fontsize=8)
    plt.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, "liana_heatmap_counts_and_scores.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot_heatmaps] Saved: {out_png}")


def plot_top_lr_bars(df: pd.DataFrame) -> None:
    """
    Save top L-R bar charts (composite score) per focus source cell type.

    Requires plotnine; skips gracefully if unavailable.
    """
    if not _HAS_PLOTNINE:
        print("[plot_top_lr_bars] plotnine not available; skipping.")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    L = 'ligand_complex'   if 'ligand_complex'   in df.columns else 'ligand'
    R = 'receptor_complex' if 'receptor_complex' in df.columns else 'receptor'

    df2 = df.copy()
    df2['composite'] = ((1 - pd.to_numeric(df2['magnitude_rank'], errors='coerce')) *
                        (1 - pd.to_numeric(df2['specificity_rank'], errors='coerce')))

    def top_lr_by_source(d, source, k=15):
        """Return top-k L-R pairs for a given source cell type."""
        d = (d[d['source'] == source].copy()
             .sort_values('composite', ascending=False))
        d['pair'] = d[L] + ' -> ' + d[R]
        d = d.groupby('pair', as_index=False)['composite'].max().head(k)
        return d

    for s in ['Hepatocytes', 'Malignant cells', 'TAMs', 'TECs']:
        d = top_lr_by_source(df2, s, k=15)
        if d.empty:
            continue
        p = (ggplot(d, aes('reorder(pair, composite)', 'composite'))
             + geom_col() + coord_flip() + theme_bw()
             + labs(title=f"Top L-R from {s}",
                    x="Ligand -> Receptor", y="Composite (score x spec)")
             + theme(axis_text_y=element_text(size=8)))
        out_pdf = os.path.join(FIGURES_DIR, f"Top_LR_{s.replace(' ', '_')}.pdf")
        p.save(filename=out_pdf, format="pdf", width=8, height=6, units="in", dpi=300)
        print(f"[plot_top_lr_bars] Saved: {out_pdf}")


def plot_dotplot(adata: sc.AnnData, df: pd.DataFrame) -> None:
    """
    Save a LIANA dotplot for the top-10 interactions from focus sources.

    Parameters
    ----------
    adata : AnnData with liana_res in uns.
    df    : filtered liana_res DataFrame.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    L = 'ligand_complex'   if 'ligand_complex'   in df.columns else 'ligand'
    R = 'receptor_complex' if 'receptor_complex' in df.columns else 'receptor'

    sources_focus = ['Malignant cells', 'TAMs', 'TECs', 'Hepatocytes']
    top_k         = 10

    tops = []
    for s in sources_focus:
        sub = df[df['source'] == s].sort_values(
            ['magnitude_rank', 'specificity_rank'], ascending=[True, True]
        ).head(top_k)
        if not sub.empty:
            tops.append(sub[['source', 'target', L, R]])
    tops     = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame(columns=['source', 'target', L, R])
    top_keys = set(map(tuple, tops[['source', 'target', L, R]].values))

    def flt_top(r):
        return (r['source'], r['target'], r.get(L, None), r.get(R, None)) in top_keys

    present_src = [s for s in sources_focus if s in df['source'].astype(str).unique()]
    present_tgt = sorted(df['target'].astype(str).unique())

    p = li.pl.dotplot(
        adata=adata,
        colour='magnitude_rank',
        size='specificity_rank',
        inverse_colour=True,
        inverse_size=True,
        source_labels=present_src,
        target_labels=present_tgt,
        figure_size=(10, 12),
        top_n=None,
        orderby=None,
        filter_fun=flt_top,
        uns_key='liana_res',
    )
    out_png = os.path.join(FIGURES_DIR, "bubble_topK_sources_dotplot.png")
    p.save(out_png, dpi=300, width=11, height=7)
    print(f"[plot_dotplot] Saved: {out_png}")


def main():
    """Run the full LIANA CCC pipeline: load, normalize, RRA, props, QA, plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1) Load
    print(f"[MAIN] Loading: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)
    print(f"       Shape: {adata.shape}")

    # 2) Set raw counts from counts layer
    adata.raw = ad.AnnData(
        X=adata.layers["counts"].copy(),
        var=adata.var.copy(),
        obs=adata.obs.copy(),
    )

    # 3) Label fixes for LIANA compatibility
    adata.obs["CellType_harmonized"] = (
        adata.obs["CellType_harmonized"]
        .replace({"cDCs": "CDCs", "Malignant_cells": "Malignant cells", "Hep": "Hepatocytes"})
    )

    # 4) Normalize
    print("[MAIN] Building log1p layer ...")
    adata = _normalize_log1p(adata)

    # 5) Run restricted RRA
    print("[MAIN] Running LIANA RRA (restricted pairs) ...")
    adata = run_liana_rra(adata)

    # 6) Add expression proportions
    print("[MAIN] Computing expression proportions ...")
    adata = add_expression_props(adata, groupby='CellType_harmonized',
                                 layer_for_props='counts')

    # 7) Filter and score
    df = filter_and_score(adata, prop_cutoff=0.10)

    # 8) QA
    run_qa(adata)

    # 9) Figures
    print("[MAIN] Generating figures ...")
    plot_heatmaps(adata, df)
    plot_top_lr_bars(df)
    plot_dotplot(adata, df)

    print(f"\n[OK] LIANA CCC analysis complete.")
    print(f"     Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
