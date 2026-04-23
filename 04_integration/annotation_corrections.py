#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_integration/annotation_corrections.py
==========================================
Applies four sequential annotation-correction passes to the SCVI-integrated
AnnData, then saves a single corrected h5ad and all associated figures.

Pass order (each corresponds to an original script):
  1. C2 marker reannotation      -- winner-takes-all gene-score override
  2. Fig3 panels fixes           -- "Hep" rename + Healthy Neutrophil rescue
  3. Reclassify and Fig3D        -- Basophil/Unclassified/pDC reclassification
  4. GSM4648565 hepatocyte fix   -- NPC sample mislabel correction

Configuration
-------------
Paths are overridden via environment variables:
  INPUT_H5AD  -- path to scvi_integrated.h5ad (read and written in-place)
  OUTPUT_DIR  -- directory for all figures and CSV outputs
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorcet as cc
from scipy.sparse import issparse

sc.settings.verbosity = 0

# ----------------------------
# Config block (env-parametrized)
# ----------------------------
INPUT_H5AD = os.getenv(
    "INPUT_H5AD",
    "/home/mdiaz/HCC_project/MERGED_adata/scvi_integrated.h5ad",
)
OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "/home/mdiaz/manuscript_revision/new_figures",
)

# ----------------------------
# Global matplotlib style
# ----------------------------
mpl.rcParams.update({
    "font.weight":           "bold",
    "axes.labelweight":      "bold",
    "axes.titleweight":      "bold",
    "axes.labelsize":        13,
    "axes.titlesize":        14,
    "xtick.labelsize":       11,
    "ytick.labelsize":       11,
    "legend.fontsize":       9,
    "legend.title_fontsize": 10,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
})

# ----------------------------
# Marker sets for C2 scoring
# ----------------------------
MARKERS = {
    # Liver parenchyma
    "Hepatocytes":       ["ALB", "APOA1", "APOB", "TTR", "FGA",
                          "FABP1", "PCK1", "CYP3A4", "SERPINA1"],
    "Cholangiocytes":    ["KRT7", "KRT19", "SOX9", "ANXA4", "TFF1", "CFTR"],
    # Macrophages
    "Macrophages":       ["MARCO", "CLEC4F", "TIMD4", "VSIG4", "LYVE1", "CD68"],
    "TAMs":              ["TREM2", "SPP1", "GPNMB", "CD9", "LGMN", "MRC1"],
    "Monocytes":         ["CD14", "LYZ", "VCAN", "FCGR3A", "S100A8"],
    # Lymphocytes
    "NK cells":          ["NKG7", "KLRD1", "GNLY", "GZMB", "NCAM1"],
    "NK-TR-CD160":       ["CD160", "KLRC1", "XCL1", "CXCR6", "ITGA1"],
    "T cells":           ["CD3D", "CD3E", "TRAC", "CD8A", "CD4"],
    "B cells":           ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Plasma cells":      ["JCHAIN", "SDC1", "MZB1", "IGHG1"],
    # Granulocytes / Mast
    "Basophils":         ["GATA2", "PRG2", "CLC", "HDC", "ENPP3", "PRSS33"],
    "Mast cells":        ["TPSAB1", "TPSB2", "KIT", "FCER1A"],
    "Neutrophils":       ["FCGR3B", "CXCR2", "ELANE", "CSF3R", "MPO"],
    # Dendritic cells
    "pDCs":              ["LILRA4", "CLEC4C", "IL3RA", "IRF7"],
    "cDCs":              ["CLEC9A", "XCR1", "CD1C", "BATF3"],
    # Endothelium
    "Endothelial cells": ["PECAM1", "VWF", "CDH5", "CLDN5", "CD34"],
    "TECs":              ["ACKR1", "APLN", "ESM1", "RELN", "SPARCL1"],
    # Stroma
    "Fibroblasts":       ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB"],
    "CAF":               ["FAP", "PDPN", "POSTN", "PDGFRA", "MMP11"],
}

PROTECTED          = {"Malignant_cells"}
THRESHOLD_DEFAULT  = 0.30
THRESHOLD_HEPATO   = 0.50
MARGIN_MIN         = 0.10


# ============================================================
# Shared helpers
# ============================================================

def _build_condition(adata: sc.AnnData) -> sc.AnnData:
    """
    Add 'Condition' column (Healthy Donors / HCC Diseased / Unknown)
    derived from the DX obs column.
    """
    dx = adata.obs["DX"].astype(str).str.strip().str.lower()
    is_diseased = (dx.str.contains(r'\badjacent tissue\b', regex=True) |
                   dx.str.contains(r'\btumou?r tissue\b', regex=True))
    is_healthy  = dx.str.fullmatch(r'healthy')
    adata.obs["Condition"] = np.where(is_diseased, "HCC Diseased",
                              np.where(is_healthy,  "Healthy Donors", "Unknown"))
    return adata


def get_expr(adata: sc.AnnData, gene: str) -> np.ndarray:
    """Return dense 1-D array of expression values for a single gene."""
    if gene not in adata.var_names:
        return np.zeros(adata.n_obs, dtype=np.float32)
    idx = adata.var_names.get_loc(gene)
    x = adata.X[:, idx]
    if issparse(x):
        x = np.asarray(x.todense()).ravel()
    return np.asarray(x, dtype=np.float32)


def _bold_legend(leg) -> None:
    """Set bold font weight on all legend text entries."""
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    if leg.get_title():
        leg.get_title().set_fontweight("bold")


def plot_umap(adata: sc.AnnData, ct_order, ct_color,
              condition_filter, title, out_png) -> None:
    """
    Render a UMAP scatter panel.

    Parameters
    ----------
    adata            : integrated AnnData with X_umap and Condition obs column.
    ct_order         : ordered list of cell-type labels for the legend.
    ct_color         : dict mapping label -> hex color.
    condition_filter : str to filter a single condition, or None for all.
    title            : panel title.
    out_png          : output file path.
    """
    umap_all = adata.obsm["X_umap"]
    if condition_filter:
        mask  = adata.obs["Condition"].values == condition_filter
        x_fg  = umap_all[mask, 0]; y_fg = umap_all[mask, 1]
        ct_fg = adata.obs["CellType_harmonized"].values[mask].astype(str)
        cts   = [c for c in ct_order if c in set(ct_fg)]
    else:
        mask  = np.isin(adata.obs["Condition"].values, ["Healthy Donors", "HCC Diseased"])
        x_fg  = umap_all[mask, 0]; y_fg = umap_all[mask, 1]
        ct_fg = adata.obs["CellType_harmonized"].values[mask].astype(str)
        cts   = [c for c in ct_order if c in set(ct_fg)]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.set_facecolor("white")
    if condition_filter:
        ax.scatter(umap_all[:, 0], umap_all[:, 1],
                   s=0.25, c="#cccccc", linewidths=0, alpha=0.35,
                   rasterized=True, zorder=1)
    for ct in reversed(cts):
        idx = ct_fg == ct
        ax.scatter(x_fg[idx], y_fg[idx], s=1.5,
                   c=ct_color[ct], linewidths=0, alpha=0.85,
                   label=ct, rasterized=True, zorder=2)
    handles = [mpatches.Patch(color=ct_color[c], label=c) for c in cts]
    leg = ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
                    frameon=False, fontsize=8.5, ncol=1,
                    handlelength=1.2, handleheight=1.2,
                    borderpad=0.4, labelspacing=0.5)
    _bold_legend(leg)
    ax.set_xlabel("UMAP 1", fontweight="bold", fontsize=13)
    ax.set_ylabel("UMAP 2", fontweight="bold", fontsize=13)
    ax.set_title(title, fontweight="bold", fontsize=14, pad=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      {out_png}")


def plot_composition(adata: sc.AnnData, ct_order_plot, ct_color, out_png) -> None:
    """
    Render a stacked bar chart of cell-type proportions per condition (2 bars).

    Parameters
    ----------
    adata         : integrated AnnData with Condition and CellType_harmonized.
    ct_order_plot : ordered list of cell types for stacking.
    ct_color      : dict mapping label -> hex color.
    out_png       : output file path.
    """
    sub    = adata[adata.obs["Condition"].isin(["Healthy Donors", "HCC Diseased"])].copy()
    counts = pd.crosstab(sub.obs["Condition"], sub.obs["CellType_harmonized"])
    if "Unclassified" in counts.columns:
        counts = counts.drop(columns=["Unclassified"])
    props  = counts.div(counts.sum(axis=1), axis=0)
    cts_p  = [c for c in ct_order_plot if c in props.columns]
    props  = props[cts_p].reindex(["Healthy Donors", "HCC Diseased"])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("white")
    bottom = np.zeros(len(props))
    x_pos  = np.arange(len(props))
    for ct in cts_p:
        vals = props[ct].values
        ax.bar(x_pos, vals, bottom=bottom,
               label=ct, color=ct_color[ct], width=0.55, edgecolor="none")
        bottom += vals
    ax.set_xticks(x_pos)
    ax.set_xticklabels(props.index, fontweight="bold", fontsize=12)
    ax.set_ylabel("Proportion", fontweight="bold", fontsize=13)
    ax.set_xlabel("Condition",  fontweight="bold", fontsize=13)
    ax.set_title("Cellular Composition by Condition", fontweight="bold", fontsize=14, pad=10)
    ax.set_ylim(0, 1)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles = [mpatches.Patch(color=ct_color[c], label=c) for c in cts_p]
    leg = ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                    frameon=False, fontsize=8.5, ncol=2,
                    handlelength=1.2, handleheight=1.2,
                    borderpad=0.4, labelspacing=0.5, columnspacing=0.8)
    _bold_legend(leg)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      {out_png}")


def plot_composition_3bar(adata: sc.AnnData, ct_order_plot, ct_color, out_png) -> None:
    """
    Render a 3-bar stacked composition chart: Healthy | Adjacent | Tumor.

    Parameters
    ----------
    adata         : integrated AnnData with Condition_detail and CellType_harmonized.
    ct_order_plot : ordered list of cell types for stacking.
    ct_color      : dict mapping label -> hex color.
    out_png       : output file path.
    """
    conditions_3 = ["Healthy Donors", "Adjacent tissue", "Tumor tissue"]
    sub    = adata[adata.obs["Condition_detail"].isin(conditions_3)].copy()
    counts = pd.crosstab(sub.obs["Condition_detail"], sub.obs["CellType_harmonized"])
    if "Unclassified" in counts.columns:
        counts = counts.drop(columns=["Unclassified"])
    props  = counts.div(counts.sum(axis=1), axis=0)
    cts_p  = [c for c in ct_order_plot if c in props.columns]
    props  = props[cts_p].reindex(conditions_3)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.set_facecolor("white")
    bottom = np.zeros(len(props))
    x_pos  = np.arange(len(props))
    for ct in cts_p:
        vals = props[ct].values
        ax.bar(x_pos, vals, bottom=bottom,
               label=ct, color=ct_color[ct], width=0.55, edgecolor="none")
        bottom += vals
    ax.set_xticks(x_pos)
    ax.set_xticklabels(props.index, fontweight="bold", fontsize=11)
    ax.set_ylabel("Proportion", fontweight="bold", fontsize=13)
    ax.set_xlabel("Condition", fontweight="bold", fontsize=13)
    ax.set_title("Cellular Composition by Tissue Type", fontweight="bold", fontsize=14, pad=10)
    ax.set_ylim(0, 1)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles = [mpatches.Patch(color=ct_color[c], label=c) for c in cts_p]
    leg = ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                    frameon=False, fontsize=8.5, ncol=2,
                    handlelength=1.2, handleheight=1.2,
                    borderpad=0.4, labelspacing=0.5, columnspacing=0.8)
    _bold_legend(leg)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      {out_png}")


# ============================================================
# ===== C2: MARKER REANNOTATION =====
# ============================================================

def apply_c2_reannotation(adata: sc.AnnData) -> sc.AnnData:
    """
    Re-annotate all cells via winner-takes-all gene scores over 19 canonical
    cell-type marker sets.

    Strategy
    --------
    - For each cell: compute scanpy score_genes for 19 cell types.
    - Winner-takes-all: assign the type with the highest score.
    - Minimum threshold: top_score > 0.30 to override the SCANVI label.
    - Hepatocyte threshold: 0.50 (markers are highly specific and robust).
    - Margin check: top_score - second_score >= 0.10.
    - Malignant_cells are never overwritten.

    Outputs
    -------
    - adata with updated CellType_harmonized.
    - C2_reannotation_summary.csv in OUTPUT_DIR.
    - Fig3A-D PNG panels in OUTPUT_DIR.
    """
    print("[C2] Starting marker reannotation ...")
    print(f"     Current unique types: {adata.obs['CellType_harmonized'].nunique()}")

    # Build Condition column
    adata = _build_condition(adata)

    # Prepare normalized copy for score_genes
    print("[C2] Preparing normalized matrix for scoring ...")
    adata_sc = adata.copy()
    if "log1p_norm" in adata_sc.layers:
        print("     Using existing 'log1p_norm' layer.")
        adata_sc.X = adata_sc.layers["log1p_norm"]
    else:
        print("     Normalizing .X (total -> log1p) ...")
        sc.pp.normalize_total(adata_sc, target_sum=1e4)
        sc.pp.log1p(adata_sc)

    # Calculate per-type scores
    print("[C2] Computing gene scores (19 cell types) ...")
    score_cols = []
    for ct, genes in MARKERS.items():
        avail = [g for g in genes if g in adata_sc.var_names]
        if len(avail) < 2:
            print(f"     [SKIP] {ct}: only {len(avail)} genes available")
            continue
        col = f"score_{ct.replace(' ', '_').replace('-', '_')}"
        sc.tl.score_genes(adata_sc, avail, score_name=col, use_raw=False)
        score_cols.append((ct, col))
        print(f"     {ct}: {len(avail)}/{len(genes)} genes -> {col}")

    # Winner-takes-all with threshold
    print("[C2] Applying winner-takes-all ...")
    scores_df = pd.DataFrame(
        {ct: adata_sc.obs[col].values for ct, col in score_cols},
        index=adata_sc.obs_names
    )
    max_scores    = scores_df.max(axis=1)
    top_label     = scores_df.idxmax(axis=1)
    second_scores = scores_df.apply(lambda row: row.nlargest(2).iloc[1], axis=1)
    margin        = max_scores - second_scores

    orig_label = adata.obs["CellType_harmonized"].astype(str).copy()
    new_label  = orig_label.copy()
    n_changed  = 0

    for idx in adata_sc.obs_names:
        if orig_label[idx] in PROTECTED:
            continue
        best_ct    = top_label[idx]
        best_score = max_scores[idx]
        mrg        = margin[idx]
        threshold  = THRESHOLD_HEPATO if best_ct == "Hepatocytes" else THRESHOLD_DEFAULT
        if best_score >= threshold and mrg >= MARGIN_MIN:
            if new_label[idx] != best_ct:
                new_label[idx] = best_ct
                n_changed += 1

    print(f"     Re-annotated: {n_changed:,} / {len(adata):,} cells total")

    # Summary of changes by condition
    print("[C2] Generating change summary ...")
    summary_rows = []
    for cond in ["Healthy Donors", "HCC Diseased"]:
        mask   = adata.obs["Condition"] == cond
        df_sub = pd.DataFrame({"before": orig_label[mask].values,
                               "after":  new_label[mask].values})
        changed = df_sub[df_sub["before"] != df_sub["after"]]
        if len(changed) > 0:
            chg_tbl = (changed.groupby(["before", "after"])
                       .size().reset_index(name="n_cells")
                       .sort_values("n_cells", ascending=False))
            chg_tbl["condition"] = cond
            summary_rows.append(chg_tbl)
            print(f"\n  {cond} — top changes:")
            print(chg_tbl.head(15).to_string(index=False))

    if summary_rows:
        summary_df = pd.concat(summary_rows, ignore_index=True)
        summary_csv = os.path.join(OUTPUT_DIR, "C2_reannotation_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n  Summary saved: {summary_csv}")

    # Composition comparison
    print("\n  Composition BEFORE vs AFTER (full dataset):")
    comp = pd.DataFrame({"before": orig_label.values, "after": new_label.values})
    before_vc   = comp["before"].value_counts().rename("before_n")
    after_vc    = comp["after"].value_counts().rename("after_n")
    comp_summary = pd.concat([before_vc, after_vc], axis=1).fillna(0).astype(int)
    comp_summary["delta"] = comp_summary["after_n"] - comp_summary["before_n"]
    comp_summary = comp_summary.sort_values("delta", ascending=False)
    print(comp_summary.to_string())

    # Apply updated labels
    adata.obs["CellType_harmonized"] = pd.Categorical(new_label)

    # Regenerate Fig 3A-D panels
    print("\n[C2] Regenerating Fig 3A-D panels ...")
    ct_order = adata.obs["CellType_harmonized"].value_counts().index.tolist()
    palette  = cc.glasbey[: len(ct_order)]
    ct_color = dict(zip(ct_order, palette))

    plot_umap(adata, ct_order, ct_color,
              "Healthy Donors", "Healthy Condition",
              os.path.join(OUTPUT_DIR, "Fig3A_UMAP_healthy.png"))
    plot_umap(adata, ct_order, ct_color,
              "HCC Diseased", "Hepatocarcinoma Condition",
              os.path.join(OUTPUT_DIR, "Fig3B_UMAP_HCC.png"))
    plot_umap(adata, ct_order, ct_color,
              None, "Merged Conditions",
              os.path.join(OUTPUT_DIR, "Fig3C_UMAP_merged.png"))
    plot_composition(adata, ct_order, ct_color,
                     os.path.join(OUTPUT_DIR, "Fig3D_composition.png"))

    print("[C2] Done.")
    return adata


# ============================================================
# ===== FIG3 PANELS FIXES =====
# ============================================================

def apply_fig3_panels_fixes(adata: sc.AnnData) -> sc.AnnData:
    """
    Apply targeted label fixes identified during Fig 3 review:
      1. "Hep" -> "Hepatocytes"
      2. Healthy Neutrophils in leiden cluster 23 -> "Macrophages" (MARCO+, Kupffer-like)
      3. All remaining Healthy Neutrophils -> "Hepatocytes"
         (CellType=Unknown; S100A8/S100A9 low; ALB high)

    Parameters
    ----------
    adata : integrated AnnData with Condition, CellType_harmonized, leiden_r3.

    Returns
    -------
    adata with corrected CellType_harmonized.
    """
    print("[Fig3] Applying Fig3 panel fixes ...")

    # Rebuild Condition (in case not present)
    adata = _build_condition(adata)

    ct     = adata.obs["CellType_harmonized"].astype(str).copy()
    leiden = adata.obs["leiden_r3"].astype(str)
    cond   = adata.obs["Condition"].values

    # Fix 3a: "Hep" -> "Hepatocytes"
    hep_mask = ct == "Hep"
    print(f"     'Hep' -> 'Hepatocytes': {hep_mask.sum()} cells")
    ct.loc[hep_mask] = "Hepatocytes"

    # Fix 3b: Healthy Neutrophils in leiden 23 -> Macrophages (MARCO+ Kupffer-like)
    mac_mask = ((ct == "Neutrophils") &
                (pd.Series(cond, index=ct.index) == "Healthy Donors") &
                (leiden == "23"))
    print(f"     Healthy Neutrophils leiden 23 -> 'Macrophages': {mac_mask.sum()} cells")
    ct.loc[mac_mask] = "Macrophages"

    # Fix 3c: All remaining Healthy Neutrophils -> Hepatocytes
    hep2_mask = ((ct == "Neutrophils") &
                 (pd.Series(cond, index=ct.index) == "Healthy Donors"))
    print(f"     Remaining Healthy Neutrophils -> 'Hepatocytes': {hep2_mask.sum()} cells")
    ct.loc[hep2_mask] = "Hepatocytes"

    adata.obs["CellType_harmonized"] = pd.Categorical(ct)
    print("\n     CellType_harmonized after Fig3 corrections:")
    print(adata.obs["CellType_harmonized"].value_counts().to_string())

    print("[Fig3] Done.")
    return adata


# ============================================================
# ===== RECLASSIFY AND FIG3D =====
# ============================================================

def apply_reclassify_and_fig3d(adata: sc.AnnData) -> sc.AnnData:
    """
    Apply three targeted reclassifications based on per-cell marker evidence,
    then regenerate Fig 3D (2-bar and 3-bar composition charts).

    Reclassifications
    -----------------
    1. Healthy "Basophils" -> T cells   [CD3D=1.689 >> TPSAB1~0]
    2. HCC "Unclassified"  -> NK cells  [NKG7 expressed in 86.9%]
    3. HCC "pDCs"          -> per-cell winner-takes-all
         JCHAIN+  (>0) -> Plasma cells
         NKG7/GZMB+ (>0) -> NK cells
         ALB+ (>0)    -> Malignant_cells (tumor) / Hepatocytes (adjacent)
         LILRA4+ (>0) -> pDCs (retained)
         none dominant -> Unclassified

    Parameters
    ----------
    adata : integrated AnnData with Condition, QC_pass, CellType_harmonized.

    Returns
    -------
    adata with corrected CellType_harmonized and new Condition_detail column.
    """
    print("[Reclassify] Applying reclassification and regenerating Fig3D ...")

    # Rebuild Condition with QC_pass guard
    dx  = adata.obs["DX"].astype(str).str.strip().str.lower()
    qc  = adata.obs["QC_pass"].astype(bool)

    is_diseased = ((dx.str.contains(r'\badjacent tissue\b', regex=True) |
                    dx.str.contains(r'\btumou?r tissue\b', regex=True)) & qc)
    is_healthy  = dx.str.fullmatch(r'healthy') & qc

    adata.obs["Condition"] = np.where(~qc, "Excluded",
                              np.where(is_diseased, "HCC Diseased",
                              np.where(is_healthy,  "Healthy Donors", "Unknown")))

    is_adjacent = dx.str.contains(r'\badjacent tissue\b', regex=True) & qc
    is_tumor    = dx.str.contains(r'\btumou?r tissue\b', regex=True) & qc
    adata.obs["Condition_detail"] = np.where(~qc, "Excluded",
                                     np.where(is_tumor,    "Tumor tissue",
                                     np.where(is_adjacent, "Adjacent tissue",
                                     np.where(is_healthy,  "Healthy Donors", "Unknown"))))

    print(f"     Condition counts:")
    print(adata.obs["Condition"].value_counts().to_string())
    print(f"\n     Condition_detail counts:")
    print(adata.obs["Condition_detail"].value_counts().to_string())

    ct   = adata.obs["CellType_harmonized"].astype(str).copy()
    cond = adata.obs["Condition"].values

    # Reclassification 1: Healthy Basophils -> T cells
    healthy_baso = ((ct == "Basophils") &
                    (pd.Series(cond, index=ct.index) == "Healthy Donors"))
    n_hbaso = healthy_baso.sum()
    ct.loc[healthy_baso] = "T cells"
    print(f"  Healthy Basophils -> T cells: {n_hbaso} cells")

    # Reclassification 2: HCC Unclassified -> NK cells
    hcc_unclass = ((ct == "Unclassified") &
                   (pd.Series(cond, index=ct.index) == "HCC Diseased"))
    n_unclass = hcc_unclass.sum()
    ct.loc[hcc_unclass] = "NK cells"
    print(f"  HCC Unclassified -> NK cells: {n_unclass} cells")

    # Reclassification 3: HCC pDCs -> per-cell winner-takes-all
    hcc_pdc_mask  = ((ct == "pDCs") &
                     (pd.Series(cond, index=ct.index) == "HCC Diseased"))
    n_pdc_total   = hcc_pdc_mask.sum()
    print(f"\n  HCC pDCs to reclassify: {n_pdc_total}")

    if n_pdc_total > 0:
        pdc_idx = adata.obs.index[hcc_pdc_mask]
        sub     = adata[pdc_idx]

        jchain = get_expr(sub, "JCHAIN")
        nkg7   = get_expr(sub, "NKG7")
        gzmb   = get_expr(sub, "GZMB")
        alb    = get_expr(sub, "ALB")
        lilra4 = get_expr(sub, "LILRA4")

        scores = np.column_stack([jchain, nkg7, gzmb, alb, lilra4])
        best   = np.argmax(scores, axis=1)
        max_s  = scores[np.arange(len(scores)), best]

        is_tumor_pdc = adata.obs.loc[pdc_idx, "Condition_detail"].values == "Tumor tissue"

        new_labels = []
        counts = {"Plasma cells": 0, "NK cells": 0, "Malignant_cells": 0,
                  "Hepatocytes": 0, "pDCs (retained)": 0, "Unclassified": 0}
        for b, ms, tumor in zip(best, max_s, is_tumor_pdc):
            if ms == 0:
                new_labels.append("Unclassified"); counts["Unclassified"] += 1
            elif b == 0:
                new_labels.append("Plasma cells"); counts["Plasma cells"] += 1
            elif b in (1, 2):
                new_labels.append("NK cells"); counts["NK cells"] += 1
            elif b == 3:
                if tumor:
                    new_labels.append("Malignant_cells"); counts["Malignant_cells"] += 1
                else:
                    new_labels.append("Hepatocytes"); counts["Hepatocytes"] += 1
            elif b == 4:
                new_labels.append("pDCs"); counts["pDCs (retained)"] += 1
            else:
                new_labels.append("Unclassified"); counts["Unclassified"] += 1

        ct.loc[pdc_idx] = new_labels
        print("  pDC reclassification result:")
        for k, v in counts.items():
            if v > 0:
                print(f"    {k}: {v}")

    adata.obs["CellType_harmonized"] = pd.Categorical(ct)
    print("\n  CellType_harmonized final:")
    print(adata.obs["CellType_harmonized"].value_counts().to_string())

    # Build palette (exclude Unclassified/Unknown for plots)
    qc_pass      = adata.obs["QC_pass"].astype(bool)
    adata_qc     = adata[qc_pass]
    ct_order     = adata_qc.obs["CellType_harmonized"].value_counts().index.tolist()
    ct_order_plot = [c for c in ct_order if c not in ("Unclassified", "Unknown")]
    palette       = cc.glasbey[: len(ct_order_plot)]
    ct_color      = dict(zip(ct_order_plot, palette))

    print("\n[Reclassify] Generating Fig3D figures ...")
    plot_composition(adata, ct_order_plot, ct_color,
                     os.path.join(OUTPUT_DIR, "Fig3D_composition.png"))
    plot_composition_3bar(adata, ct_order_plot, ct_color,
                          os.path.join(OUTPUT_DIR, "Fig3D_composition_3bar.png"))

    print("[Reclassify] Done.")
    return adata


# ============================================================
# ===== GSM4648565 HEPATOCYTE FIX =====
# ============================================================

def apply_gsm4648565_fix(adata: sc.AnnData) -> sc.AnnData:
    """
    Correct the GSM4648565 NPC sample mislabeling.

    GSM4648565 is a non-parenchymal cell (NPC) fraction.  SCANVI mislabeled
    ~4,886 cells as "Hepatocytes" (CYP3A4=0.9%, CellTypist=0 hepatocytes).

    Correction:
      - Hepatocytes from GSM4648565 -> Unclassified
      - QC_pass remains True (the sample is valid for immune analysis)
      - All other cell types in this sample are unchanged.

    Parameters
    ----------
    adata : integrated AnnData with Sample and CellType_harmonized.

    Returns
    -------
    adata with corrected CellType_harmonized.
    """
    print("[GSM4648565] Applying hepatocyte fix for NPC sample ...")

    gsm_mask = adata.obs["Sample"].astype(str).str.contains("GSM4648565")
    gsm_hep  = gsm_mask & (adata.obs["CellType_harmonized"].astype(str) == "Hepatocytes")

    print(f"     GSM4648565 total cells:   {gsm_mask.sum()}")
    print(f"     GSM4648565 'Hepatocytes': {gsm_hep.sum()}")
    print(f"\n     Current distribution GSM4648565:")
    print(adata.obs.loc[gsm_mask, "CellType_harmonized"].value_counts().to_string())

    ct = adata.obs["CellType_harmonized"].astype(str).copy()

    # Add "Unclassified" category if not present
    if hasattr(adata.obs["CellType_harmonized"], "cat"):
        if "Unclassified" not in adata.obs["CellType_harmonized"].cat.categories:
            adata.obs["CellType_harmonized"] = (
                adata.obs["CellType_harmonized"].cat.add_categories(["Unclassified"])
            )

    ct.loc[gsm_hep] = "Unclassified"
    adata.obs["CellType_harmonized"] = pd.Categorical(ct)

    print(f"     Reclassified: {gsm_hep.sum()} cells -> Unclassified")
    print(f"\n     Final distribution GSM4648565:")
    print(adata.obs.loc[gsm_mask, "CellType_harmonized"].value_counts().to_string())

    # Global hepatocyte sanity check
    print(f"\n     Total Hepatocytes now: {(adata.obs['CellType_harmonized'] == 'Hepatocytes').sum()}")
    if "Condition" in adata.obs.columns:
        qc_healthy = adata.obs["Condition"] == "Healthy Donors"
        hep_healthy = adata.obs.loc[qc_healthy & (adata.obs["CellType_harmonized"] == "Hepatocytes")]
        print(f"\n     Hepatocytes per healthy donor (QC_pass=True):")
        print(hep_healthy.groupby("Sample", observed=True).size().to_string())

    print("[GSM4648565] Done.")
    return adata


# ============================================================
# ===== MAIN =====
# ============================================================

def main():
    """
    Load scvi_integrated.h5ad once, apply all four correction passes in order,
    and save the final corrected h5ad.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load ----
    print(f"[MAIN] Loading: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)
    print(f"       Shape: {adata.shape}")
    print(f"       Current unique types: {adata.obs['CellType_harmonized'].nunique()}")

    # ===== C2: MARKER REANNOTATION =====
    adata = apply_c2_reannotation(adata)

    # ===== FIG3 PANELS FIXES =====
    adata = apply_fig3_panels_fixes(adata)

    # ===== RECLASSIFY AND FIG3D =====
    adata = apply_reclassify_and_fig3d(adata)

    # ===== GSM4648565 HEPATOCYTE FIX =====
    adata = apply_gsm4648565_fix(adata)

    # ---- Save once ----
    print(f"\n[MAIN] Saving corrected AnnData -> {INPUT_H5AD}")
    adata.write_h5ad(INPUT_H5AD)
    print("       Saved.")
    print(f"\n[OK] All corrections applied.")
    print(f"     h5ad: {INPUT_H5AD}")
    print(f"     Figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
