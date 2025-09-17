#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Harmonize cell-type labels (Healthy + HCC) preserving fine-grained classes,
integrate with SCVI, and produce publication-quality UMAPs.

Changes vs previous version:
- Keep granularity by normalizing label text without collapsing biologically
  meaningful subtypes (CD4/CD8/NK/MAIT, M1/M2, endothelial subsets, CAF flavors).
- Convert patterns like 'c1-ANGPT2-endothelial' -> 'Endothelial cell (ANGPT2+)'.
- Standardize plurals/casing (e.g., 'Cholangiocytes' -> 'Cholangiocyte').
- Custom UMAP plotting with colorcet.glasbey + adjustText label placement.

Outputs:
- Integrated object with .obsm["X_scVI"] and .obs["leiden_r3"].
- UMAPs before/after SCVI and by harmonized labels.
"""

import os
import re
import warnings
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# Optional aesthetics
try:
    import colorcet as cc  # palette: cc.glasbey
except Exception:
    cc = None
try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None

# ----------------------------
# I/O paths
# ----------------------------
HCC_FINAL_PATH     = "/home/mdiaz/HCC_project/hcc_adata/1_5T_unintegrated.h5ad"
HEALTHY_FINAL_PATH = "/home/mdiaz/HCC_project/healthy_adata/1_5C_unintegrated.h5ad"

OUT_DIR      = "/home/mdiaz/HCC_project/integration"
OUT_FIG_DIR  = os.path.join(OUT_DIR, "figures")
OUT_H5AD     = os.path.join(OUT_DIR, "scvi_integrated.h5ad")

# ----------------------------
# Runtime defaults
# ----------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sc.settings.verbosity = 0
sc.set_figure_params(dpi=150, facecolor="white")


# ----------------------------
# Gene name utilities
# ----------------------------
def standardize_varnames(ad: sc.AnnData) -> sc.AnnData:
    """Drop trailing '.digits', strip, enforce uniqueness."""
    v = pd.Index(ad.var_names.astype(str)).str.strip().str.replace(r"\.\d+$", "", regex=True)
    ad.var_names = v
    ad.var_names_make_unique()
    return ad


def intersect_genes(a: sc.AnnData, b: sc.AnnData) -> Tuple[sc.AnnData, sc.AnnData]:
    common = a.var_names.intersection(b.var_names)
    if len(common) < 2000:
        warnings.warn(f"[WARN] Low gene overlap: {len(common)} genes.")
    return a[:, common].copy(), b[:, common].copy()


def to_csr_counts(ad: sc.AnnData) -> None:
    """Ensure .X is CSR. Warn if non-integers (SCVI prefers counts)."""
    X = ad.layers["counts"] if "counts" in ad.layers else ad.X
    if not sp.issparse(X):
        X = csr_matrix(X)
    if X.nnz and not np.allclose(X.data, np.rint(X.data)):
        warnings.warn("[WARN] Matrix contains non-integers; ensure counts upstream for best SCVI behavior.")
    ad.X = X


# ----------------------------
# Label harmonization 
# ----------------------------
_PLURAL_FIX: Dict[str, str] = {
    "hepatocytes": "Hepatocyte",
    "cholangiocytes": "Cholangiocyte",
    "plasma cells": "Plasma cell",
    "mast cells": "Mast cell",
    "endothelial cells": "Endothelial cell",
    "b cells": "B cells",
    "t cells": "T cells",
}

def _strip_c_prefix(s: str) -> str:
    # remove leading tokens like 'c12-' or 'c3_'
    return re.sub(r"^c\d+[-_]+", "", s, flags=re.IGNORECASE)

def _singularize_and_title(s: str) -> str:
    low = s.lower().strip()
    if low in _PLURAL_FIX:
        return _PLURAL_FIX[low]
    return s[:1].upper() + s[1:]

def _is_gene_token(tok: str) -> bool:
    # crude heuristic: uppercase letters/numbers and length 2–8 (e.g., GZMK, VWF, TPSB2)
    return bool(re.fullmatch(r"[A-Z0-9]{2,8}", tok))

def _mk_parenthetic(gene: Optional[str]) -> str:
    return f" ({gene}+)" if gene else ""

def harmonize_label_preserving_granularity(raw: str) -> str:
    """
    Convert many variants to short, intuitive but still specific labels.
    Examples:
      - 'Cholangiocytes' -> 'Cholangiocyte'
      - 'c1-ANGPT2-endothelial' -> 'Endothelial cell (ANGPT2+)'
      - 'CD8-GZMK-effector memory T cells' -> 'CD8 T cell, effector-memory (GZMK+)'
      - 'NK-CD160-tissue resident' -> 'NK cell, tissue-resident'
      - 'TAM/TAMs' -> 'Tumor-associated macrophage (TAM)'
      - 'TEC/TECs' -> 'Tumor endothelial cell (TEC)'
    """
    if raw is None or pd.isna(raw):
        return "Unclassified"

    s = str(raw).strip()
    s = _strip_c_prefix(s)
    low = s.lower()

    # Fast path: tumor synonyms
    if low == "tumor" or "malignant" in low:
        return "Malignant cell"

    # TEC/TAM families
    if re.fullmatch(r"tec[s]?", low):
        return "Tumor endothelial cell (TEC)"
    if re.fullmatch(r"tam[s]?", low):
        return "Tumor-associated macrophage (TAM)"

    # MAIT
    if re.fullmatch(r"mait", low):
        return "MAIT cell"

    # NK with descriptors
    if low.startswith("nk-") or "natural killer" in low:
        # NK-<GENE>-<descriptor>?
        parts = re.split(r"[-_]", s)
        gene = None
        desc = None
        for p in parts[1:]:
            if _is_gene_token(p):
                gene = p
            else:
                desc = p if p not in ("cell", "cells") else desc
        desc_txt = (", " + desc.replace("resident", "resident").replace("circulatory", "circulatory")) if desc else ""
        return f"NK cell{desc_txt}{_mk_parenthetic(gene)}"

    # Rich CD4/CD8 T cell patterns
    if low.startswith("cd4-") or low.startswith("cd8-"):
        parts = re.split(r"[-_]", s)
        lineage = "CD4" if low.startswith("cd4-") else "CD8"
        gene = None
        # collect descriptors after gene token if present
        desc_tokens: List[str] = []
        for p in parts[1:]:
            if p.lower() in ("t", "tcell", "tcell(s)", "cells", "cell"):
                continue
            if _is_gene_token(p):
                gene = p
            else:
                desc_tokens.append(p)
        # normalize common phrases
        desc = "-".join(desc_tokens).replace("effector memory", "effector-memory") \
                                    .replace("central memory", "central-memory") \
                                    .replace("regulatory", "regulatory") \
                                    .replace("memory", "memory")
        desc = ", " + desc if desc else ""
        return f"{lineage} T cell{desc}{_mk_parenthetic(gene)}"

    # T cells proliferative
    if low.startswith("t cells-mki67"):
        return "T cell, proliferative (MKI67+)"

    # Generic T cells
    if "t cell" in low or low == "t cells":
        return "T cells"

    # Endothelial subsets like '<GENE>-endothelial'
    m_end = re.search(r"([A-Za-z0-9]+)[-_]endothelial", low)
    if m_end:
        gene = m_end.group(1).upper()
        return f"Endothelial cell{_mk_parenthetic(gene)}"

    # Macrophage M1/M2 with gene
    m_mphi = re.search(r"(m1|m2)\s*macrophage", low)
    if m_mphi:
        flavor = m_mphi.group(1).upper()
        gene = None
        # find a gene token if present
        for tok in re.split(r"[-_\s]", s):
            if _is_gene_token(tok):
                gene = tok
        return f"Macrophage, {flavor}{_mk_parenthetic(gene)}"

    # Monocyte with gene
    m_mono = re.search(r"([A-Za-z0-9]+)[-_]monocyte", low)
    if m_mono:
        gene = m_mono.group(1).upper()
        return f"Monocyte{_mk_parenthetic(gene)}"

    # Mast cell with gene
    m_mast = re.search(r"([A-Za-z0-9]+)[-_]mast\s*cells?", low)
    if m_mast:
        gene = m_mast.group(1).upper()
        return f"Mast cell{_mk_parenthetic(gene)}"

    # CAF flavors
    if "hepatocyte like caf" in low:
        gene = None
        for tok in re.split(r"[-_\s]", s):
            if _is_gene_token(tok):
                gene = tok
        return f"Cancer-associated fibroblast, hepatocyte-like{_mk_parenthetic(gene)}"
    if "vascular caf" in low:
        gene = None
        for tok in re.split(r"[-_\s]", s):
            if _is_gene_token(tok):
                gene = tok
        return f"Cancer-associated fibroblast, vascular{_mk_parenthetic(gene)}"
    if "inflammatory caf" in low:
        gene = None
        for tok in re.split(r"[-_\s]", s):
            if _is_gene_token(tok):
                gene = tok
        return f"Cancer-associated fibroblast, inflammatory{_mk_parenthetic(gene)}"

    # Conventional / Plasmacytoid DC
    if "plasmacytoid dendritic" in low:
        return "Plasmacytoid dendritic cell"
    if "conventional dendritic" in low:
        return "Conventional dendritic cell"

    # Generic families
    mapping_simple = {
        "cholangiocyte": "Cholangiocyte",
        "hepatocyte": "Hepatocyte",
        "macrophage": "Macrophage",
        "monocyte": "Monocyte",
        "neutrophil": "Neutrophil",
        "basophil": "Basophil",
        "fibroblast": "Fibroblast",
        "endothelial cell": "Endothelial cell",
        "plasma cell": "Plasma cell",
        "b cells": "B cells",
        "unclassified": "Unclassified",
        "unspecified": "Unclassified",
    }
    for k, v in mapping_simple.items():
        if k in low:
            return v

    # As a last step, singularize/Title without collapsing:
    s2 = s.replace("cells", "cell").replace("Cells", "Cell")
    return _singularize_and_title(s2)


def pick_label_column(ad: sc.AnnData, candidates: Optional[List[str]] = None) -> Optional[str]:
    if candidates is None:
        candidates = ["predicted", "scVI_Model_Predictions", "CellType"]
    for c in candidates:
        if c in ad.obs.columns:
            return c
    return None


# ----------------------------
# UMAP helpers (custom style)
# ----------------------------
def _compute_umap(adata: sc.AnnData, use_rep: Optional[str] = None) -> None:
    if use_rep is None:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
    else:
        sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)

def _fancy_umap(
    adata: sc.AnnData,
    label: str,
    out_png: str,
    title: Optional[str] = None,
    point_size: float = 8.0,
) -> None:
    if "X_umap" not in adata.obsm:
        raise ValueError("UMAP coordinates not found. Run sc.tl.umap first.")

    umap = adata.obsm["X_umap"]
    x, y = umap[:, 0], umap[:, 1]
    labels = adata.obs[label].astype(str)

    # unique labels / colors
    cell_types = sorted(labels.unique().tolist())
    if cc is not None:
        palette = cc.glasbey[: len(cell_types)]
    else:
        palette = sc.pl.palettes.default_102[: len(cell_types)]

    color_dict = dict(zip(cell_types, palette))
    colors = labels.map(color_dict)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("white")
    fig.patch.set_alpha(0.0)

    sc = ax.scatter(x, y, s=point_size, c=colors, linewidths=0, alpha=0.9)

    # label placements
    texts = []
    if adjust_text is not None:
        for ct in cell_types:
            mask = labels == ct
            if mask.sum() == 0:
                continue
            x_ct = np.median(x[mask])
            y_ct = np.median(y[mask])
            txt = ax.text(
                x_ct, y_ct, ct, fontsize=9, weight="bold",
                ha="center", va="center", color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.3"),
                zorder=10,
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground="white"),
                path_effects.Normal(),
            ])
            texts.append(txt)
        adjust_text(
            texts,
            only_move={"points": "y", "texts": "xy"},
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax.set_xlabel("UMAP1", fontsize=14, weight="bold")
    ax.set_ylabel("UMAP2", fontsize=14, weight="bold")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=12, weight="bold")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    # 1) Load finals
    print(f"[INFO] Loading HCC FINAL:     {HCC_FINAL_PATH}")
    ad_hcc = sc.read_h5ad(HCC_FINAL_PATH)
    print(f"[INFO] Loading Healthy FINAL: {HEALTHY_FINAL_PATH}")
    ad_hlt = sc.read_h5ad(HEALTHY_FINAL_PATH)

    # 2) Standardize varnames and intersect
    ad_hcc = standardize_varnames(ad_hcc)
    ad_hlt = standardize_varnames(ad_hlt)
    ad_hcc, ad_hlt = intersect_genes(ad_hcc, ad_hlt)

    # Flags
    ad_hcc.obs["dataset"] = "HCC"
    ad_hlt.obs["dataset"] = "Healthy"

    # CSR
    to_csr_counts(ad_hcc)
    to_csr_counts(ad_hlt)

    # 3) Harmonize labels while preserving granularity
    for ad in (ad_hcc, ad_hlt):
        col = pick_label_column(ad)
        if col is None:
            ad.obs["CellType_harmonized"] = "Unclassified"
        else:
            ad.obs["CellType_harmonized"] = ad.obs[col].astype(str).map(harmonize_label_preserving_granularity)

    # Debug: show diversity before/after (sanity check)
    for name, ad in [("HCC", ad_hcc), ("Healthy", ad_hlt)]:
        col = pick_label_column(ad)
        n_raw = ad.obs[col].nunique() if col else 0
        n_harm = ad.obs["CellType_harmonized"].nunique()
        print(f"[INFO] {name}: {n_raw} → {n_harm} unique labels after harmonization.")

    # 4) Concatenate
    adata = sc.concat([ad_hcc, ad_hlt], join="inner", label=None)
    indiv_key = "Patient" if "Patient" in adata.obs.columns else ("Sample" if "Sample" in adata.obs.columns else "dataset")
    adata.obs["Batch"] = adata.obs["dataset"].astype(str) + ":" + adata.obs[indiv_key].astype(str)

    # 5) UMAP BEFORE (uncorrected; quick log1p for visualization only)
    adata_pre = adata.copy()
    sc.pp.normalize_total(adata_pre, target_sum=1e4)
    sc.pp.log1p(adata_pre)
    sc.pp.highly_variable_genes(adata_pre, n_top_genes=2000, flavor="seurat", batch_key="Batch", subset=True)
    _compute_umap(adata_pre, use_rep=None)

    # Save pre-SCVI UMAPs
    _fancy_umap(adata_pre, label="dataset",
                out_png=os.path.join(OUT_FIG_DIR, "umap_pre_scvi_dataset.png"),
                title="Before SCVI (dataset)")
    _fancy_umap(adata_pre, label="CellType_harmonized",
                out_png=os.path.join(OUT_FIG_DIR, "umap_pre_scvi_celltype.png"),
                title="Before SCVI (harmonized cell types)")

    # 6) SCVI integration
    print("[INFO] Training SCVI...")
    scvi.model.SCVI.setup_anndata(adata, batch_key="Batch")
    model = scvi.model.SCVI(adata, n_latent=30)
    model.train()

    # 7) Latent + neighbors + UMAP + Leiden
    adata.obsm["X_scVI"] = model.get_latent_representation()
    _compute_umap(adata, use_rep="X_scVI")
    sc.tl.leiden(adata, resolution=3.0, key_added="leiden_r3")

    # 8) UMAP AFTER
    _fancy_umap(adata, label="dataset",
                out_png=os.path.join(OUT_FIG_DIR, "umap_post_scvi_dataset.png"),
                title="After SCVI (dataset)")
    _fancy_umap(adata, label="CellType_harmonized",
                out_png=os.path.join(OUT_FIG_DIR, "umap_post_scvi_celltype.png"),
                title="After SCVI (harmonized cell types)")
    _fancy_umap(adata, label="leiden_r3",
                out_png=os.path.join(OUT_FIG_DIR, "umap_post_scvi_leiden_r3.png"),
                title="After SCVI (Leiden r=3)")

    # 9) Save
    adata.write_h5ad(OUT_H5AD)
    print(f"[INFO] Saved integrated AnnData: {OUT_H5AD}")


if __name__ == "__main__":
    main()
