#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cell labelling for healthy liver samples using **CellTypist only** (no SCANVI).

Pipeline:
1) Load healthy AnnData (post-doublet removal).
2) Minimal preprocessing (filter genes, normalize, log1p).
3) Run CellTypist with two healthy reference models.
4) Save per-cell predictions (labels + confidence scores) to CSV.
5) Merge predictions back into AnnData and save an annotated .h5ad.

Notes:
- Expects CellTypist models to be available in ~/.celltypist/data/models.
- Adjust input/output paths as needed.
"""

import os
import warnings
from typing import Dict, Optional

import pandas as pd
import scanpy as sc
import celltypist
from celltypist import models

# ----------------------------
# I/O paths
# ----------------------------
# Input healthy AnnData (already doublet-filtered)
HEALTHY_INPUT = "/home/mdiaz/HCC_project/healthy_adata/0C_doub_remov.h5ad"

# Outputs
OUT_DIR = "/home/mdiaz/HCC_project/healthy_adata"
PRED_CSV = os.path.join(OUT_DIR, "HEALTHY_PREDICTIONS.csv")
HEALTHY_PREPRO = os.path.join(OUT_DIR, "1C_prepro.h5ad")
HEALTHY_CELLTYPED = os.path.join(OUT_DIR, "1C_celltypist_only.h5ad")

# Trained CellTypist models (must be available to celltypist)
MODEL_NAMES = {
    "healthy_model": "Healthy_Human_Liver.pkl",                 # built-in / pre-installed model
    "ref3_healthy_model": "Healthy_Human_Liver_cellxgeneDataSet.pkl",
}

# Scanpy defaults
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def basic_preprocess(adata: sc.AnnData) -> sc.AnnData:
    """Minimal preprocessing prior to predictions."""
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


# ----------------------------
# CellTypist helpers
# ----------------------------

def load_celltypist_models(names: Dict[str, str]) -> Dict[str, Optional[models.Model]]:  # type: ignore
    """
    Try to load CellTypist models by file name (resolved inside ~/.celltypist/data/models).
    Returns a dict with the same keys; missing models are set to None with a warning.
    """
    loaded = {}
    for key, name in names.items():
        try:
            loaded[key] = models.Model.load(model=name)  # type: ignore
            print(f"[INFO] Loaded model '{name}' as '{key}'.")
        except Exception as e:
            loaded[key] = None
            warnings.warn(f"[WARN] Could not load model '{name}' ({key}). Skipping. Reason: {e}")
    return loaded


def annotate_with_models(
    adata: sc.AnnData,
    loaded_models: Dict[str, Optional[models.Model]],  # type: ignore
) -> pd.DataFrame:
    """
    Run CellTypist annotate() on `adata` with the available healthy models.
    Returns a DataFrame with <label, score> columns per model.
    """
    colmap = {
        "healthy_model": ("typist_label", "typist_score"),
        "ref3_healthy_model": ("ref3_healthy_label", "ref3_healthy_score"),
    }

    preds = pd.DataFrame(index=adata.obs_names)
    for key, model in loaded_models.items():
        if model is None or key not in colmap:
            continue
        label_col, score_col = colmap[key]
        print(f"[INFO] Annotating with '{key}' → columns: {label_col}, {score_col}")
        pred = celltypist.annotate(adata, model=model, majority_voting=False)
        pred_adata = pred.to_adata()
        preds[label_col] = pred_adata.obs.loc[adata.obs_names, "predicted_labels"]
        preds[score_col] = pred_adata.obs.loc[adata.obs_names, "conf_score"]
    return preds


# ----------------------------
# Main
# ----------------------------

def main():
    ensure_dir(OUT_DIR)

    # 1) Load healthy data and preprocess for CellTypist predictions
    print(f"[INFO] Reading healthy AnnData from: {HEALTHY_INPUT}")
    adata = sc.read_h5ad(HEALTHY_INPUT)
    print(f"[INFO] Healthy shape before preprocessing: {adata.shape}")
    adata = basic_preprocess(adata)
    adata.write_h5ad(HEALTHY_PREPRO)
    print(f"[INFO] Saved preprocessed healthy AnnData to: {HEALTHY_PREPRO}")

    # 2) Load CellTypist models and annotate
    loaded_models = load_celltypist_models(MODEL_NAMES)
    preds_df = annotate_with_models(adata, loaded_models)

    # 3) Save predictions and merge into AnnData
    preds_df.to_csv(PRED_CSV)
    print(f"[INFO] Saved CellTypist predictions to: {PRED_CSV}")

    adata.obs = adata.obs.merge(preds_df, left_index=True, right_index=True, how="left")

    # 4) Save final annotated AnnData
    adata.write_h5ad(HEALTHY_CELLTYPED)
    print(f"[INFO] Saved healthy AnnData with CellTypist labels to: {HEALTHY_CELLTYPED}")


if __name__ == "__main__":
    main()
