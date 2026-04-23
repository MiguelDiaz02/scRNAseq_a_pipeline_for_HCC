#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Healthy liver cell annotation — CellTypist then scANVI label transfer.

Consolidates two previously sequential scripts:
  - 3_celltypist_labeling.py  (PART 1)
  - 3_scANVI_labeling.py      (PART 2)

Pipeline:
  PART 1 — CellTypist
    1. Load healthy AnnData (post-doublet removal).
    2. Minimal preprocessing (filter genes, normalize, log1p).
    3. Annotate with two healthy CellTypist models.
    4. Save per-cell predictions to CSV and an intermediate .h5ad.

  PART 2 — scANVI
    1. Load preprocessed healthy AnnData (output of PART 1).
    2. Load healthy reference (Tabula Sapiens liver, cellxgene .h5ad).
    3. Concatenate query + reference, select HVGs.
    4. Train SCVI → SCANVI; transfer labels to query cells.
    5. Optionally merge PART 1 CellTypist predictions.
    6. Save final annotated .h5ad.
"""

import os
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import celltypist
from celltypist import models

# ----------------------------
# Threading / stability
# ----------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

# ----------------------------
# Config variables (override via environment)
# ----------------------------
# Input healthy AnnData (post-doublet removal)
HEALTHY_INPUT = os.getenv("HEALTHY_INPUT", "/home/mdiaz/HCC_project/healthy_adata/0C_doub_remov.h5ad")

# Output directory
OUT_DIR = os.getenv("OUT_DIR", "/home/mdiaz/HCC_project/healthy_adata")

# Intermediate and final output files
PRED_CSV        = os.path.join(OUT_DIR, "HEALTHY_PREDICTIONS.csv")
HEALTHY_PREPRO  = os.path.join(OUT_DIR, "1C_prepro.h5ad")
HEALTHY_CELLTYPED = os.path.join(OUT_DIR, "1C_celltypist_only.h5ad")
HEALTHY_FINAL   = os.path.join(OUT_DIR, "1_5C_unintegrated.h5ad")

# CellTypist model names (must be resolvable by celltypist from ~/.celltypist/data/models/)
MODEL_NAMES: Dict[str, str] = {
    "healthy_model":      os.getenv("CELLTYPIST_MODEL_HEALTHY", "Healthy_Human_Liver.pkl"),
    "ref3_healthy_model": os.getenv("CELLTYPIST_MODEL_REF3",    "Healthy_Human_Liver_cellxgeneDataSet.pkl"),
}

# Healthy reference for scANVI (Tabula Sapiens – Liver)
REF3_DIR  = os.getenv("REF3_DIR",  "/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data3_healthy")
REF3_FILE = os.getenv("REF3_FILE", "3ae74888-412b-4b5c-801b-841c04ad448f.h5ad")


# ----------------------------
# Shared utilities
# ----------------------------

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def basic_preprocess(adata: sc.AnnData) -> sc.AnnData:
    """Minimal preprocessing: filter genes, normalize, log1p."""
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if isinstance(adata.X, np.ndarray) and adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32, copy=False)
    return adata


def standardize_varnames(ad: sc.AnnData) -> sc.AnnData:
    """
    Standardize var_names:
    - cast to str
    - strip whitespace
    - drop trailing dot-number suffixes (e.g., 'TP53.1' -> 'TP53')
    - enforce uniqueness
    """
    v = pd.Index(ad.var_names.astype(str))
    v = v.str.strip().str.replace(r"\.\d+$", "", regex=True)
    ad.var_names = v
    ad.var_names_make_unique()
    return ad


# ===== PART 1: CELLTYPIST =====

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


def annotate_with_celltypist_models(
    adata: sc.AnnData,
    loaded_models: Dict[str, Optional[models.Model]],  # type: ignore
) -> pd.DataFrame:
    """
    Run CellTypist annotate() on `adata` with the available healthy models.
    Returns a DataFrame with <label, score> columns per model.
    """
    colmap = {
        "healthy_model":      ("typist_label",       "typist_score"),
        "ref3_healthy_model": ("ref3_healthy_label",  "ref3_healthy_score"),
    }

    preds = pd.DataFrame(index=adata.obs_names)
    for key, model in loaded_models.items():
        if model is None or key not in colmap:
            continue
        label_col, score_col = colmap[key]
        print(f"[INFO] Annotating with '{key}' -> columns: {label_col}, {score_col}")
        pred = celltypist.annotate(adata, model=model, majority_voting=False)
        pred_adata = pred.to_adata()
        preds[label_col] = pred_adata.obs.loc[adata.obs_names, "predicted_labels"]
        preds[score_col] = pred_adata.obs.loc[adata.obs_names, "conf_score"]
    return preds


def run_celltypist_part(adata: sc.AnnData) -> pd.DataFrame:
    """
    Execute PART 1: annotate with CellTypist healthy models.
    Saves PRED_CSV and HEALTHY_CELLTYPED in-place.
    Returns predictions DataFrame (also written to PRED_CSV).
    """
    ensure_dir(OUT_DIR)

    loaded_models = load_celltypist_models(MODEL_NAMES)
    preds_df = annotate_with_celltypist_models(adata, loaded_models)

    preds_df.to_csv(PRED_CSV)
    print(f"[INFO] Saved CellTypist predictions to: {PRED_CSV}")

    adata_ct = adata.copy()
    adata_ct.obs = adata_ct.obs.merge(preds_df, left_index=True, right_index=True, how="left")
    adata_ct.write_h5ad(HEALTHY_CELLTYPED)
    print(f"[INFO] Saved healthy AnnData with CellTypist labels to: {HEALTHY_CELLTYPED}")

    return preds_df


# ===== PART 2: SCANVI =====

def load_ref3_healthy(dirpath: str, h5ad_file: str) -> sc.AnnData:
    """Load Tabula Sapiens liver reference and keep obs=['cell_type','ID']."""
    path = os.path.join(dirpath, h5ad_file)
    print(f"[INFO] Loading Healthy (.h5ad) reference: {path}")
    ad = sc.read(path)

    if "cell_type" not in ad.obs.columns:
        raise ValueError("[ERROR] 'cell_type' not found in healthy reference obs.")
    ad.obs["ID"] = "Healthy_Liver_CELLxGENE_DB"
    ad.obs = ad.obs[["cell_type", "ID"]].copy()

    try:
        import mygene
        mg = mygene.MyGeneInfo()
        ens = ad.var_names.tolist()
        res = mg.querymany(ens, scopes="ensembl.gene", fields="symbol", species="human")
        mapper = {r["query"]: r["symbol"] for r in res if "symbol" in r}
        print(f"[INFO] Healthy ref: mapped {len(mapper)} Ensembl IDs to symbols.")
        new = ad.var_names.map(mapper)
        ad.var_names = new.where(new.notna(), ad.var_names)
        ad.var_names_make_unique()

        if hasattr(ad.var_names, "str"):
            keep = ~ad.var_names.str.startswith("ENSG")
            ad = ad[:, keep].copy()
            print(f"[INFO] Healthy ref: kept {ad.n_vars} genes after symbol filtering.")
    except Exception as e:
        warnings.warn(f"[WARN] Skipping Ensembl->symbol mapping for Healthy ref: {e}")

    ad = standardize_varnames(ad)
    return ad


def harmonize_celltype_categories(adata: sc.AnnData) -> sc.AnnData:
    """Light harmonization for common liver nomenclature."""
    mapping = {
        "B cell": "B cells",
        "B-cell": "B cells",
        "T cell": "T cells",
        "T-cell": "T cells",
        "CAF": "CAFs",
        "Malignant cell": "Malignant cells",
    }
    if "CellType" in adata.obs.columns:
        adata.obs["CellType"] = adata.obs["CellType"].replace(mapping).astype("category")
        adata.obs["CellType"] = adata.obs["CellType"].cat.remove_unused_categories()
    return adata


def run_scanvi_part(c_adata: sc.AnnData) -> None:
    """
    Execute PART 2: scANVI label transfer from Tabula Sapiens healthy reference.
    Merges CellTypist predictions from PRED_CSV (if present) and saves HEALTHY_FINAL.
    """
    ensure_dir(OUT_DIR)

    c_adata = standardize_varnames(c_adata)

    c_adata.obs["CellType"] = "Unknown"
    if "Sample" not in c_adata.obs.columns:
        raise ValueError("[ERROR] 'Sample' not found in c_adata.obs. It is required for Batch.")
    c_adata.obs["Batch"] = c_adata.obs["Sample"]

    rdata_3_healthy = load_ref3_healthy(REF3_DIR, REF3_FILE)
    rdata_3_healthy.obs["Batch"] = "ref3"
    rdata_3_healthy.obs["Sample"] = rdata_3_healthy.obs["ID"]
    rdata_3_healthy.obs.rename(columns={"cell_type": "CellType"}, inplace=True)

    dater = sc.concat((c_adata, rdata_3_healthy), join="outer")
    dater.obs["CellType"] = dater.obs["CellType"].astype("category")
    if "Unknown" not in list(dater.obs["CellType"].cat.categories):
        dater.obs["CellType"] = dater.obs["CellType"].cat.add_categories(["Unknown"])

    dater = harmonize_celltype_categories(dater)

    sc.pp.highly_variable_genes(dater, flavor="seurat_v3", n_top_genes=2000, batch_key="Batch", subset=False)

    print("[INFO] Training SCVI/SCANVI for healthy donors label transfer...")
    scvi.model.SCVI.setup_anndata(dater, batch_key="Batch", labels_key="CellType", categorical_covariate_keys=["Batch"])
    vae = scvi.model.SCVI(dater)
    vae.train()

    lvae = scvi.model.SCANVI.from_scvi_model(
        vae, adata=dater, unlabeled_category="Unknown", labels_key="CellType"
    )
    lvae.train(max_epochs=20, n_samples_per_label=100)

    dater.obs["predicted"] = lvae.predict(dater)
    dater.obs["transfer_score"] = lvae.predict(soft=True).max(axis=1)

    batch_values = c_adata.obs["Batch"].unique()
    dater = dater[dater.obs["Batch"].isin(batch_values)].copy()

    c_adata.obs = c_adata.obs.merge(
        dater.obs[["predicted", "transfer_score"]],
        left_index=True, right_index=True, how="left"
    )

    # Optionally merge prior CellTypist predictions if CSV exists (output of PART 1)
    if os.path.exists(PRED_CSV):
        try:
            preds = pd.read_csv(PRED_CSV, index_col=0)
            c_adata.obs = c_adata.obs.merge(preds, left_index=True, right_index=True, how="left")
            print(f"[INFO] Merged CellTypist predictions from: {PRED_CSV}")
        except Exception as e:
            warnings.warn(f"[WARN] Could not merge CellTypist predictions CSV: {e}")

    c_adata.write_h5ad(HEALTHY_FINAL)
    print(f"[INFO] Saved final healthy donors AnnData with labels to: {HEALTHY_FINAL}")


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(OUT_DIR)

    # Load and preprocess healthy AnnData (shared between both parts)
    print(f"[INFO] Reading healthy AnnData from: {HEALTHY_INPUT}")
    adata = sc.read_h5ad(HEALTHY_INPUT)
    print(f"[INFO] Healthy shape before preprocessing: {adata.shape}")
    adata = basic_preprocess(adata)
    adata.write_h5ad(HEALTHY_PREPRO)
    print(f"[INFO] Saved preprocessed healthy AnnData to: {HEALTHY_PREPRO}")

    # ===== PART 1: CELLTYPIST =====
    print("\n[INFO] ===== PART 1: CELLTYPIST =====")
    run_celltypist_part(adata)

    # ===== PART 2: SCANVI =====
    # Reload from disk to get a clean copy without PART 1 obs columns merged in
    print("\n[INFO] ===== PART 2: SCANVI =====")
    c_adata = sc.read_h5ad(HEALTHY_PREPRO)
    run_scanvi_part(c_adata)

    print("\n[DONE] Healthy annotation complete.")
    print(f"  Preprocessed:   {HEALTHY_PREPRO}")
    print(f"  CellTypist:     {HEALTHY_CELLTYPED}")
    print(f"  CellTypist CSV: {PRED_CSV}")
    print(f"  Final (scANVI): {HEALTHY_FINAL}")


if __name__ == "__main__":
    main()
