#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cell labelling for HEALTHY donors using a Tabula Sapiens liver reference (ref3_healthy)
and SCVI/SCANVI label transfer, with standardized var_names.

Changes vs previous version
---------------------------
- Adds `standardize_varnames()` (strip, drop trailing .digits, make unique), applied to:
  * c_adata (query)
  * rdata_3_healthy (reference) after Ensembl→symbol mapping and ENSG filtering
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scvi

# ---------- Threading / stability ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

sc.settings.verbosity = 0

# ---------- I/O paths ----------
HEALTHY_INPUT = "/home/mdiaz/HCC_project/healthy_adata/0C_doub_remov.h5ad"
HEALTHY_PREPRO = "/home/mdiaz/HCC_project/healthy_adata/1C_prepro.h5ad"
HEALTHY_FINAL = "/home/mdiaz/HCC_project/healthy_adata/1_5C_unintegrated.h5ad"
PRED_CSV = "/home/mdiaz/HCC_project/healthy_adata/HEALTHY_PREDICTIONS.csv"  # optional

REF3_DIR = "/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data3_healthy"
REF3_FILE = "3ae74888-412b-4b5c-801b-841c04ad448f.h5ad"  # Tabula Sapiens – Liver


# ---------- Utilities ----------
def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def basic_preprocess(adata: sc.AnnData) -> sc.AnnData:
    """Minimal preprocessing on the query before SCANVI (consistent with HCC flow)."""
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


def load_ref3_healthy(dirpath: str, h5ad_file: str) -> sc.AnnData:
    """Load Tabula Sapiens liver reference and keep obs=['cell_type','ID']."""
    path = os.path.join(dirpath, h5ad_file)
    print(f"[INFO] Loading Healthy (.h5ad) reference: {path}")
    ad = sc.read(path)

    if "cell_type" not in ad.obs.columns:
        raise ValueError("[ERROR] 'cell_type' not found in healthy reference obs.")
    ad.obs["ID"] = "Healthy_Liver_CELLxGENE_DB"
    ad.obs = ad.obs[["cell_type", "ID"]].copy()

    # Map Ensembl → gene symbols (safe/no-op if already symbols)
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

        # Drop remaining Ensembl ids; keep symbols only
        if hasattr(ad.var_names, "str"):
            keep = ~ad.var_names.str.startswith("ENSG")
            ad = ad[:, keep].copy()
            print(f"[INFO] Healthy ref: kept {ad.n_vars} genes after symbol filtering.")
    except Exception as e:
        warnings.warn(f"[WARN] Skipping Ensembl→symbol mapping for Healthy ref: {e}")

    # Finally, standardize names (strip, drop .digits, unique)
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


# ---------- Main ----------
def main():
    ensure_dir(HEALTHY_FINAL)

    # 1) Load HEALTHY donors and preprocess
    print(f"[INFO] Reading healthy donors AnnData from: {HEALTHY_INPUT}")
    c_adata = sc.read_h5ad(HEALTHY_INPUT)
    print(f"[INFO] Healthy donors shape before preprocessing: {c_adata.shape}")
    c_adata = basic_preprocess(c_adata)

    # Standardize query var_names (robust alignment later)
    c_adata = standardize_varnames(c_adata)

    c_adata.write_h5ad(HEALTHY_PREPRO)
    print(f"[INFO] Saved preprocessed healthy donors AnnData to: {HEALTHY_PREPRO}")

    # Prepare query metadata for SCANVI
    c_adata.obs["CellType"] = "Unknown"
    if "Sample" not in c_adata.obs.columns:
        raise ValueError("[ERROR] 'Sample' not found in c_adata.obs. It is required for Batch.")
    c_adata.obs["Batch"] = c_adata.obs["Sample"]

    # 2) Load Healthy reference
    rdata_3_healthy = load_ref3_healthy(REF3_DIR, REF3_FILE)
    rdata_3_healthy.obs["Batch"] = "ref3"
    rdata_3_healthy.obs["Sample"] = rdata_3_healthy.obs["ID"]
    rdata_3_healthy.obs.rename(columns={"cell_type": "CellType"}, inplace=True)

    # 3) Build combined AnnData and train SCVI/SCANVI
    dater = sc.concat((c_adata, rdata_3_healthy), join="outer")
    dater.obs["CellType"] = dater.obs["CellType"].astype("category")
    if "Unknown" not in list(dater.obs["CellType"].cat.categories):
        dater.obs["CellType"] = dater.obs["CellType"].cat.add_categories(["Unknown"])

    dater = harmonize_celltype_categories(dater)

    # HVGs on combined (no subsetting)
    sc.pp.highly_variable_genes(dater, flavor="seurat_v3", n_top_genes=2000, batch_key="Batch", subset=False)

    print("[INFO] Training SCVI/SCANVI for healthy donors label transfer...")
    scvi.model.SCVI.setup_anndata(dater, batch_key="Batch", labels_key="CellType", categorical_covariate_keys=["Batch"])
    vae = scvi.model.SCVI(dater)
    vae.train()

    lvae = scvi.model.SCANVI.from_scvi_model(
        vae, adata=dater, unlabeled_category="Unknown", labels_key="CellType"
    )
    lvae.train(max_epochs=20, n_samples_per_label=100)

    # Predictions (keep only healthy donors rows afterward)
    dater.obs["predicted"] = lvae.predict(dater)
    dater.obs["transfer_score"] = lvae.predict(soft=True).max(axis=1)

    # 4) Keep only query cells (healthy donors) and merge annotations
    batch_values = c_adata.obs["Batch"].unique()
    dater = dater[dater.obs["Batch"].isin(batch_values)].copy()

    c_adata.obs = c_adata.obs.merge(
        dater.obs[["predicted", "transfer_score"]],
        left_index=True, right_index=True, how="left"
    )

    # 5) Optionally merge prior CellTypist predictions if CSV exists
    if os.path.exists(PRED_CSV):
        try:
            preds = pd.read_csv(PRED_CSV, index_col=0)
            c_adata.obs = c_adata.obs.merge(preds, left_index=True, right_index=True, how="left")
            print(f"[INFO] Merged CellTypist predictions from: {PRED_CSV}")
        except Exception as e:
            warnings.warn(f"[WARN] Could not merge CellTypist predictions CSV: {e}")

    # 6) Save final healthy donors AnnData
    c_adata.write_h5ad(HEALTHY_FINAL)
    print(f"[INFO] Saved final healthy donors AnnData with labels to: {HEALTHY_FINAL}")


if __name__ == "__main__":
    main()
