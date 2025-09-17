#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train CellTypist reference models for liver single-cell datasets (HCC and healthy).

This script:
- Loads 4 reference datasets (two in Matrix Market format with per-cell Type,
  one healthy reference as a .h5ad from cellxgene, and one HCC dataset with
  normalized counts in a gzipped table).
- Applies a minimal, consistent preprocessing: filter genes (min_cells=10),
  normalize_total (1e4), log1p.
- Trains one CellTypist model per dataset with feature selection (top 300 genes).
- Saves models to disk and verifies they can be loaded.
"""

import os
import gzip
import warnings
from typing import List, Optional

import pandas as pd
import scanpy as sc

# celltypist is typically heavy to import; we do it once.
import celltypist
from celltypist import models

# Optional dependency used to convert Ensembl IDs to gene symbols (dataset 3)
try:
    import mygene
    HAVE_MYG = True
except Exception:
    HAVE_MYG = False
    warnings.warn(
        "mygene is not installed; Ensembl→symbol conversion will be skipped for dataset 3."
    )

# ----------------------------
# User-configurable paths
# ----------------------------
# Input roots (as in your notebook)
REF1_DIR = "/datos/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data1"
REF2_DIR = "/datos/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data2"
REF3_DIR = "/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data3_healthy"
REF3_FILE = "3ae74888-412b-4b5c-801b-841c04ad448f.h5ad"
REF4_DIR = "/datos/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data4"

# Output directory for CellTypist models (kept consistent with your original paths)
MODEL_DIR = "/datos/home/mdiaz/.celltypist/data/models"

REF1_OUT = os.path.join(MODEL_DIR, "HepatoCellularCarcinoma_Human_Liver_GSE151530.pkl")
REF2_OUT = os.path.join(MODEL_DIR, "HepatoCellularCarcinoma_Human_Liver_GSE189903.pkl")
REF3_OUT = os.path.join(MODEL_DIR, "Healthy_Human_Liver_cellxgeneDataSet.pkl")
REF4_OUT = os.path.join(MODEL_DIR, "HepatoCellularCarcinoma_Human_Liver_GSE229772.pkl")

# Training params
TOP_GENES = 300
USE_SGD = False
N_JOBS_DEFAULT = 32  # adapt to your machine


# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_and_save_model(
    adata: sc.AnnData,
    label_key: str,
    out_path: str,
    n_jobs: Optional[int] = None,
    top_genes: int = TOP_GENES,
    use_sgd: bool = USE_SGD,
) -> None:
    """Train a CellTypist model with feature selection and save it."""
    n_jobs = n_jobs or N_JOBS_DEFAULT
    print(f"[INFO] Training CellTypist model (labels='{label_key}', top_genes={top_genes}, n_jobs={n_jobs})...")
    model = celltypist.train(
        adata,
        labels=label_key,
        n_jobs=n_jobs,
        use_SGD=use_sgd,
        feature_selection=True,
        top_genes=top_genes,
    )
    print(f"[INFO] Saving model to: {out_path}")
    model.write(out_path)
    print("[INFO] Model saved.")


def basic_preprocess(adata: sc.AnnData) -> sc.AnnData:
    """Minimal consistent preprocessing prior to training."""
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


# ----------------------------
# Loaders for each dataset
# ----------------------------
def load_ref_matrix_market_with_info(data_dir: str, info_key: str = "Type") -> sc.AnnData:
    """
    Load a reference dataset stored as:
      - matrix.mtx(.gz) (Matrix Market), with genes.txt and barcodes.txt
      - an 'Info' table containing per-cell labels in column `info_key`
    Returns a single AnnData with obs['ID'] and obs[info_key].
    """
    files = os.listdir(data_dir)

    # Heuristics from your notebook
    info_file = [x for x in files if "Info" in x][0]
    matrix_file = [x for x in files if "matrix" in x][0]
    gene_file = [x for x in files if "genes" in x][0]
    barcode_file = [x for x in files if "barcodes" in x][0]

    basename = matrix_file.split(".")[0]
    print(f"[INFO] Loading Matrix Market dataset in: {data_dir} (base: {basename})")
    ad = sc.read_mtx(os.path.join(data_dir, matrix_file)).T  # genes as var, cells as obs

    # Gene names
    gene_names = pd.read_csv(os.path.join(data_dir, gene_file), header=None, sep="\t")
    ad.var.index = gene_names[1].values  # second column has official symbols
    ad.var_names_make_unique()

    # Barcodes
    barcodes = pd.read_csv(os.path.join(data_dir, barcode_file), header=None, sep="\t")
    ad.obs_names = barcodes[0].values

    # Cell annotations
    anno = pd.read_table(os.path.join(data_dir, info_file), index_col=2)[[info_key]]

    # Merge obs
    ad.obs = ad.obs.merge(anno, left_index=True, right_index=True, how="left")
    ad.obs["ID"] = basename

    # Basic checks
    missing = ad.obs[info_key].isna().sum()
    if missing > 0:
        warnings.warn(f"[WARN] {missing} cells without '{info_key}' label.")

    print(f"[INFO] Loaded with shape: {ad.shape} and labels: {ad.obs[info_key].nunique()} classes.")
    return ad


def load_ref_cellxgene_h5ad(data_dir: str, h5ad_file: str) -> sc.AnnData:
    """
    Load a healthy-liver .h5ad from cellxgene and standardize:
    - Keep obs columns: 'cell_type' + an 'ID' tag.
    - Convert Ensembl IDs to symbols (if mygene is available).
    - Drop genes still starting with 'ENSG' after mapping.
    """
    path = os.path.join(data_dir, h5ad_file)
    print(f"[INFO] Loading .h5ad: {path}")
    ad = sc.read(path)

    # Keep only cell_type and add ID
    if "cell_type" not in ad.obs.columns:
        raise ValueError("[ERROR] 'cell_type' column not found in obs.")
    ad.obs = ad.obs[["cell_type"]].copy()
    ad.obs["ID"] = "Healthy_Liver_CELLxGENE_DB"

    # Map Ensembl → symbols if possible
    if HAVE_MYG:
        print("[INFO] Mapping Ensembl gene IDs to symbols via mygene...")
        mg = mygene.MyGeneInfo()
        ensembl_ids = ad.var_names.tolist()
        gene_info = mg.querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human", as_dataframe=True)
        # Build mapping dictionary robustly
        mapping = gene_info["symbol"].dropna().to_dict()
        new_names = [mapping.get(g, g) for g in ad.var_names]
        ad.var_names = pd.Index(new_names)
        ad.var_names_make_unique()
        # Drop any genes still starting with 'ENSG'
        keep_mask = ~ad.var_names.str.startswith("ENSG")
        ad = ad[:, keep_mask].copy()
        print(f"[INFO] Post-mapping genes: {ad.n_vars} (non-ENSG).")
    else:
        print("[INFO] Skipping Ensembl→symbol mapping (mygene not available).")

    print(f"[INFO] Loaded healthy reference with shape: {ad.shape}")
    return ad


def load_ref_norm_counts_gz(data_dir: str, label_col: str = "subtype", label_key: str = "CellType") -> sc.AnnData:
    """
    Load an HCC dataset with:
      - A gzipped table of normalized counts: first column = gene names, rest = cell expression columns.
      - A cell subtypes TSV with columns ['sample', 'subtype'].
      - An 'id_name_match' file (not strictly needed for building the AnnData here).
    Returns AnnData with obs[label_key] and obs['ID'].
    """
    files = os.listdir(data_dir)
    cell_subtypes_file = [x for x in files if "cell_subtypes" in x][0]
    norm_counts_file = [x for x in files if "norm_counts" in x][0]
    id_match_file = [x for x in files if "id_name_match" in x][0]  # not used but validated

    print(f"[INFO] Loading gzipped normalized counts from: {norm_counts_file}")
    # Read gz text once in chunks, then concat
    chunks: List[pd.DataFrame] = []
    with gzip.open(os.path.join(data_dir, norm_counts_file), "rt") as f:
        for chunk in pd.read_csv(
            f, sep=r"\s+", header=None, chunksize=10000, low_memory=False, skiprows=1
        ):
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError("[ERROR] No chunks read from normalized counts file.")
    norm_counts = pd.concat(chunks, axis=0)
    print(f"[INFO] Combined norm count matrix shape (rows x cols): {norm_counts.shape}")

    # First col = gene names; rest are expression values
    gene_names = norm_counts.iloc[:, 0].astype(str).values
    expr = norm_counts.iloc[:, 1:].values  # shape: genes x cells

    # Subtypes (obs)
    cell_subtypes = pd.read_csv(os.path.join(data_dir, cell_subtypes_file), sep="\t")
    if "sample" not in cell_subtypes.columns or label_col not in cell_subtypes.columns:
        raise ValueError("[ERROR] cell_subtypes must have 'sample' and the label column.")

    # Build AnnData: we need cells as rows (obs) and genes as columns (var)
    # expr is genes x cells, so transpose to cells x genes
    ad = sc.AnnData(X=expr.T)
    ad.var_names = pd.Index(gene_names)
    ad.var_names_make_unique()

    # obs names: should match the number of rows (cells)
    # Use 'sample' column as obs_names
    if len(cell_subtypes) != ad.n_obs:
        warnings.warn(
            f"[WARN] cell_subtypes rows ({len(cell_subtypes)}) != number of cells ({ad.n_obs}). "
            "Proceeding, but check your inputs."
        )
    # Truncate/align to min length if needed to avoid shape mismatch
    min_cells = min(len(cell_subtypes), ad.n_obs)
    if min_cells != ad.n_obs:
        ad = ad[:min_cells, :].copy()
    cell_subtypes = cell_subtypes.iloc[:min_cells, :].copy()

    ad.obs_names = cell_subtypes["sample"].astype(str).values
    ad.obs[label_key] = cell_subtypes[label_col].astype(str).values
    ad.obs["ID"] = "GSE229772"

    missing = ad.obs[label_key].isna().sum()
    if missing > 0:
        warnings.warn(f"[WARN] {missing} cells without '{label_key}' label.")

    print(f"[INFO] Loaded gz reference with shape: {ad.shape}, labels: {ad.obs[label_key].nunique()} classes.")
    return ad


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(MODEL_DIR)

    # ---------- Reference 1 (HCC, Matrix Market + Info: 'Type') ----------
    ref1 = load_ref_matrix_market_with_info(REF1_DIR, info_key="Type")
    print("[INFO] Preprocessing REF1...")
    ref1 = basic_preprocess(ref1)
    print("[INFO] Training REF1 model...")
    train_and_save_model(ref1, label_key="Type", out_path=REF1_OUT, n_jobs=N_JOBS_DEFAULT)

    # ---------- Reference 2 (HCC, Matrix Market + Info: 'Type') ----------
    ref2 = load_ref_matrix_market_with_info(REF2_DIR, info_key="Type")
    print("[INFO] Preprocessing REF2...")
    ref2 = basic_preprocess(ref2)
    print("[INFO] Training REF2 model...")
    train_and_save_model(ref2, label_key="Type", out_path=REF2_OUT, n_jobs=N_JOBS_DEFAULT)

    # ---------- Reference 3 (Healthy, .h5ad from cellxgene, label: 'cell_type') ----------
    ref3 = load_ref_cellxgene_h5ad(REF3_DIR, REF3_FILE)
    print("[INFO] Preprocessing REF3 (healthy)...")
    ref3 = basic_preprocess(ref3)
    print("[INFO] Training REF3 model (healthy)...")
    train_and_save_model(ref3, label_key="cell_type", out_path=REF3_OUT, n_jobs=N_JOBS_DEFAULT)

    # ---------- Reference 4 (HCC, gz normalized counts + subtypes) ----------
    ref4 = load_ref_norm_counts_gz(REF4_DIR, label_col="subtype", label_key="CellType")
    print("[INFO] Preprocessing REF4...")
    ref4 = basic_preprocess(ref4)
    # Drop any obs/var fully NA if present (defensive)
    ref4 = ref4[~ref4.obs.isnull().any(axis=1), :]
    ref4 = ref4[:, ~ref4.var.isnull().any(axis=1)]
    print("[INFO] Training REF4 model...")
    train_and_save_model(ref4, label_key="CellType", out_path=REF4_OUT, n_jobs=N_JOBS_DEFAULT)

    # ---------- Sanity check: load models back ----------
    print("[INFO] Verifying saved models can be loaded...")
    _ = models.Model.load(model=os.path.basename(REF1_OUT))
    _ = models.Model.load(model=os.path.basename(REF2_OUT))
    _ = models.Model.load(model=os.path.basename(REF3_OUT))
    _ = models.Model.load(model=os.path.basename(REF4_OUT))
    print("[INFO] All models loaded successfully.")

    print("[DONE] All reference models trained and saved.")


if __name__ == "__main__":
    main()
