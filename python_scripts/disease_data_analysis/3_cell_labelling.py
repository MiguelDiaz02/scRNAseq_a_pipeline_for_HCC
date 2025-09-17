#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HCC cell labelling with CellTypist + SCANVI.

Change: ref4 is loaded FIRST (notebook-style) right after loading raw HCC,
so any ref4 I/O issues surface immediately. No fallback: ref4 is mandatory.
References remain RAW for SCANVI (no normalize/log/filter); CellTypist runs on a
preprocessed copy of HCC only.
"""

import os
import gzip
import warnings
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import celltypist
from celltypist import models
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import gc

# --- Limit thread contention for stability ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# ----------------------------
# I/O paths
# ----------------------------
HCC_INPUT  = "/home/mdiaz/HCC_project/hcc_adata/0C_doub_remov.h5ad"
OUT_DIR    = "/home/mdiaz/HCC_project/hcc_adata"
PRED_CSV   = os.path.join(OUT_DIR, "HCC_PREDICTIONS.csv")
HCC_PREPRO = os.path.join(OUT_DIR, "1T_prepro.h5ad")
HCC_FINAL  = os.path.join(OUT_DIR, "1_5T_unintegrated.h5ad")

# References (RAW sources for SCANVI)
REF1_DIR = "/datos/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data1"        # NCI-CLARITY (GSE151530)
REF2_DIR = "/datos/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data2"        # Multi-Regional (GSE189903)
REF3_DIR = "/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data3_healthy"      # Tabula Sapiens - Liver
REF3_FILE = "3ae74888-412b-4b5c-801b-841c04ad448f.h5ad"
REF4_DIR = "/datos/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data4"        # Sequential NCI-CLARITY (GSE229772)

# Trained CellTypist models
MODEL_NAMES = {
    "ref1_model": "HepatoCellularCarcinoma_Human_Liver_GSE151530.pkl",
    "ref2_model": "HepatoCellularCarcinoma_Human_Liver_GSE189903.pkl",
    "ref4_model": "HepatoCellularCarcinoma_Human_Liver_GSE229772.pkl",
    "healthy_model": "Healthy_Human_Liver.pkl",
    "ref3_healthy_model": "Healthy_Human_Liver_cellxgeneDataSet.pkl",
}

# SCANVI hyperparameters
SCANVI_MAX_EPOCHS = 20
SCANVI_SAMPLES_PER_LABEL = 100

# Tuning for ref4 read
REF4_CHUNK_SIZE = 10000   # you can reduce to 6000/8000 if memory is tight

sc.settings.verbosity = 0


# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def preprocess_for_celltypist(adata: sc.AnnData) -> sc.AnnData:
    """Preprocess ONLY for CellTypist (not for SCANVI)."""
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if isinstance(adata.X, np.ndarray) and adata.X.dtype != np.float32:
        adata.X = adata.X.astype(np.float32, copy=False)
    return adata


# ----------------------------
# CellTypist
# ----------------------------
def load_celltypist_models(names: Dict[str, str]) -> Dict[str, Optional[models.Model]]:
    loaded = {}
    for key, name in names.items():
        try:
            loaded[key] = models.Model.load(model=name)
            print(f"[INFO] Loaded model '{name}' as '{key}'.")
        except Exception as e:
            loaded[key] = None
            warnings.warn(f"[WARN] Could not load model '{name}' ({key}). Skipping. Reason: {e}")
    return loaded


def annotate_with_models(adata: sc.AnnData, loaded_models: Dict[str, Optional[models.Model]]) -> pd.DataFrame:
    colmap = {
        "ref1_model": ("hcc_ref1_label", "hcc_ref1_score"),
        "ref2_model": ("hcc_ref2_label", "hcc_ref2_score"),
        "ref4_model": ("hcc_ref4_label", "hcc_ref4_score"),
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
        padata = pred.to_adata()
        preds[label_col] = padata.obs.loc[adata.obs_names, "predicted_labels"]
        preds[score_col] = padata.obs.loc[adata.obs_names, "conf_score"]
    return preds


# ----------------------------
# Reference loaders (RAW for SCANVI)
# ----------------------------
def load_ref_matrix_market_with_info(data_dir: str, info_key: str = "Type") -> sc.AnnData:
    files = os.listdir(data_dir)
    info_file = [x for x in files if "Info" in x][0]
    matrix_file = [x for x in files if "matrix" in x][0]
    gene_file = [x for x in files if "genes" in x][0]
    barcode_file = [x for x in files if "barcodes" in x][0]

    base = matrix_file.split(".")[0]
    print(f"[INFO] Loading Matrix Market reference in: {data_dir} (base: {base})")

    ad = sc.read_mtx(os.path.join(data_dir, matrix_file)).T
    gene_names = pd.read_csv(os.path.join(data_dir, gene_file), header=None, sep="\t")
    ad.var.index = gene_names[1].values
    ad.var_names_make_unique()

    barcodes = pd.read_csv(os.path.join(data_dir, barcode_file), header=None, sep="\t")
    ad.obs_names = barcodes[0].values

    anno = pd.read_table(os.path.join(data_dir, info_file), index_col=2)[[info_key]]
    ad.obs = ad.obs.merge(anno, left_index=True, right_index=True, how="left")
    ad.obs["ID"] = base
    return ad


def load_ref_cellxgene_h5ad_safe(
    data_dir: str,
    h5ad_file: str,
    gene_whitelist: Optional[pd.Index] = None,
    max_cells_per_label: Optional[int] = 2000,
) -> sc.AnnData:
    """Safe loader for Tabula Sapiens (.h5ad), no normalization/log for SCANVI."""
    path = os.path.join(data_dir, h5ad_file)
    print(f"[INFO] Loading healthy .h5ad reference (safe): {path}")
    ad_b = sc.read_h5ad(path, backed="r")
    if "cell_type" not in ad_b.obs.columns:
        raise ValueError("[ERROR] 'cell_type' column not found in healthy reference.")

    obs = ad_b.obs[["cell_type"]].copy()
    obs["ID"] = "Healthy_Liver_CELLxGENE_DB"

    if max_cells_per_label is not None:
        keep = []
        for _, g in obs.groupby("cell_type", sort=False):
            keep.extend(g.sample(n=min(len(g), max_cells_per_label), random_state=0).index.tolist())
        obs = obs.loc[keep]

    if gene_whitelist is not None:
        gene_mask = ad_b.var_names.isin(gene_whitelist)
    else:
        gene_mask = slice(None)

    ad_view = ad_b[obs.index, gene_mask]
    ad = ad_view.to_memory()
    ad.obs = obs.loc[ad.obs_names].copy()
    ad.var_names_make_unique()

    if sp.issparse(ad.X):
        ad.X = ad.X.tocsr().astype(np.float32)
    else:
        ad.X = csr_matrix(ad.X.astype(np.float32))
    return ad


def load_ref4_like_notebook(
    data_dir: str,
    chunk_size: int = REF4_CHUNK_SIZE,
    dtype: str = "float32",
) -> sc.AnnData:
    """
    Load ref4 EXACTLY like in the notebook:
      - Read gz with skiprows=1 (ignoring any header), sep='\\s+', header=None, chunked.
      - Concatenate chunks to a dense DataFrame (first column = genes, remaining = cells).
      - Columns 1..N are assumed to be ordered exactly as rows in cell_subtypes['sample'].
      - Build AnnData with X = expression.T, var_names = genes, obs_names = cell_subtypes['sample'].
      - Attach obs['CellType'] from cell_subtypes['subtype'], and obs['ID'] = "GSE229772".
    """
    files = os.listdir(data_dir)
    cell_subtypes_file = [x for x in files if "cell_subtypes" in x][0]
    norm_counts_file  = [x for x in files if "norm_counts" in x][0]

    print(f"[INFO] ref4 notebook-style: cell_subtypes={cell_subtypes_file}, norm_counts={norm_counts_file}")

    # Read cell_subtypes (order is critical)
    cell_subtypes = pd.read_csv(os.path.join(data_dir, cell_subtypes_file), sep="\t")
    if "sample" not in cell_subtypes.columns or "subtype" not in cell_subtypes.columns:
        raise ValueError("[ERROR] cell_subtypes must contain 'sample' and 'subtype' columns.")
    sample_order = cell_subtypes["sample"].astype(str).values
    subtype_vec  = cell_subtypes["subtype"].astype(str).values
    n_cells = len(sample_order)

    # Stream-read gz (skip header line), collect chunks
    chunks: List[pd.DataFrame] = []
    gz_path = os.path.join(data_dir, norm_counts_file)
    with gzip.open(gz_path, "rt") as f:
        for df in pd.read_csv(
            f, sep=r"\s+", header=None, chunksize=chunk_size,
            low_memory=False, skiprows=1, engine="c"
        ):
            # Ensure dtypes: col 0 is gene (string); others numeric
            df.iloc[:, 0] = df.iloc[:, 0].astype(str)
            for c in df.columns[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(dtype)
            chunks.append(df)

    if not chunks:
        raise RuntimeError("[ERROR] No data chunks read from ref4 gz file.")

    norm_counts = pd.concat(chunks, axis=0, copy=False)

    # Split gene names and expression matrix
    gene_names = norm_counts.iloc[:, 0].astype(str).values
    expr = norm_counts.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)  # genes x cells

    # Sanity check: number of columns in expr must match n_cells
    if expr.shape[1] != n_cells:
        raise RuntimeError(
            f"[ERROR] ref4 shape mismatch: expression has {expr.shape[1]} cells "
            f"but cell_subtypes has {n_cells}. Ensure column order == cell_subtypes row order."
        )

    # Build AnnData: cells x genes, CSR float32
    ad = sc.AnnData(X=csr_matrix(expr.T))
    ad.var_names = pd.Index(gene_names)
    ad.var_names_make_unique()

    ad.obs_names = pd.Index(sample_order, dtype=str)
    ad.obs["CellType"] = subtype_vec
    ad.obs["ID"] = "GSE229772"

    # Proactively free large temporaries
    del chunks, norm_counts, expr, gene_names
    gc.collect()

    return ad


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(OUT_DIR)

    # 0) Read HCC (RAW) once
    print(f"[INFO] Reading HCC AnnData from: {HCC_INPUT}")
    adata_raw = sc.read_h5ad(HCC_INPUT)
    print(f"[INFO] HCC shape (raw): {adata_raw.shape}")

    # 1) EARLY load of ref4 (first, to surface I/O issues quickly)
    print("[INFO] Early-loading ref4 first (notebook-style)...")
    ref4 = load_ref4_like_notebook(REF4_DIR)
    ref4.obs["Batch"] = "ref4"
    ref4.obs["Sample"] = ref4.obs["ID"]
    print(f"[INFO] ref4 loaded successfully with shape: {ref4.shape}")

    # 2) CellTypist on PREPROCESSED COPY of HCC
    adata_ct = preprocess_for_celltypist(adata_raw.copy())
    adata_ct.write_h5ad(HCC_PREPRO)
    print(f"[INFO] Saved preprocessed HCC AnnData (CellTypist view) to: {HCC_PREPRO}")

    loaded_models = load_celltypist_models(MODEL_NAMES)
    preds_df = annotate_with_models(adata_ct, loaded_models)
    preds_df.to_csv(PRED_CSV)
    print(f"[INFO] Saved CellTypist predictions to: {PRED_CSV}")

    # 3) SCANVI on RAW HCC + RAW references
    adata_scanvi = adata_raw.copy()
    adata_scanvi.obs["CellType"] = "Unknown"
    adata_scanvi.obs["Batch"]   = adata_scanvi.obs["Sample"]

    # ref1 & ref2 (MatrixMarket + Info)
    ref1 = load_ref_matrix_market_with_info(REF1_DIR, info_key="Type")
    ref1.obs.rename(columns={"Type": "CellType"}, inplace=True)
    ref1.obs["Batch"]  = "ref1"
    ref1.obs["Sample"] = ref1.obs["ID"]

    ref2 = load_ref_matrix_market_with_info(REF2_DIR, info_key="Type")
    ref2.obs.rename(columns={"Type": "CellType"}, inplace=True)
    ref2.obs["Batch"]  = "ref2"
    ref2.obs["Sample"] = ref2.obs["ID"]

    # ref3 (Tabula Sapiens) – safe backed loader (match genes to HCC)
    ref3 = load_ref_cellxgene_h5ad_safe(REF3_DIR, REF3_FILE, gene_whitelist=adata_scanvi.var_names)
    ref3.obs.rename(columns={"cell_type": "CellType"}, inplace=True)
    ref3.obs["Batch"]  = "ref3"
    ref3.obs["Sample"] = ref3.obs["ID"]

    # Combine: HCC + ref1 + ref2 + ref3 + ref4
    combined = sc.concat((adata_scanvi, ref1, ref2, ref3, ref4))
    combined.obs["CellType"] = combined.obs["CellType"].astype("category")

    # HVGs by batch
    sc.pp.highly_variable_genes(
        combined, flavor="seurat_v3", n_top_genes=2000, batch_key="Batch", subset=False
    )

    # Harmonize common labels
    category_mapping = {
        "B cell": "B cells",
        "B-cell": "B cells",
        "CAF": "CAFs",
        "T cell": "T cells",
        "T-cell": "T cells",
        "Malignant cell": "Malignant cells",
    }
    combined.obs["CellType"] = combined.obs["CellType"].replace(category_mapping).astype("category")
    combined.obs["CellType"] = combined.obs["CellType"].cat.remove_unused_categories()
    if "Unknown" not in list(combined.obs["CellType"].cat.categories):
        combined.obs["CellType"] = combined.obs["CellType"].cat.add_categories(["Unknown"])
    combined.obs["CellType"] = combined.obs["CellType"].fillna("Unknown")

    # Train SCVI/SCANVI
    print("[INFO] Training SCVI and SCANVI for label transfer...")
    scvi.model.SCVI.setup_anndata(
        combined, batch_key="Batch", labels_key="CellType", categorical_covariate_keys=["Batch"]
    )
    vae = scvi.model.SCVI(combined)
    vae.train()

    lvae = scvi.model.SCANVI.from_scvi_model(
        vae, adata=combined, unlabeled_category="Unknown", labels_key="CellType"
    )
    lvae.train(max_epochs=SCANVI_MAX_EPOCHS, n_samples_per_label=SCANVI_SAMPLES_PER_LABEL)

    combined.obs["predicted"] = lvae.predict(combined)
    combined.obs["transfer_score"] = lvae.predict(soft=True).max(axis=1)

    # Keep only original HCC cells
    hcc_batches = adata_scanvi.obs["Batch"].unique()
    combined_hcc = combined[combined.obs["Batch"].isin(hcc_batches)].copy()

    # Merge back predictions + CellTypist
    print("[INFO] Merging predictions into HCC AnnData...")
    adata_scanvi.obs = adata_scanvi.obs.merge(
        combined_hcc.obs[["predicted", "transfer_score"]],
        left_index=True, right_index=True, how="left"
    )
    adata_scanvi.obs = adata_scanvi.obs.merge(preds_df, left_index=True, right_index=True, how="left")

    adata_scanvi.write_h5ad(HCC_FINAL)
    print(f"[INFO] Saved final HCC AnnData with labels to: {HCC_FINAL}")


if __name__ == "__main__":
    main()
