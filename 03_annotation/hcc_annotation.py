#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HCC cell annotation via SCVI/SCANVI label transfer (updated version).

This script:
1. Loads the preprocessed HCC AnnData (t_adata) and injects raw counts from
   the doublet-removed AnnData (q_adata) into t_adata.layers["counts"] and
   t_adata.raw. The critical fix is using layer="counts" for SCVI setup.
2. Loads four reference datasets (two Matrix Market, one cellxgene h5ad, one
   gzipped normalized counts).
3. Concatenates query + references, selects HVGs, then trains SCVI → SCANVI.
4. Transfers predicted labels and confidence scores back to t_adata.
5. Saves the final HCC AnnData to FINAL_OUT.

NOTE: This is the UPDATED version (3_5_cell_labeling_updated.py) including the
      critical `layer="counts"` fix for SCVI setup.
"""

import os
import gzip
import warnings
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy.sparse import csr_matrix, issparse

# ----------------------------
# Threading / stability
# ----------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
sc.settings.verbosity = 2

# ----------------------------
# Config variables (override via environment)
# ----------------------------
# Reference data directories
REF1_DIR = os.getenv("REF1_DIR", "/datos/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data1")
REF2_DIR = os.getenv("REF2_DIR", "/datos/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data2")
REF3_DIR = os.getenv("REF3_DIR", "/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data3_healthy")
REF3_FILE = os.getenv("REF3_FILE", "3ae74888-412b-4b5c-801b-841c04ad448f.h5ad")
REF4_DIR = os.getenv("REF4_DIR", "/datos/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data4")

# HCC query — preprocessed view
T_ADATA_PRIMARY = os.getenv("T_ADATA_PRIMARY", "/home/mdiaz/HCC_project/hcc_adata/1T_prepro.h5ad")
T_ADATA_FALLBACK = os.getenv("T_ADATA_FALLBACK", "/home/mdiaz/sc_liver_data/checkpoints/1T_prepro.h5ad")

# HCC raw (for injecting counts)
Q_ADATA_PRIMARY = os.getenv("Q_ADATA_PRIMARY", "/home/mdiaz/HCC_project/hcc_adata/0C_doub_remov.h5ad")
Q_ADATA_FALLBACK = os.getenv("Q_ADATA_FALLBACK", "/home/mdiaz/sc_liver_data/checkpoints/0C_doub_remov.h5ad")

# Optional CellTypist predictions CSV (unused in main flow but kept for traceability)
T_PREDICTIONS_CSV = os.getenv("T_PREDICTIONS_CSV", "/home/mdiaz/HCC_project/hcc_adata/HCC_PREDICTIONS.csv")

# Final output
FINAL_OUT = os.getenv("FINAL_OUT", "/home/mdiaz/HCC_project/hcc_adata/1_5T_unintegrated.h5ad")


# ----------------------------
# Helpers
# ----------------------------
def ensure_csr_f32(adata: sc.AnnData) -> None:
    """Convert adata.X to CSR float32 in place."""
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    adata.X = adata.X.astype(np.float32, copy=False)


def inject_counts_and_raw_from_q(t_adata: sc.AnnData, q_adata: sc.AnnData) -> None:
    """
    Copy raw UMIs from q_adata into t_adata.layers["counts"] and freeze t_adata.raw.
    Does not alter t_adata.X (normalized). Aligns on cell/gene intersection if needed.
    """
    if q_adata is None:
        raise RuntimeError("[ERROR] No se pudo cargar el HCC crudo.")
    if "counts" in t_adata.layers and t_adata.raw is not None:
        return  # already done

    common_cells = t_adata.obs_names.intersection(q_adata.obs_names)
    if len(common_cells) == 0:
        raise RuntimeError("[ERROR] No hay celdas en común entre 1T_prepro y 0C_doub_remov.")
    common_genes = t_adata.var_names.intersection(q_adata.var_names)
    if len(common_genes) == 0:
        raise RuntimeError("[ERROR] No hay genes en común entre 1T_prepro y 0C_doub_remov.")

    if (len(common_cells) < t_adata.n_obs) or (len(common_genes) < t_adata.n_vars):
        print(f"[WARN] Subseteando a intersección: cells {len(common_cells)}/{t_adata.n_obs}, genes {len(common_genes)}/{t_adata.n_vars}")
        t_adata._inplace_subset_obs(t_adata.obs_names.isin(common_cells))
        t_adata._inplace_subset_var(t_adata.var_names.isin(common_genes))

    q_sub = q_adata[common_cells, common_genes].copy()
    if "counts" in q_sub.layers:
        counts_mat = q_sub.layers["counts"]
    else:
        counts_mat = q_sub.X

    if not issparse(counts_mat):
        counts_mat = csr_matrix(counts_mat)
    counts_mat = counts_mat.astype(np.float32, copy=False)

    t_adata.layers["counts"] = counts_mat.copy()

    # Freeze raw with raw counts
    X_norm = t_adata.X
    t_adata.X = counts_mat
    t_adata.raw = t_adata.copy()
    t_adata.X = X_norm


def ensure_counts_alias_in_refs(*refs: sc.AnnData) -> None:
    """Create a 'counts' layer in refs pointing to X (no copy)."""
    for ad in refs:
        if "counts" not in ad.layers:
            ad.layers["counts"] = ad.X


# ----------------------------
# Reference loaders
# ----------------------------
def load_ref1(dirpath: str) -> sc.AnnData:
    """Load HCC Matrix Market reference 1 (GSE151530)."""
    print(f"[INFO] Loading ref1 from: {dirpath}")
    files = os.listdir(dirpath)
    info_file = [x for x in files if "Info" in x][0]
    matrix_file = [x for x in files if "matrix" in x][0]
    gene_file = [x for x in files if "genes" in x][0]
    barcode_file = [x for x in files if "barcodes" in x][0]

    basename = matrix_file.split(".")[0]
    temp_data = sc.read_mtx(os.path.join(dirpath, matrix_file)).T
    gene_names = pd.read_csv(os.path.join(dirpath, gene_file), header=None, sep="\t")
    temp_data.var.index = gene_names[1].values
    temp_data.var_names_make_unique()

    barcodes = pd.read_csv(os.path.join(dirpath, barcode_file), header=None, sep="\t")
    temp_data.obs_names = barcodes[0].values

    temp_anno = pd.read_table(os.path.join(dirpath, info_file), index_col=2)[["Type"]]
    temp_data.obs = temp_data.obs.merge(right=temp_anno, left_index=True, right_index=True)
    temp_data.obs["ID"] = basename

    rdata_1 = sc.concat([temp_data])
    rdata_1.obs["Batch"] = "ref1"
    rdata_1.obs["Sample"] = rdata_1.obs["ID"]
    rdata_1.obs.rename(columns={"Type": "CellType"}, inplace=True)
    ensure_csr_f32(rdata_1)
    return rdata_1


def load_ref2(dirpath: str) -> sc.AnnData:
    """Load HCC Matrix Market reference 2 (GSE189903)."""
    print(f"[INFO] Loading ref2 from: {dirpath}")
    files = os.listdir(dirpath)
    info_file = [x for x in files if "Info" in x][0]
    matrix_file = [x for x in files if "matrix" in x][0]
    gene_file = [x for x in files if "genes" in x][0]
    barcode_file = [x for x in files if "barcodes" in x][0]

    basename = matrix_file.split(".")[0]
    temp_data = sc.read_mtx(os.path.join(dirpath, matrix_file)).T
    gene_names = pd.read_csv(os.path.join(dirpath, gene_file), header=None, sep="\t")
    temp_data.var.index = gene_names[1].values
    temp_data.var_names_make_unique()

    barcodes = pd.read_csv(os.path.join(dirpath, barcode_file), header=None, sep="\t")
    temp_data.obs_names = barcodes[0].values

    temp_anno = pd.read_table(os.path.join(dirpath, info_file), index_col=2)[["Type"]]
    temp_data.obs = temp_data.obs.merge(right=temp_anno, left_index=True, right_index=True)
    temp_data.obs["ID"] = basename

    rdata_2 = sc.concat([temp_data])
    rdata_2.obs["Batch"] = "ref2"
    rdata_2.obs["Sample"] = rdata_2.obs["ID"]
    rdata_2.obs.rename(columns={"Type": "CellType"}, inplace=True)
    ensure_csr_f32(rdata_2)
    return rdata_2


def load_ref3_healthy(dirpath: str, h5ad_file: str) -> sc.AnnData:
    """Load healthy liver reference from cellxgene .h5ad (Tabula Sapiens)."""
    print(f"[INFO] Loading ref3_healthy from: {os.path.join(dirpath, h5ad_file)}")
    adata = sc.read(os.path.join(dirpath, h5ad_file))
    print(f"[INFO] ref3_healthy loaded with shape: {adata.shape}")

    adata.obs["ID"] = "Healthy_Liver_CELLxGENE_DB"
    rdata_3_healthy = sc.concat([adata])
    rdata_3_healthy.obs = rdata_3_healthy.obs[["cell_type", "ID"]]

    try:
        import mygene
        mg = mygene.MyGeneInfo()
        ensembl_ids = rdata_3_healthy.var_names.tolist()
        gene_info = mg.querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human")
        ensembl_to_symbol = {item["query"]: item["symbol"] for item in gene_info if "symbol" in item}
        print(f"[INFO] Mapped {len(ensembl_to_symbol)} genes to symbols (ref3_healthy).")
        new_names = rdata_3_healthy.var_names.map(ensembl_to_symbol)
        rdata_3_healthy.var_names = new_names.where(new_names.notna(), rdata_3_healthy.var_names)
        rdata_3_healthy.var_names_make_unique()
    except Exception as e:
        warnings.warn(f"[WARN] Gene symbol mapping for ref3_healthy skipped: {e}")

    rdata_3_healthy.obs.rename(columns={"cell_type": "CellType"}, inplace=True)
    rdata_3_healthy.obs["Batch"] = "ref3"
    rdata_3_healthy.obs["Sample"] = rdata_3_healthy.obs["ID"]
    ensure_csr_f32(rdata_3_healthy)
    return rdata_3_healthy


def load_ref4_notebook_style(dirpath: str, chunk_size: int = 10000) -> sc.AnnData:
    """Load HCC reference 4 (GSE229772) from gzipped normalized counts."""
    print(f"[INFO] Loading ref4 (notebook-style) from: {dirpath}")
    files = os.listdir(dirpath)
    cell_subtypes_file = [x for x in files if "cell_subtypes" in x][0]
    norm_counts_file = [x for x in files if "norm_counts" in x][0]

    print(f"[INFO] Files: cell_subtypes={cell_subtypes_file}, norm_counts={norm_counts_file}")

    cell_subtypes = pd.read_csv(os.path.join(dirpath, cell_subtypes_file), sep="\t")
    print(f"[INFO] cell_subtypes head:\n{cell_subtypes.head()}")

    chunks: List[pd.DataFrame] = []
    with gzip.open(os.path.join(dirpath, norm_counts_file), "rt") as f:
        for ch in pd.read_csv(f, sep=r"\s+", header=None, chunksize=chunk_size, low_memory=False, skiprows=1):
            chunks.append(ch)
    if not chunks:
        raise RuntimeError("[ERROR] No data chunks read from ref4 gz file.")

    norm_counts = pd.concat(chunks, axis=0)
    print(f"[INFO] Combined normalized matrix shape: {norm_counts.shape}")

    gene_names = norm_counts.iloc[:, 0].astype(str).values
    expression_matrix = norm_counts.iloc[:, 1:].values  # genes x cells

    temp_data = sc.AnnData(X=expression_matrix.T)
    temp_data.var_names = gene_names
    temp_data.obs_names = cell_subtypes["sample"].values
    temp_data.obs["CellType"] = cell_subtypes["subtype"].values
    temp_data.obs["ID"] = "GSE229772"

    rdata_4 = sc.concat([temp_data])
    rdata_4.obs["Batch"] = "ref4"
    rdata_4.obs["Sample"] = rdata_4.obs["ID"]
    ensure_csr_f32(rdata_4)
    return rdata_4


# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Load t_adata (preprocessed HCC view)
    t_path = T_ADATA_PRIMARY if os.path.exists(T_ADATA_PRIMARY) else T_ADATA_FALLBACK
    print(f"[INFO] Reading t_adata from: {t_path}")
    t_adata = sc.read_h5ad(t_path)
    print(f"[INFO] t_adata shape: {t_adata.shape}")

    # 2) Load raw HCC AnnData and inject counts + raw
    q_path = Q_ADATA_PRIMARY if os.path.exists(Q_ADATA_PRIMARY) else Q_ADATA_FALLBACK
    print(f"[INFO] Reading q_adata (RAW) from: {q_path}")
    q_adata = sc.read_h5ad(q_path)
    if q_adata.raw is None:
        warnings.warn("[WARN] q_adata.raw es None; continuaré pero se esperaba que existiera.")
    inject_counts_and_raw_from_q(t_adata, q_adata)

    t_adata.obs["CellType"] = "Unknown"
    t_adata.obs["Batch"] = t_adata.obs["Sample"]
    ensure_csr_f32(t_adata)

    # 3) Load references
    rdata_1 = load_ref1(REF1_DIR)
    rdata_2 = load_ref2(REF2_DIR)
    rdata_3_healthy = load_ref3_healthy(REF3_DIR, REF3_FILE)
    rdata_4 = load_ref4_notebook_style(REF4_DIR)

    # 4) Ensure layer "counts" in refs (no memory copy)
    ensure_counts_alias_in_refs(rdata_1, rdata_2, rdata_3_healthy, rdata_4)

    # 5) Build combined AnnData (temporarily remove raw from t_adata to save RAM)
    raw_snapshot = t_adata.raw
    t_adata.raw = None

    print("[INFO] Building combined AnnData (concat)...")
    dater = sc.concat((t_adata, rdata_1, rdata_2, rdata_3_healthy, rdata_4))
    sc.pp.highly_variable_genes(dater, flavor="seurat_v3", n_top_genes=2000, batch_key="Batch", subset=False)

    category_mapping = {
        "B cell": "B cells",
        "B-cell": "B cells",
        "CAF": "CAFs",
        "T cell": "T cells",
        "T-cell": "T cells",
        "Malignant cell": "Malignant cells",
        "Unknown": "Unknown",
    }
    dater.obs["CellType"] = dater.obs["CellType"].replace(category_mapping).astype("category")
    dater.obs["CellType"] = dater.obs["CellType"].cat.remove_unused_categories()
    dater.obs["CellType"] = dater.obs["CellType"].fillna("Unknown")

    # 6) SCVI/SCANVI on RAW counts layer — critical fix: layer="counts"
    print("[INFO] Training SCVI/SCANVI on RAW counts layer...")
    scvi.model.SCVI.setup_anndata(
        dater,
        layer="counts",                    # <- forces raw counts
        batch_key="Batch",
        labels_key="CellType",
        categorical_covariate_keys=["Batch"],
    )
    vae = scvi.model.SCVI(dater)
    vae.train()

    lvae = scvi.model.SCANVI.from_scvi_model(
        vae, adata=dater, unlabeled_category="Unknown", labels_key="CellType"
    )
    lvae.train(max_epochs=20, n_samples_per_label=100)

    dater.obs["predicted"] = lvae.predict(dater)
    soft = lvae.predict(soft=True)
    if hasattr(soft, "max"):
        dater.obs["transfer_score"] = np.asarray(soft).max(axis=1)
    else:
        dater.obs["transfer_score"] = lvae.predict(dater, soft=False)

    # 7) Retain only HCC query cells and merge results back to t_adata
    hcc_batches = t_adata.obs["Batch"].unique()
    dater_hcc = dater[dater.obs["Batch"].isin(hcc_batches)].copy()

    if "counts" not in t_adata.layers:
        raise RuntimeError("No encuentro la capa 'counts' en t_adata para reconstruir .raw.")

    _Xnorm = t_adata.X
    t_adata.X = t_adata.layers["counts"]
    t_adata.raw = t_adata.copy()
    t_adata.X = _Xnorm
    del _Xnorm

    t_adata.obs = t_adata.obs.merge(
        dater_hcc.obs[["predicted", "transfer_score"]],
        left_index=True, right_index=True, how="left"
    )

    # 8) Save final output
    os.makedirs(os.path.dirname(FINAL_OUT), exist_ok=True)
    t_adata.write_h5ad(FINAL_OUT)
    print(f"[INFO] Saved final HCC AnnData (with counts/raw) to: {FINAL_OUT}")

    # 9) Optional diagnostic CSVs
    out_dir = os.path.dirname(FINAL_OUT)

    shapes_df = pd.DataFrame({
        "dataset": ["t_adata(HCC)", "r1", "r2", "r3", "r4", "dater (all)"],
        "n_cells": [t_adata.n_obs, rdata_1.n_obs, rdata_2.n_obs, rdata_3_healthy.n_obs, rdata_4.n_obs, dater.n_obs],
        "n_genes": [t_adata.n_vars, rdata_1.n_vars, rdata_2.n_vars, rdata_3_healthy.n_vars, rdata_4.n_vars, dater.n_vars],
    })
    shapes_df.to_csv(os.path.join(out_dir, "manual_compare_shapes.csv"), index=False)

    def counts_by_label(df_obs, label_col="CellType"):
        """Return value counts for a label column."""
        if label_col in df_obs.columns:
            return df_obs[label_col].value_counts().sort_index()
        return pd.Series(dtype=int)

    counts_dict = {
        "r1": counts_by_label(rdata_1.obs),
        "r2": counts_by_label(rdata_2.obs),
        "r3": counts_by_label(rdata_3_healthy.obs),
        "r4": counts_by_label(rdata_4.obs),
    }
    pd.DataFrame(counts_dict).fillna(0).astype(int).to_csv(
        os.path.join(out_dir, "manual_compare_celltype_counts.csv")
    )

    if "predicted" in t_adata.obs.columns:
        t_adata.obs["predicted"].value_counts().sort_index().to_csv(
            os.path.join(out_dir, "manual_compare_predicted_counts.csv"), header=["count"]
        )

    if "transfer_score" in t_adata.obs.columns:
        ts = t_adata.obs["transfer_score"].dropna()
        stats = {
            "n": int(ts.shape[0]),
            "min": float(ts.min()) if ts.shape[0] else np.nan,
            "p25": float(ts.quantile(0.25)) if ts.shape[0] else np.nan,
            "median": float(ts.median()) if ts.shape[0] else np.nan,
            "p75": float(ts.quantile(0.75)) if ts.shape[0] else np.nan,
            "max": float(ts.max()) if ts.shape[0] else np.nan,
        }
        pd.DataFrame([stats]).to_csv(
            os.path.join(out_dir, "manual_compare_transfer_score_summary.csv"), index=False
        )


if __name__ == "__main__":
    main()
