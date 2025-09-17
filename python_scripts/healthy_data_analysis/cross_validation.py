#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross‑validation (CV) for the Healthy dataset (REF3) using SCVI/SCANVI.
- Excludes all diseased datasets.
- K‑fold CV stratified by cell types.
- Artifacts per fold: train/test .h5ad, confusion_table.csv, counts, pred_score_summary.csv,
  oott_rate.csv (~0 in CV), histogram of pred_score, UMAPs comparing true vs predicted labels,
  and classification metrics (precision/recall/F1) per class + PR curves.

Requirements: scanpy, scvi-tools, pandas, numpy, matplotlib, seaborn, scikit-learn
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scvi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

# ---------- Threading / allocator hints ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

sc.settings.verbosity = 0

# ---------- Paths ----------
REF3_DIR  = "/home/mdiaz/sc_liver_data/ref_data/model_1/ref_data3_healthy"
REF3_FILE = "3ae74888-412b-4b5c-801b-841c04ad448f.h5ad"
OUT_BASE  = "/home/mdiaz/HCC_project/Healthy_validation_analysis"
N_SPLITS  = 5
SEED      = 0

# ---------- Utils ----------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def standardize_varnames(ad: sc.AnnData) -> sc.AnnData:
    v = pd.Index(ad.var_names.astype(str))
    v = v.str.strip().str.replace(r"\.\d+$", "", regex=True)
    ad.var_names = v
    ad.var_names_make_unique()
    return ad


def canonicalize_celltypes(
    adata: sc.AnnData,
    src_col: str = "cell_type",
    out_col: str = "CellType"
) -> sc.AnnData:
    mapping = {
        "fibroblast": "Fibroblasts",
        "t cell": "T cells",
        "endothelial cell": "Endothelial",
        "hepatocyte": "Hepatocytes",
        "macrophage": "Macrophages",
        "b cell": "B cells",
        "monocyte": "Monocytes",
        "natural killer cell": "NK cells",
        "basophil": "Basophils",
        "neutrophil": "Neutrophils",
        "plasmacytoid dendritic cell": "pDCs",
        "plasma cell": "Plasma cells",
        "conventional dendritic cell": "cDCs",
        "cholangiocyte": "Cholangiocytes",
    }

    src_raw = adata.obs[src_col].astype("string")           
    src_std = src_raw.str.strip().str.lower()               

    mapped = src_std.map(mapping)

    fallback = src_raw.fillna("Unknown").astype(str)
    canon = mapped.fillna(fallback)

    adata.obs[out_col] = pd.Categorical(canon)
    adata.obs[out_col] = adata.obs[out_col].cat.remove_unused_categories()

    if "Unknown" not in adata.obs[out_col].cat.categories:
        adata.obs[out_col] = adata.obs[out_col].cat.add_categories(["Unknown"])

    return adata



# ---------- CV helpers ----------
def stratified_kfold_indices(labels: pd.Series, n_splits: int = N_SPLITS, random_state: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx = np.arange(labels.shape[0])
    return list(skf.split(idx, labels.values))

# ------- Sanitize before scANVI labeling -------
def sanitize_labels_for_scanvi(adata: sc.AnnData, key: str = "CellType", unlabeled: str = "Unknown") -> None:
    s = adata.obs[key]
    # Pasa por StringDtype para capturar <NA>, luego normaliza a objeto
    s = s.astype("string").fillna(unlabeled)
    # Normaliza blancos y variantes textuales de nulos
    s = s.str.strip().replace(
        {pd.NA: unlabeled, "nan": unlabeled, "NaN": unlabeled, "None": unlabeled, "": unlabeled}
    )
    # Construye categórico sin categorías nulas
    cat = pd.Categorical(s)
    clean_cats = [c for c in cat.categories if (c is not None) and (c == c)]  # c==c filtra NaN
    cat = pd.Categorical(s, categories=clean_cats, ordered=False)
    adata.obs[key] = cat
    if unlabeled not in adata.obs[key].cat.categories:
        adata.obs[key] = adata.obs[key].cat.add_categories([unlabeled])
    # Defensa extra
    assert not any(pd.isna(adata.obs[key].cat.categories)), "Null categories still present after sanitize."



# ---------- Run one fold ----------
def run_fold(
    ad_full: sc.AnnData,
    fold_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    out_base: str,
    seed: int = 0,
    n_latent: int = 30,
    max_epochs_scvi: int = 200,
    max_epochs_scanvi: int = 20,
    n_samples_per_label: int = 100,
    score_threshold: float = 0.8,
):
    scvi.settings.seed = seed
    out_dir = os.path.join(out_base, fold_name)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "pr_curves"))

    ad_train = ad_full[train_idx].copy()
    ad_test  = ad_full[test_idx].copy()

    # --- Quick sanity check on TEST ---
    print("[DEBUG] TEST shape:", ad_test.shape)
    print("[DEBUG] TEST obs columns:", list(ad_test.obs.columns)[:20])
    print("[DEBUG] TEST var genes (n):", ad_test.n_vars)
    print("[DEBUG] TEST CellType unique (first 15):", pd.Index(ad_test.obs.get("CellType", pd.Series([], dtype=str))).unique()[:15])

    # --- Align genes between TRAIN and TEST on intersection (order matched to TRAIN) ---
    common_genes = ad_train.var_names.intersection(ad_test.var_names)
    if len(common_genes) == 0:
        raise RuntimeError("No overlapping genes between TRAIN and TEST.")

# Reindex TEST to TRAIN’s gene order
    ad_test = ad_test[:, common_genes].copy()
    ad_test = ad_test[:, ad_train.var_names.intersection(ad_test.var_names)].copy()  # keep order consistent

# (Optional) reindex TRAIN too, to be explicit and ensure identical order:
    ad_train = ad_train[:, common_genes].copy()
    ad_train = ad_train[:, ad_test.var_names].copy()

    sc.pp.highly_variable_genes(ad_train, n_top_genes=2000, flavor="seurat_v3", batch_key="Batch", subset=False)

    sanitize_labels_for_scanvi(ad_train, key="CellType", unlabeled="Unknown")
    scvi.model.SCVI.setup_anndata(ad_train, batch_key="Batch", labels_key="CellType")
    vae = scvi.model.SCVI(ad_train, n_latent=n_latent)
    vae.train(max_epochs=max_epochs_scvi)


    lvae = scvi.model.SCANVI.from_scvi_model(
        vae, adata=ad_train, unlabeled_category="Unknown", labels_key="CellType"
    )
    lvae.train(max_epochs=max_epochs_scanvi, n_samples_per_label=n_samples_per_label)

    # --- TEST: preparar sin columna de etiquetas para evitar conflictos de categorías ---
    ad_test_copy = ad_test.copy()

# Conserva las etiquetas verdaderas para evaluación y figuras
    ad_test_copy.obs["CellType_true"] = ad_test_copy.obs["CellType"].astype(str).values

# Elimina la columna de etiquetas de TEST antes del registro (¡clave!)
    if "CellType" in ad_test_copy.obs.columns:
        ad_test_copy.obs.drop(columns=["CellType"], inplace=True)

# Registra SOLO el batch en TEST
    scvi.model.SCVI.setup_anndata(ad_test_copy, batch_key="Batch")

# Cargar TEST en el modelo entrenado (sin extend_categories)
    scanvi_test = scvi.model.SCANVI.load_query_data(ad_test_copy, lvae)
    scanvi_test.train(max_epochs=0)  # no actualiza pesos

# Predicciones y scores
    ad_test_copy.obs["predicted"]  = scanvi_test.predict(ad_test_copy)
    soft = scanvi_test.predict(ad_test_copy, soft=True)
    ad_test_copy.obs["pred_score"] = soft.max(axis=1)


    ad_test_copy.obs["CellType_true"] = ad_test.obs["CellType"].astype(str).values
    oott_rate = 0.0

    train_path = os.path.join(out_dir, "train.h5ad")
    test_path  = os.path.join(out_dir, "test_predicted.h5ad")
    ad_train.write(train_path)
    ad_test_copy.write(test_path)

    cm = pd.crosstab(ad_test_copy.obs["CellType_true"], ad_test_copy.obs["predicted"], dropna=False)
    cm.to_csv(os.path.join(out_dir, "confusion_table.csv"))

    ad_test_copy.obs["predicted"].value_counts().sort_index().to_csv(
        os.path.join(out_dir, "predicted_counts.csv"), header=["count"]
    )
    ad_test_copy.obs["CellType_true"].value_counts().sort_index().to_csv(
        os.path.join(out_dir, "true_counts.csv"), header=["count"]
    )

    s = ad_test_copy.obs["pred_score"].dropna()
    pd.DataFrame([{
        "n_cells": int(s.shape[0]),
        "frac_ge_threshold": float((s >= score_threshold).mean()) if s.shape[0] else np.nan,
        "min": float(s.min()) if s.shape[0] else np.nan,
        "p25": float(s.quantile(0.25)) if s.shape[0] else np.nan,
        "median": float(s.median()) if s.shape[0] else np.nan,
        "p75": float(s.quantile(0.75)) if s.shape[0] else np.nan,
        "max": float(s.max()) if s.shape[0] else np.nan,
    }]).to_csv(os.path.join(out_dir, "pred_score_summary.csv"), index=False)

    pd.DataFrame([{"out_of_taxonomy_rate": oott_rate}]).to_csv(
        os.path.join(out_dir, "oott_rate.csv"), index=False
    )

    plt.figure(figsize=(6, 4), dpi=150)
    sns.histplot(ad_test_copy.obs["pred_score"], bins=50, kde=False)
    plt.axvline(score_threshold, color="red", linestyle="--", linewidth=1)
    plt.title(f"{fold_name}: SCANVI prediction score (Healthy CV)")
    plt.xlabel("pred_score")
    plt.ylabel("cells")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_score_hist.png"))
    plt.close()

    ad_comb = sc.concat((ad_train, ad_test_copy), join="outer")
    ad_comb.raw = None
# Nos quedamos solo con TEST para las figuras que usan CellType_true
    is_test = ad_comb.obs["Batch"].isin(ad_test.obs["Batch"].unique())
    ad_comb = ad_comb[is_test].copy()

# Evitar que scanpy busque en .raw por error
    ad_comb.raw = None

# Vecindarios/UMAP sobre la representación latente si ya la calculaste en TRAIN/TEST;
# en su defecto, calcula UMAP básico en ad_comb
    if "X_scVI" in ad_comb.obsm:
        sc.pp.neighbors(ad_comb, use_rep="X_scVI")
    else:
        sc.pp.pca(ad_comb)
        sc.pp.neighbors(ad_comb)
    sc.tl.umap(ad_comb)

# --- Plots con use_raw=False para evitar conflictos ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

# TEST: etiquetas verdaderas
    sc.pl.umap(
        ad_comb,
        color="CellType_true",
        ax=ax[0],
        show=False,
        title="TEST: true CellType",
        legend_loc="right margin",
        legend_fontsize=7,
        size=3,
        frameon=False,
        use_raw=False,
    )

# TEST: etiquetas predichas
    if "predicted" in ad_comb.obs.columns:
        sc.pl.umap(
            ad_comb,
            color="predicted",
            ax=ax[1],
            show=False,
            title="TEST: predicted (SCANVI)",
            legend_loc="right margin",
            legend_fontsize=7,
            size=3,
            frameon=False,
            use_raw=False,
        )

    for axi in ax:
        axi.set_xticks([]); axi.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_test_true_vs_pred.png"), dpi=300, bbox_inches="tight")
    plt.close()

    y_true = ad_test_copy.obs["CellType_true"].astype(str)
    y_pred = ad_test_copy.obs["predicted"].astype(str)
    classes = sorted(y_true.unique().tolist())

    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(out_dir, "classification_report.csv"))

    # === PNG exports for confusion matrix and per-class metrics ===
    # 1) Confusion matrix (raw counts)
    plt.figure(figsize=(max(6, 0.4 * cm.shape[1]), max(5, 0.35 * cm.shape[0])), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=True, square=True)
    plt.title(f"{fold_name}: Confusion matrix (counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_counts.png"))
    plt.close()

    # 2) Confusion matrix (row-normalized)
    cm_norm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    plt.figure(figsize=(max(6, 0.4 * cm.shape[1]), max(5, 0.35 * cm.shape[0])), dpi=150)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis", cbar=True, square=True, vmin=0, vmax=1)
    plt.title(f"{fold_name}: Confusion matrix (row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"))
    plt.close()

    # 3) Per-class Precision / Recall / F1 barplot
    per_class = df_report.loc[df_report.index.isin(classes), ["precision", "recall", "f1-score", "support"]].copy()
    per_class.index.name = "class"
    per_class.reset_index(inplace=True)
    plt.figure(figsize=(max(8, 0.5 * len(per_class)), 5), dpi=150)
    per_class_melt = per_class.melt(id_vars=["class", "support"], value_vars=["precision", "recall", "f1-score"], var_name="metric", value_name="value")
    sns.barplot(data=per_class_melt, x="class", y="value", hue="metric")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{fold_name}: Per-class Precision/Recall/F1")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_metrics.png"))
    plt.close()

    # 4) Micro / Macro / Weighted summary barplot
    # Note: 'micro avg' may be absent depending on sklearn; handle safely
    summary_keys = [k for k in ["micro avg", "macro avg", "weighted avg"] if k in df_report.index]
    summary_df = df_report.loc[summary_keys, ["precision", "recall", "f1-score"]].copy()
    summary_df.index.name = "summary"
    summary_df.reset_index(inplace=True)
    plt.figure(figsize=(6, 4), dpi=150)
    summary_melt = summary_df.melt(id_vars=["summary"], var_name="metric", value_name="value")
    sns.barplot(data=summary_melt, x="summary", y="value", hue="metric")
    plt.ylim(0, 1.05)
    plt.title(f"{fold_name}: Micro/Macro/Weighted metrics")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_metrics.png"))
    plt.close()

    proba_df = pd.DataFrame(soft, index=ad_test_copy.obs_names, columns=scanvi_test._label_mapping)
    proba_df = proba_df.reindex(columns=classes, fill_value=0.0)

    pr_rows = []
    for cls in classes:
        y_bin = (y_true == cls).astype(int).values
        scores = proba_df[cls].values
        if np.all(y_bin == 0):
            ap = np.nan
            precision, recall = np.array([1.0]), np.array([0.0])
        else:
            ap = average_precision_score(y_bin, scores)
            precision, recall, _ = precision_recall_curve(y_bin, scores)

        plt.figure(figsize=(4, 4), dpi=150)
        plt.step(recall, precision, where="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR curve – {cls}\nAP={ap:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pr_curves", f"pr_{cls.replace(' ', '_')}.png"))
        plt.close()

        pr_rows.append({"class": cls, "average_precision": ap})

    pd.DataFrame(pr_rows).to_csv(os.path.join(out_dir, "average_precision_by_class.csv"), index=False)

    print(f"[INFO][{fold_name}] Saved artifacts under: {out_dir}")


# ---------- Main ----------
def main():
    ensure_dir(OUT_BASE)

    ad = sc.read(os.path.join(REF3_DIR, REF3_FILE))
    ad.obs["ID"] = "Healthy_Liver_CELLxGENE_DB"

    if "cell_type" not in ad.obs.columns:
        raise ValueError("Healthy reference must contain obs['cell_type'].")

    ad = canonicalize_celltypes(ad, src_col="cell_type", out_col="CellType")
    ad = standardize_varnames(ad)

    if "Sample" not in ad.obs.columns:
        ad.obs["Sample"] = ad.obs.get("donor", pd.Series(["Ref3"] * ad.n_obs, index=ad.obs_names)).astype(str).values
    ad.obs["Batch"] = ad.obs["Sample"]

    folds = stratified_kfold_indices(ad.obs["CellType"].astype(str), n_splits=N_SPLITS, random_state=SEED)

    for i, (tr, te) in enumerate(folds, start=1):
        run_fold(
            ad_full=ad,
            fold_name=f"fold{i}_HealthyCV",
            train_idx=tr,
            test_idx=te,
            out_base=OUT_BASE,
            seed=SEED + i - 1,
            n_latent=30,
            max_epochs_scvi=200,
            max_epochs_scanvi=20,
            n_samples_per_label=100,
            score_threshold=0.8,
        )

    # Optional ntfy notification
    # os.system("curl -d 'Healthy CV finished successfully' ntfy.sh/mdiazketernotificaciones")


if __name__ == "__main__":
    main()
