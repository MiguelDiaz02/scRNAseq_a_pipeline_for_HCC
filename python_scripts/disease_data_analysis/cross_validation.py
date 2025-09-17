#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diseased_cross_validation.py
Cross-validation (CV) for diseased HCC single-cell datasets using SCVI/SCANVI with
StratifiedGroupKFold (stratify by CellType, block by Sample). Uses a raw-counts
layer for HVGs and model training/mapping.

Author: Miguel & Assistant
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, average_precision_score)

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    from sklearn.model_selection import GroupKFold
    HAS_SGKF = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import scvi
from scvi.model import SCVI, SCANVI
from anndata import AnnData


# ------------------------- Utilities -------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int = 0) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def standardize_varnames(adata: AnnData) -> None:
    var_new = pd.Index(adata.var_names).str.replace(r"\.\d+$", "", regex=True)
    adata.var_names = var_new
    adata.var_names_make_unique()


def load_canonical_map(path_json: Optional[str]) -> Optional[Dict[str, str]]:
    if not path_json:
        return None
    if not os.path.exists(path_json):
        warnings.warn(f"Canonical map JSON not found: {path_json}. Proceeding without it.")
        return None
    with open(path_json, "r", encoding="utf-8") as f:
        try:
            mapping = json.load(f)
            return mapping if isinstance(mapping, dict) else None
        except Exception as e:
            warnings.warn(f"Failed to parse canonical map JSON: {e}. Ignoring.")
            return None


def apply_canonical_map(
    adata: AnnData,
    label_col: str,
    canon_map: Optional[Dict[str, str]],
    report_dir: Optional[str] = None
) -> None:
    if canon_map is None:
        return
    if label_col not in adata.obs.columns:
        raise ValueError(f"Label column '{label_col}' not found in adata.obs.")

    # Normaliza a minúsculas para buscar en el mapa (case-insensitive)
    ser = adata.obs[label_col].astype(str).str.strip()
    norm_map = {str(k).strip().lower(): v for k, v in canon_map.items()}

    # Etiquetas mapeadas (o se mantienen si no hay regla)
    lower = ser.str.lower()
    mapped = lower.map(norm_map).fillna(ser)

    # —— REPORTE de etiquetas no mapeadas —— #
    if report_dir is not None:
        unmapped_mask = lower.map(norm_map).isna()
        unmapped = ser[unmapped_mask].value_counts().sort_values(ascending=False)
        if len(unmapped) > 0:
            os.makedirs(report_dir, exist_ok=True)
            unmapped.to_csv(
                os.path.join(report_dir, "unmapped_labels_before_canonical.csv"),
                index_label="original_label", header=["count"]
            )

    # Aplica el mapeo al AnnData
    adata.obs[label_col] = pd.Categorical(mapped.astype(str))
    adata.obs[label_col] = adata.obs[label_col].cat.remove_unused_categories()


def sanitize_labels_for_scanvi(adata: AnnData, label_col: str, unknown_label: str = "Unknown") -> None:
    if label_col not in adata.obs.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    ser = adata.obs[label_col].astype(str)
    ser = ser.fillna(unknown_label).str.strip()
    ser = ser.replace("", unknown_label)
    cat = pd.Categorical(ser.astype(str))
    if unknown_label not in list(cat.categories):
        cat = cat.add_categories([unknown_label])
    adata.obs[label_col] = cat


def filter_classes_by_support(
    adata: AnnData,
    label_col: str,
    group_col: str,
    min_cells_total: int = 150,
    min_groups: int = 3,
) -> Tuple[List[str], List[str]]:
    labels = adata.obs[label_col].astype(str)
    counts = labels.value_counts()
    groups_per_label = adata.obs.groupby(label_col)[group_col].nunique()

    kept, rare = [], []
    for lbl in counts.index.tolist():
        if (counts[lbl] >= min_cells_total) and (groups_per_label.get(lbl, 0) >= min_groups):
            kept.append(lbl)
        else:
            rare.append(lbl)
    return kept, rare


def align_by_gene_intersection(ad_train: AnnData, ad_test: AnnData) -> Tuple[AnnData, AnnData]:
    inter = ad_train.var_names.intersection(ad_test.var_names)
    ad_train = ad_train[:, inter].copy()
    ad_test = ad_test[:, inter].copy()
    return ad_train, ad_test


def compute_unseen_and_oott(
    y_true: pd.Series,
    train_seen_labels: List[str],
    unknown_label: str = "Unknown"
) -> Tuple[pd.Series, float, List[str]]:
    unseen_mask = ~y_true.isin(train_seen_labels)
    unseen_labels = sorted(y_true.loc[unseen_mask].unique().tolist())
    y_true_recoded = y_true.where(~unseen_mask, other=unknown_label)
    oott_rate = float(unseen_mask.mean())
    return y_true_recoded, oott_rate, unseen_labels


def safe_stratified_group_kfold(labels: pd.Series, groups: pd.Series, n_splits: int, seed: int):
    idx = np.arange(labels.shape[0])
    if HAS_SGKF:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(sgkf.split(idx, y=labels, groups=groups))
    else:
        warnings.warn("StratifiedGroupKFold not available; using GroupKFold fallback.")
        unique_groups = groups.astype(str).unique().tolist()
        rng = np.random.default_rng(seed)
        best, best_score = None, np.inf
        for _ in range(64):
            rng.shuffle(unique_groups)
            assign = {g: (i % n_splits) for i, g in enumerate(unique_groups)}
            fold_idx = {k: [] for k in range(n_splits)}
            for i, g in enumerate(groups.astype(str)):
                fold_idx[assign[g]].append(i)
            per_fold_class = []
            for k in range(n_splits):
                yk = labels.iloc[fold_idx[k]]
                per_fold_class.append(yk.value_counts())
            df = pd.DataFrame(per_fold_class).fillna(0.0)
            score = df.std(axis=0).sum()
            if score < best_score:
                best_score, best = score, fold_idx
        splits = []
        for k in range(n_splits):
            test_idx = np.array(best[k], dtype=int)
            mask = np.ones(labels.shape[0], dtype=bool)
            mask[test_idx] = False
            train_idx = np.where(mask)[0]
            splits.append((train_idx, test_idx))
        return splits


def plot_pred_score_hist(pred_scores: np.ndarray, out_png: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(pred_scores, bins=40, alpha=0.9)
    plt.xlabel("pred_score")
    plt.ylabel("count")
    plt.title("Prediction score distribution (TEST)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_umap_true_vs_pred(ad_test: AnnData, label_col: str, pred_col: str, out_png: str) -> None:
    X_key = "X_scVI" if "X_scVI" in ad_test.obsm_keys() else ("X_pca" if "X_pca" in ad_test.obsm_keys() else None)
    ad = ad_test.copy()
    if X_key is not None:
        sc.pp.neighbors(ad, use_rep=X_key)
    else:
        sc.pp.pca(ad)
        sc.pp.neighbors(ad)
    sc.tl.umap(ad, random_state=0)
    with plt.rc_context({"figure.figsize": (10, 4)}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sc.pl.umap(ad, color=label_col, ax=axes[0], show=False, title="TEST: True labels")
        sc.pl.umap(ad, color=pred_col,  ax=axes[1], show=False, title="TEST: Predicted labels")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_per_class_metrics(df_report: pd.DataFrame, out_png: str) -> None:
    df = df_report.copy()
    df = df.loc[~df.index.isin(["accuracy", "macro avg", "weighted avg"])]
    if df.empty:
        return
    df_plot = df[["precision", "recall", "f1-score"]].sort_values("f1-score", ascending=False)
    df_plot.plot(kind="bar", figsize=(10, 4), rot=45)
    plt.title("Per-class metrics (TEST)")
    plt.ylabel("score")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_summary_metrics(df_report: pd.DataFrame, out_png: str) -> None:
    rows = df_report.loc[["accuracy", "macro avg", "weighted avg"], ["precision", "recall", "f1-score"]]
    rows = rows.copy()
    rows.index = ["accuracy", "macro", "weighted"]
    rows.plot(kind="bar", figsize=(6, 4))
    plt.title("Summary metrics (TEST)")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def compute_and_plot_pr_curves(y_true: pd.Series, y_proba_df: pd.DataFrame, classes_for_metrics: List[str], outdir: str) -> pd.DataFrame:
    ensure_dir(os.path.join(outdir, "pr_curves"))
    ap_rows = []
    for cls in classes_for_metrics:
        y_bin = (y_true == cls).astype(int)
        scores = y_proba_df.get(cls, pd.Series(np.zeros(len(y_true), dtype=float)))
        try:
            precision, recall, _ = precision_recall_curve(y_bin, scores)
            ap = average_precision_score(y_bin, scores)
        except Exception:
            precision, recall, ap = np.array([1.0]), np.array([0.0]), 0.0
        plt.figure(figsize=(5, 4))
        plt.plot(recall, precision, lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR curve: {cls} (AP={ap:.3f})")
        plt.tight_layout()
        out_png = os.path.join(outdir, "pr_curves", f"PR_{cls}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        ap_rows.append({"class": cls, "average_precision": ap})
    ap_df = pd.DataFrame(ap_rows).sort_values("average_precision", ascending=False)
    ap_df.to_csv(os.path.join(outdir, "average_precision_by_class.csv"), index=False)
    return ap_df


def dataframe_to_confusion_and_counts(y_true: pd.Series, y_pred: pd.Series, label_order: List[str]):
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
    true_counts = y_true.value_counts().reindex(label_order, fill_value=0).rename("true_count")
    pred_counts = y_pred.value_counts().reindex(label_order, fill_value=0).rename("pred_count")
    return cm_df, true_counts.to_frame(), pred_counts.to_frame()


# ------------------------- Training / Inference -------------------------

def train_scvi_scanvi_on_train(
    ad_train: AnnData,
    label_col: str,
    batch_col: str,
    counts_layer: str,
    n_latent: int = 30,
    scvi_epochs: int = 200,
    scanvi_epochs: int = 20,
    n_samples_per_label: int = 100,
    unknown_label: str = "Unknown",
    seed: int = 0
) -> SCANVI:
    set_all_seeds(seed)

    # Setup and train SCVI using counts layer
    X_backup = None
    try:
        SCVI.setup_anndata(ad_train, batch_key=batch_col, labels_key=label_col, layer=counts_layer)
    except TypeError:
        # Older scvi-tools: temporarily move counts to X
        X_backup = ad_train.X
        ad_train.X = ad_train.layers[counts_layer]
        SCVI.setup_anndata(ad_train, batch_key=batch_col, labels_key=label_col)

    scvi_model = SCVI(ad_train, n_latent=n_latent, gene_likelihood="nb")
    scvi_model.train(max_epochs=scvi_epochs)

    scanvi_model = SCANVI.from_scvi_model(scvi_model, unlabeled_category=unknown_label)
    scanvi_model.train(max_epochs=scanvi_epochs, n_samples_per_label=n_samples_per_label)

    # Restore X if we swapped
    if X_backup is not None:
        ad_train.X = X_backup

    return scanvi_model


def map_and_predict_on_test(
    scanvi_model: SCANVI,
    ad_test: AnnData,
    label_col: str,
    batch_col: str,
    counts_layer: str,
    unknown_label: str = "Unknown",
    seed: int = 0
):
    set_all_seeds(seed)

    # (a) Backup de verdad-terreno y forzar Unknown en el query
    if "_true_labels_backup" not in ad_test.obs.columns:
        ad_test.obs["_true_labels_backup"] = ad_test.obs[label_col].astype(str).values

    # construir categorías del modelo (sin extender)
    train_cats = list(scanvi_model.adata_manager.get_state_registry("labels").categorical_mapping)
    if unknown_label not in train_cats:
        train_cats = [unknown_label] + train_cats
    ad_test.obs[label_col] = pd.Categorical([unknown_label] * ad_test.n_obs, categories=train_cats)

    # (b) Intento preferente: load_query_data (sin extend_categories)
    try:
        qmodel = SCANVI.load_query_data(ad_test, scanvi_model)  # no extiende categorías
        qmodel.train(max_epochs=0)
        used_model = qmodel
    except Exception:
        # (c) Fallback: configurar anndata del TEST con labels_key + unlabeled_category
        try:
            SCANVI.setup_anndata(
                ad_test, batch_key=batch_col, labels_key=label_col,
                unlabeled_category=unknown_label, layer=counts_layer
            )
        except TypeError:
            _Xb = ad_test.X
            ad_test.X = ad_test.layers[counts_layer]
            SCANVI.setup_anndata(
                ad_test, batch_key=batch_col, labels_key=label_col,
                unlabeled_category=unknown_label
            )
            ad_test.X = _Xb
        used_model = scanvi_model

    # Predicciones
    probs = used_model.predict(ad_test, soft=True)
    cats = used_model.adata_manager.get_state_registry("labels").categorical_mapping
    y_proba_df = pd.DataFrame(probs, columns=list(cats))
    y_pred_cls = y_proba_df.idxmax(axis=1)
    pred_score = y_proba_df.max(axis=1)

    # Latente para UMAP
    try:
        ad_test.obsm["X_scVI"] = used_model.get_latent_representation(ad_test)
    except Exception:
        pass

    return y_pred_cls, pred_score, y_proba_df, ad_test


# ------------------------- Fold Runner -------------------------

def run_fold(fold_id: int, adata: AnnData, train_idx, test_idx, cfg, outdir_fold: str) -> None:
    ensure_dir(outdir_fold)

    label_col = cfg.label_col
    batch_col = cfg.batch_col
    unknown_label = cfg.unknown_label

    ad_train = adata[train_idx].copy()
    ad_test = adata[test_idx].copy()

    ad_train, ad_test = align_by_gene_intersection(ad_train, ad_test)

    # HVGs on TRAIN using counts layer (fallback moves counts to X if needed)
    try:
        sc.pp.highly_variable_genes(ad_train, flavor="seurat_v3", n_top_genes=2000, batch_key=batch_col, layer=cfg.counts_layer)
        hv = ad_train.var["highly_variable"].values
    except TypeError:
        _X_train, _X_test = ad_train.X, ad_test.X
        ad_train.X = ad_train.layers[cfg.counts_layer]
        ad_test.X = ad_test.layers[cfg.counts_layer]
        sc.pp.highly_variable_genes(ad_train, flavor="seurat_v3", n_top_genes=2000, batch_key=batch_col)
        hv = ad_train.var["highly_variable"].values
        ad_train.X, ad_test.X = _X_train, _X_test

    ad_train = ad_train[:, hv].copy()
    ad_test = ad_test[:, ad_train.var_names].copy()

    scanvi_model = train_scvi_scanvi_on_train(
        ad_train, label_col, batch_col, counts_layer=cfg.counts_layer,
        n_latent=cfg.n_latent, scvi_epochs=cfg.scvi_epochs, scanvi_epochs=cfg.scanvi_epochs,
        n_samples_per_label=cfg.n_samples_per_label, unknown_label=unknown_label, seed=cfg.seed
    )

    y_pred_cls, pred_score, y_proba_df, ad_test = map_and_predict_on_test(
        scanvi_model, ad_test, label_col, batch_col, counts_layer=cfg.counts_layer,
        unknown_label=unknown_label, seed=cfg.seed
    )

    y_true = ad_test.obs["_true_labels_backup"].astype(str)
    train_seen_labels = ad_train.obs[label_col].astype(str).unique().tolist()
    y_true_recoded, oott_rate, unseen_lbls = compute_unseen_and_oott(y_true, train_seen_labels, unknown_label)

    with open(os.path.join(outdir_fold, "unseen_labels.txt"), "w", encoding="utf-8") as f:
        for lbl in unseen_lbls:
            f.write(f"{lbl}\n")
    pd.DataFrame({"out_of_taxonomy_rate": [oott_rate]}).to_csv(os.path.join(outdir_fold, "oott_rate.csv"), index=False)
    if len(unseen_lbls) > 0:
        pd.DataFrame({"CellType_true": unseen_lbls, "n_cells": [int((y_true == l).sum()) for l in unseen_lbls]}).to_csv(
            os.path.join(outdir_fold, "unseen_labels_recoded_to_unknown.csv"), index=False
        )

    ad_test.obs["CellType_pred"] = y_pred_cls.values
    ad_test.obs["pred_score"] = pred_score.values

    label_order = sorted(set(train_seen_labels + [unknown_label]))
    cm_df, true_counts_df, pred_counts_df = dataframe_to_confusion_and_counts(y_true_recoded, y_pred_cls, label_order)
    cm_df.to_csv(os.path.join(outdir_fold, "confusion_table.csv"))
    true_counts_df.to_csv(os.path.join(outdir_fold, "true_counts.csv"))
    pred_counts_df.to_csv(os.path.join(outdir_fold, "predicted_counts.csv"))

    support = y_true_recoded.value_counts()
    classes_for_metrics = [c for c in label_order if support.get(c, 0) >= cfg.min_test_support]

    # Mask low-support to Unknown for report
    mask_low = ~y_true_recoded.isin(classes_for_metrics)
    y_true_eval = y_true_recoded.where(~mask_low, other=unknown_label)
    y_pred_eval = y_pred_cls.where(~mask_low, other=unknown_label)

    report = classification_report(y_true_eval, y_pred_eval, labels=label_order, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(outdir_fold, "classification_report.csv"))

    pred_score_summary = pd.DataFrame({
        "pred_score_mean": [pred_score.mean()],
        "pred_score_median": [pred_score.median()],
        "pred_score_p10": [pred_score.quantile(0.10)],
        "pred_score_p90": [pred_score.quantile(0.90)]
    })
    pred_score_summary.to_csv(os.path.join(outdir_fold, "pred_score_summary.csv"), index=False)
    plot_pred_score_hist(pred_score.values, os.path.join(outdir_fold, "pred_score_hist.png"))

    try:
        plot_umap_true_vs_pred(ad_test, label_col, "CellType_pred", os.path.join(outdir_fold, "umap_test_true_vs_pred.png"))
    except Exception as e:
        warnings.warn(f"UMAP plotting failed: {e}")

    try:
        plot_per_class_metrics(report_df, os.path.join(outdir_fold, "per_class_metrics.png"))
        plot_summary_metrics(report_df, os.path.join(outdir_fold, "summary_metrics.png"))
    except Exception as e:
        warnings.warn(f"Summary plotting failed: {e}")

    try:
        _ = compute_and_plot_pr_curves(y_true_recoded, y_proba_df, classes_for_metrics, outdir_fold)
    except Exception as e:
        warnings.warn(f"PR curves failed: {e}")

    ad_train.write(os.path.join(outdir_fold, "train.h5ad"))
    ad_test.write(os.path.join(outdir_fold, "test_predicted.h5ad"))


# ------------------------- Main -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-validation for diseased HCC using SCVI/SCANVI (counts layer).")
    p.add_argument("--adata_path", required=True, type=str, help="Path to diseased AnnData .h5ad")
    p.add_argument("--outdir", required=True, type=str, help="Output directory base")
    p.add_argument("--label_col", default="CellType", type=str, help="Label column (ground truth)")
    p.add_argument("--batch_col", default="Batch", type=str, help="Batch/dataset column for SCVI")
    p.add_argument("--group_col", default="Sample", type=str, help="Grouping column for blocking (Sample)")
    p.add_argument("--canon_map", default="", type=str, help="Path to canonical map JSON")
    p.add_argument("--unknown_label", default="Unknown", type=str, help="Unknown label name")
    p.add_argument("--n_splits", default=4, type=int, help="Number of CV folds")
    p.add_argument("--seed", default=0, type=int, help="Random seed")

    # Support thresholds
    p.add_argument("--global_min_cells_total", default=150, type=int, help="Min total cells to keep a class globally")
    p.add_argument("--global_min_groups", default=3, type=int, help="Min distinct groups(Sample) globally")
    p.add_argument("--min_test_support", default=30, type=int, help="Min test cells in fold to compute per-class PR/F1")

    # Counts layer
    p.add_argument("--counts_layer", default="counts", type=str, help="Layer name with raw counts for SCVI/SCANVI and HVGs")

    # SCVI/SCANVI knobs
    p.add_argument("--n_latent", default=30, type=int)
    p.add_argument("--scvi_epochs", default=200, type=int)
    p.add_argument("--scanvi_epochs", default=20, type=int)
    p.add_argument("--n_samples_per_label", default=100, type=int)

    return p.parse_args()


def main():
    cfg = parse_args()
    ensure_dir(cfg.outdir)
    set_all_seeds(cfg.seed)

    adata = sc.read_h5ad(cfg.adata_path)
    standardize_varnames(adata)

    # verify columns
    for col in [cfg.label_col, cfg.batch_col, cfg.group_col]:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs. Available: {list(adata.obs.columns)}")

    # verify counts layer
    if cfg.counts_layer not in (adata.layers.keys() if adata.layers is not None else []):
        raise ValueError(f"Counts layer '{cfg.counts_layer}' not found. Available layers: {list(adata.layers.keys())}")

    canon_map = load_canonical_map(cfg.canon_map)
    apply_canonical_map(adata, cfg.label_col, canon_map, report_dir=cfg.outdir)
    sanitize_labels_for_scanvi(adata, cfg.label_col, unknown_label=cfg.unknown_label)

    kept, rare = filter_classes_by_support(adata, cfg.label_col, cfg.group_col,
                                           min_cells_total=cfg.global_min_cells_total,
                                           min_groups=cfg.global_min_groups)

    supp = pd.DataFrame({
        "class": adata.obs[cfg.label_col].astype(str).value_counts().index,
        "n_cells": adata.obs[cfg.label_col].astype(str).value_counts().values,
    })
    supp["n_groups"] = adata.obs.groupby(cfg.label_col)[cfg.group_col].nunique().reindex(supp["class"]).values
    supp.to_csv(os.path.join(cfg.outdir, "global_class_support.csv"), index=False)
    pd.DataFrame({"kept": kept}).to_csv(os.path.join(cfg.outdir, "kept_classes.csv"), index=False)
    pd.DataFrame({"rare": rare}).to_csv(os.path.join(cfg.outdir, "rare_classes.csv"), index=False)

    labels = adata.obs[cfg.label_col].astype(str)
    groups = adata.obs[cfg.group_col].astype(str)
    splits = safe_stratified_group_kfold(labels, groups, n_splits=cfg.n_splits, seed=cfg.seed)

    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        outdir_fold = os.path.join(cfg.outdir, f"fold{i}_DiseasedCV")
        run_fold(i, adata, train_idx, test_idx, cfg, outdir_fold)

    # aggregate summary
    agg_rows = []
    for i in range(1, len(splits)+1):
        rep_path = os.path.join(cfg.outdir, f"fold{i}_DiseasedCV", "classification_report.csv")
        if os.path.exists(rep_path):
            df = pd.read_csv(rep_path, index_col=0)
            row = {"fold": i}
            for k in ["accuracy", "macro avg", "weighted avg"]:
                if k in df.index:
                    row[f"{k}_precision"] = df.loc[k, "precision"]
                    row[f"{k}_recall"] = df.loc[k, "recall"]
                    row[f"{k}_f1"] = df.loc[k, "f1-score"]
            agg_rows.append(row)
    if len(agg_rows) > 0:
        pd.DataFrame(agg_rows).sort_values("fold").to_csv(os.path.join(cfg.outdir, "aggregate_summary.csv"), index=False)

    print("[INFO] Completed Diseased CV. Outputs at:", cfg.outdir)


if __name__ == "__main__":
    main()
