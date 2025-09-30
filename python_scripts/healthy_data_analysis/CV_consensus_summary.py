
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consensus CV summarizer for Healthy donors.
See header comments for usage.
"""
import argparse
import json
import logging
import os
import sys
from glob import glob
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import anndata as ad
except Exception:
    ad = None


def parse_args():
    p = argparse.ArgumentParser(description="Build consensus summary from CV folds (Healthy).")
    p.add_argument("--cv_outdir", required=True, help="Directory containing fold directories (e.g., fold1_HealthyCV).")
    p.add_argument("--fold_glob", default="fold*_HealthyCV", help="Glob to discover folds inside cv_outdir.")
    p.add_argument("--export_prefix", default="HealthyCV_", help="Prefix for output file names.")
    p.add_argument("--min_test_support", type=int, default=10, help="Min test support threshold for flags.")
    p.add_argument("--class_order_csv", default=None, help="Optional CSV with a column 'class' defining order.")
    p.add_argument("--collect_pred_scores", default="false", choices=["true", "false"],
                   help="If 'true', will read test_predicted.h5ad per fold to collect per-cell pred_score.")
    p.add_argument("--fig_dpi", type=int, default=300, help="DPI for output figures.")
    args = p.parse_args()
    args.collect_pred_scores = args.collect_pred_scores.lower() == "true"
    return args


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def ensure_outdir(base_dir: str) -> str:
    outdir = os.path.join(base_dir, "consensus")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def discover_folds(cv_outdir: str, fold_glob: str) -> List[str]:
    pattern = os.path.join(cv_outdir, fold_glob)
    candidates = sorted(glob(pattern))
    if not candidates:
        logging.error("No fold directories found matching %s", pattern)
        sys.exit(1)
    logging.info("Discovered %d folds.", len(candidates))
    return candidates


def read_csv_safe(path: str, **kwargs) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            logging.warning("Failed to read %s: %s", path, e)
            return None
    else:
        logging.warning("Missing expected file: %s", path)
        return None


def read_classification_report(fold_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(fold_dir, "classification_report.csv")
    df = read_csv_safe(path, index_col=0)
    if df is not None:
        cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=cols)
        if "f1-score" in df.columns and "f1" not in df.columns:
            df = df.rename(columns={"f1-score": "f1"})
    return df


def extract_global_metrics(rep: pd.DataFrame, fold_id: str) -> pd.DataFrame:
    rows = []
    if rep is None:
        return pd.DataFrame()
    rep = rep.copy()
    rep.index = rep.index.astype(str)

    if "accuracy" in rep.index:
        acc_row = rep.loc["accuracy"]
        if "accuracy" in rep.columns:
            acc = acc_row.get("accuracy", np.nan)
        else:
            acc = acc_row.get("precision", np.nan)
    else:
        acc = np.nan

    macro = rep.loc["macro avg"] if "macro avg" in rep.index else pd.Series(dtype=float)
    weighted = rep.loc["weighted avg"] if "weighted avg" in rep.index else pd.Series(dtype=float)
    rows.append({
        "fold": fold_id,
        "accuracy": acc,
        "macro_precision": macro.get("precision", np.nan),
        "macro_recall": macro.get("recall", np.nan),
        "macro_f1": macro.get("f1", macro.get("f1-score", np.nan)),
        "weighted_precision": weighted.get("precision", np.nan),
        "weighted_recall": weighted.get("recall", np.nan),
        "weighted_f1": weighted.get("f1", weighted.get("f1-score", np.nan)),
    })
    return pd.DataFrame(rows)


def extract_per_class(rep: pd.DataFrame, fold_id: str) -> pd.DataFrame:
    if rep is None:
        return pd.DataFrame()
    rep = rep.copy()
    rep.index = rep.index.astype(str)
    drop_rows = {"accuracy", "macro avg", "weighted avg"}
    df = rep[~rep.index.isin(drop_rows)].reset_index().rename(columns={"index": "class"})
    if "f1-score" in df.columns and "f1" not in df.columns:
        df = df.rename(columns={"f1-score": "f1"})
    for k in ["precision", "recall", "f1", "support"]:
        if k not in df.columns:
            df[k] = np.nan
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["fold"] = fold_id
    return df[["fold", "class", "precision", "recall", "f1", "support"]]


def read_ap_by_class(fold_dir: str, fold_id: str) -> pd.DataFrame:
    path = os.path.join(fold_dir, "average_precision_by_class.csv")
    df = read_csv_safe(path)
    if df is None or df.empty:
        return pd.DataFrame()
    if "class" not in df.columns:
        df = df.rename(columns={df.columns[0]: "class"})
    if "AP" not in df.columns and "average_precision" in df.columns:
        df = df.rename(columns={"average_precision": "AP"})
    if "AP" not in df.columns and "ap" in df.columns:
        df = df.rename(columns={"ap": "AP"})
    df["fold"] = fold_id
    df["AP"] = pd.to_numeric(df["AP"], errors="coerce")
    return df[["fold", "class", "AP"]]


def read_oott(fold_dir: str, fold_id: str) -> pd.DataFrame:
    path = os.path.join(fold_dir, "oott_rate.csv")
    df = read_csv_safe(path)
    if df is None or df.empty:
        return pd.DataFrame()
    df["fold"] = fold_id
    return df


def read_confusion(fold_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(fold_dir, "confusion_table.csv")
    df = read_csv_safe(path, index_col=0)
    return df


def summarize_numeric(df: pd.DataFrame, value_cols: List[str], by: List[str]) -> pd.DataFrame:
    agg = {c: ["mean", "std", "median"] for c in value_cols}
    out = df.groupby(by).agg(agg)
    out.columns = ["{}_{}".format(v, stat) for v, stat in out.columns]
    return out.reset_index()


def label_stability(row, f1_mean_cut=0.80, f1_sd_cut=0.05, support_total_cut=0):
    f1m = row.get("f1_mean", np.nan)
    f1sd = row.get("f1_std", np.nan)
    support = row.get("support_total", 0)
    if pd.notna(f1m) and f1m >= f1_mean_cut and (pd.isna(f1sd) or f1sd <= f1_sd_cut):
        return "stable"
    if pd.notna(f1m) and f1m < 0.60:
        return "problematic"
    if support_total_cut and support < support_total_cut:
        return "problematic"
    return "intermediate"


def row_normalize(cm: pd.DataFrame) -> pd.DataFrame:
    cm = cm.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        sums = cm.sum(axis=1).replace(0, np.nan)
        out = cm.div(sums, axis=0)
    return out.fillna(0.0)


def top_confusions(cm_row: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    records = []
    for i in cm_row.index:
        for j in cm_row.columns:
            if i == j:
                continue
            v = cm_row.loc[i, j]
            if v > 0:
                records.append((i, j, float(v)))
    df = pd.DataFrame(records, columns=["true", "pred", "row_norm"])
    df = df.sort_values("row_norm", ascending=False).head(top_k).reset_index(drop=True)
    return df


def symmetric_confusion(cm_row: pd.DataFrame) -> pd.DataFrame:
    labels = list(cm_row.index)
    recs = []
    for a in labels:
        for b in labels:
            if a >= b:
                continue
            vab = cm_row.loc[a, b] if (a in cm_row.index and b in cm_row.columns) else 0.0
            vba = cm_row.loc[b, a] if (b in cm_row.index and a in cm_row.columns) else 0.0
            sym = 0.5 * (float(vab) + float(vba))
            if sym > 0:
                recs.append((a, b, sym))
    return pd.DataFrame(recs, columns=["class_A", "class_B", "sym_conf"]).sort_values("sym_conf", ascending=False)


def propose_merges(sym_df: pd.DataFrame, per_class_cons: pd.DataFrame, ap_cons: pd.DataFrame,
                   sym_cut=0.20, f1_cut=0.70, ap_cut=0.65) -> pd.DataFrame:
    """
    Build data-driven merge suggestions. Robust to the case where all pairs are filtered out.
    """
    empty_cols = ["class_A", "class_B", "sym_conf", "F1_A", "F1_B", "AP_A", "AP_B", "support_A", "support_B"]
    if sym_df is None or sym_df.empty:
        return pd.DataFrame(columns=empty_cols)
    f1map = per_class_cons.set_index("class")["f1_mean"].to_dict() if not per_class_cons.empty else {}
    supmap = per_class_cons.set_index("class")["support_total"].to_dict() if not per_class_cons.empty else {}
    apmap = ap_cons.set_index("class")["AP_mean"].to_dict() if (ap_cons is not None and not ap_cons.empty) else {}
    rows = []
    for _, r in sym_df.iterrows():
        a, b, sym = r["class_A"], r["class_B"], r["sym_conf"]
        if sym < sym_cut:
            continue
        f1a, f1b = f1map.get(a, np.nan), f1map.get(b, np.nan)
        apa, apb = apmap.get(a, np.nan), apmap.get(b, np.nan)
        sa, sb = supmap.get(a, np.nan), supmap.get(b, np.nan)
        if (pd.isna(f1a) or f1a < f1_cut) or (pd.isna(f1b) or f1b < f1_cut) or \
           (pd.isna(apa) or apa < ap_cut) or (pd.isna(apb) or apb < ap_cut):
            rows.append({
                "class_A": a, "class_B": b, "sym_conf": sym,
                "F1_A": f1a, "F1_B": f1b, "AP_A": apa, "AP_B": apb,
                "support_A": sa, "support_B": sb
            })
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    df_rows = pd.DataFrame(rows)
    if "sym_conf" not in df_rows.columns or df_rows.empty:
        return pd.DataFrame(columns=empty_cols)
    return df_rows.sort_values("sym_conf", ascending=False)


def plot_per_class_bar(df: pd.DataFrame, value_col: str, out_png: str, dpi: int = 300):
    if df.empty:
        return
    d = df.sort_values(value_col, ascending=True)
    labels = d["class"].astype(str).tolist()
    vals = d[value_col].astype(float).tolist()
    plt.figure()
    plt.barh(labels, vals)
    plt.xlabel(value_col)
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_confusion_heatmap(cm_row: pd.DataFrame, out_png: str, dpi: int = 300):
    if cm_row.empty:
        return
    arr = cm_row.to_numpy().astype(float)
    np.fill_diagonal(arr, np.nan)
    masked = np.ma.array(arr, mask=np.isnan(arr))
    plt.figure()
    im = plt.imshow(masked, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(cm_row.shape[1]), labels=cm_row.columns, rotation=90)
    plt.yticks(ticks=np.arange(cm_row.shape[0]), labels=cm_row.index)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def pred_score_hist(df_ps_all: pd.DataFrame, out_png: str, dpi: int = 300):
    if df_ps_all.empty:
        return
    plt.figure()
    plt.hist(df_ps_all["pred_score"].astype(float).values, bins=50)
    plt.xlabel("pred_score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def pred_score_violin(df_ps_all: pd.DataFrame, out_png: str, dpi: int = 300, max_classes: int = 40):
    if df_ps_all.empty:
        return
    grp = df_ps_all.groupby("CellType_pred")["pred_score"].apply(list)
    sizes = df_ps_all.groupby("CellType_pred").size().sort_values(ascending=False)
    keep = sizes.index[:max_classes].tolist()
    data = [grp.get(k, []) for k in keep]
    plt.figure()
    plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(ticks=np.arange(1, len(keep) + 1), labels=keep, rotation=90)
    plt.ylabel("pred_score")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def read_pred_scores_from_h5ad(fold_dir: str, label_pred: str = "predicted") -> pd.DataFrame:
    if ad is None:
        logging.warning("anndata not available; skip per-cell pred_score.")
        return pd.DataFrame()
    path = os.path.join(fold_dir, "test_predicted.h5ad")
    if not os.path.exists(path):
        logging.warning("Missing %s; skip per-cell pred_score.", path)
        return pd.DataFrame()
    try:
        X = ad.read_h5ad(path, backed=None)
        cols = X.obs.columns.astype(str).tolist()
        pred_col = label_pred if label_pred in cols else ("CellType_pred" if "CellType_pred" in cols else None)
        score_col = "pred_score" if "pred_score" in cols else None
        if pred_col is None or score_col is None:
            logging.warning("Columns %s/%s not found in obs; skip.", label_pred, "pred_score")
            return pd.DataFrame()
        df = X.obs[[pred_col, score_col]].copy()
        df = df.rename(columns={pred_col: "CellType_pred"})
        df["pred_score"] = pd.to_numeric(df["pred_score"], errors="coerce")
        df = df.dropna(subset=["pred_score"])
        return df.reset_index(drop=True)
    except Exception as e:
        logging.warning("Failed reading %s: %s", path, e)
        return pd.DataFrame()


def main():
    setup_logging()
    args = parse_args()
    base = args.cv_outdir
    outdir = ensure_outdir(base)
    logging.info("Consensus outputs will be written to: %s", outdir)

    folds = discover_folds(base, args.fold_glob)

    class_order = None
    if args.class_order_csv and os.path.exists(args.class_order_csv):
        co = pd.read_csv(args.class_order_csv)
        if "class" in co.columns:
            class_order = co["class"].astype(str).tolist()

    globals_tbl = []
    per_class_tbl = []
    ap_tbl = []
    oott_tbl = []
    pred_score_tbl = []
    unseen_records = []
    cm_sum = None

    for fold_dir in folds:
        fold_id = os.path.basename(fold_dir)
        logging.info("Processing %s", fold_id)

        rep = read_classification_report(fold_dir)
        if rep is not None:
            globals_tbl.append(extract_global_metrics(rep, fold_id))
            per_class_tbl.append(extract_per_class(rep, fold_id))

        ap_tbl.append(read_ap_by_class(fold_dir, fold_id))
        oott_tbl.append(read_oott(fold_dir, fold_id))

        unseen_txt = os.path.join(fold_dir, "unseen_labels.txt")
        if os.path.exists(unseen_txt):
            try:
                with open(unseen_txt, "r", encoding="utf-8") as fh:
                    for line in fh:
                        lab = line.strip()
                        if lab:
                            unseen_records.append({"fold": fold_id, "label": lab})
            except Exception as e:
                logging.warning("Failed reading %s: %s", unseen_txt, e)

        cm = read_confusion(fold_dir)
        if cm is not None and not cm.empty:
            cm = cm.copy()
            cm.index = cm.index.astype(str)
            cm.columns = cm.columns.astype(str)
            if cm_sum is None:
                cm_sum = cm
            else:
                idx = sorted(set(cm_sum.index).union(cm.index))
                cols = sorted(set(cm_sum.columns).union(cm.columns))
                cm_sum = cm_sum.reindex(index=idx, columns=cols, fill_value=0)
                cm = cm.reindex(index=idx, columns=cols, fill_value=0)
                cm_sum = cm_sum.add(cm, fill_value=0)

        if args.collect_pred_scores:
            ps = read_pred_scores_from_h5ad(fold_dir, label_pred="predicted")
            if not ps.empty:
                ps["fold"] = fold_id
                pred_score_tbl.append(ps)

    if globals_tbl:
        df_globals = pd.concat(globals_tbl, ignore_index=True)
        df_globals.to_csv(os.path.join(outdir, f"{args.export_prefix}global_metrics_by_fold.csv"), index=False)
        val_cols = [c for c in df_globals.columns if c != "fold"]
        df_globals_summary = pd.DataFrame({
            "metric": val_cols,
            "mean": [pd.to_numeric(df_globals[c], errors="coerce").mean() for c in val_cols],
            "std": [pd.to_numeric(df_globals[c], errors="coerce").std() for c in val_cols],
            "median": [pd.to_numeric(df_globals[c], errors="coerce").median() for c in val_cols],
        })
        df_globals_summary.to_csv(os.path.join(outdir, f"{args.export_prefix}consensus_global_metrics_mean_sd.csv"), index=False)

    df_pc_cons = pd.DataFrame()
    if per_class_tbl:
        df_pc = pd.concat(per_class_tbl, ignore_index=True)
        sum_pc = summarize_numeric(df_pc, ["precision", "recall", "f1"], by=["class"])
        sup = df_pc.groupby("class")["support"].agg(["sum", "count"]).reset_index().rename(
            columns={"sum": "support_total", "count": "n_folds_present"}
        )
        df_pc_cons = sum_pc.merge(sup, on="class", how="outer")
        support_cut = args.min_test_support * max(1, len(folds) // 2)
        df_pc_cons["stability_label"] = df_pc_cons.apply(
            lambda r: label_stability(r, f1_mean_cut=0.80, f1_sd_cut=0.05, support_total_cut=support_cut), axis=1
        )
        if class_order:
            df_pc_cons["__ord"] = df_pc_cons["class"].apply(lambda c: class_order.index(c) if c in class_order else 1e9)
            df_pc_cons = df_pc_cons.sort_values(["__ord", "f1_mean"], ascending=[True, False]).drop(columns="__ord")
        else:
            df_pc_cons = df_pc_cons.sort_values("f1_mean", ascending=False)
        df_pc_cons.to_csv(os.path.join(outdir, f"{args.export_prefix}per_class_consensus.csv"), index=False)

    df_ap_cons = pd.DataFrame()
    if ap_tbl:
        df_ap_all = pd.concat([x for x in ap_tbl if x is not None and not x.empty], ignore_index=True) \
            if any([x is not None and not x.empty for x in ap_tbl]) else pd.DataFrame()
        if not df_ap_all.empty:
            df_ap_cons = summarize_numeric(df_ap_all, ["AP"], by=["class"])
            df_ap_cons.to_csv(os.path.join(outdir, f"{args.export_prefix}average_precision_consensus.csv"), index=False)

    if cm_sum is not None and not cm_sum.empty:
        if class_order:
            labels = [c for c in class_order if c in cm_sum.index]
            remaining = [c for c in cm_sum.index if c not in labels]
            new_idx = labels + remaining
            cm_sum = cm_sum.reindex(index=new_idx, columns=new_idx, fill_value=0)
        cm_sum.to_csv(os.path.join(outdir, f"{args.export_prefix}CM_sum.csv"))
        cm_row = row_normalize(cm_sum)
        cm_row.to_csv(os.path.join(outdir, f"{args.export_prefix}CM_row_norm.csv"))

        plot_confusion_heatmap(cm_row, os.path.join(outdir, f"{args.export_prefix}consensus_confusion_heatmap.png"), dpi=args.fig_dpi)

        topc = top_confusions(cm_row, top_k=20)
        topc.to_csv(os.path.join(outdir, f"{args.export_prefix}top_confusions.csv"), index=False)

        sym = symmetric_confusion(cm_row)
        sym.to_csv(os.path.join(outdir, f"{args.export_prefix}confusion_graph_edgelist.csv"), index=False)

        if not df_pc_cons.empty:
            merges = propose_merges(sym, df_pc_cons, df_ap_cons if not df_ap_cons.empty else pd.DataFrame())
            merges.to_csv(os.path.join(outdir, f"{args.export_prefix}merge_suggestions.csv"), index=False)

    if oott_tbl:
        oott_all = pd.concat([x for x in oott_tbl if x is not None and not x.empty], ignore_index=True) \
            if any([x is not None and not x.empty for x in oott_tbl]) else pd.DataFrame()
        if not oott_all.empty:
            num_cols = oott_all.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                summary = oott_all[num_cols].agg(["mean", "std", "median"]).T.reset_index().rename(columns={"index": "metric"})
                summary.to_csv(os.path.join(outdir, f"{args.export_prefix}oott_rate_summary.csv"), index=False)
            else:
                oott_all.to_csv(os.path.join(outdir, f"{args.export_prefix}oott_rate_raw.csv"), index=False)

    if unseen_records:
        df_unseen = pd.DataFrame(unseen_records)
        agg_unseen = df_unseen.groupby("label").size().reset_index(name="count_folds_seen")
        agg_unseen.to_csv(os.path.join(outdir, f"{args.export_prefix}unseen_labels_aggregated.csv"), index=False)

    if args.collect_pred_scores and pred_score_tbl:
        df_ps_all = pd.concat(pred_score_tbl, ignore_index=True)
        df_ps_all.to_csv(os.path.join(outdir, f"{args.export_prefix}pred_score_all_cells.csv"), index=False)
        ps_sum = df_ps_all.groupby("CellType_pred")["pred_score"].agg(["mean", "median", "std", "count"]).reset_index()
        ps_sum.to_csv(os.path.join(outdir, f"{args.export_prefix}pred_score_by_class_summary.csv"), index=False)
        pred_score_hist(df_ps_all, os.path.join(outdir, f"{args.export_prefix}pred_score_hist_global.png"), dpi=args.fig_dpi)
        pred_score_violin(df_ps_all, os.path.join(outdir, f"{args.export_prefix}pred_score_violin_by_pred.png"), dpi=args.fig_dpi)

    if not df_pc_cons.empty:
        plot_per_class_bar(df_pc_cons.rename(columns={"f1_mean": "F1_mean"}), "F1_mean",
                           os.path.join(outdir, f"{args.export_prefix}per_class_F1_bar.png"),
                           dpi=args.fig_dpi)
    if not (df_ap_cons is None or df_ap_cons.empty):
        plot_per_class_bar(df_ap_cons.rename(columns={"AP_mean": "AP_mean"}), "AP_mean",
                           os.path.join(outdir, f"{args.export_prefix}per_class_AP_bar.png"),
                           dpi=args.fig_dpi)

    cfg = {
        "cv_outdir": args.cv_outdir,
        "fold_glob": args.fold_glob,
        "export_prefix": args.export_prefix,
        "min_test_support": args.min_test_support,
        "class_order_csv": args.class_order_csv,
        "collect_pred_scores": args.collect_pred_scores,
        "fig_dpi": args.fig_dpi,
        "version": "1.0.0-healthy"
    }
    with open(os.path.join(outdir, f"{args.export_prefix}consensus_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    logging.info("Done. Outputs in: %s", outdir)


if __name__ == "__main__":
    main()