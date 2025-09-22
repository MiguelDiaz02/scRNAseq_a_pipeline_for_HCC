#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search for Leiden on scVI latent; saves metrics and best result.

Outputs (in --outdir):
- grid_metrics.csv                  : full grid (metric, k, resolution, ARI, silhouette, n_clusters)
- best_config.json                  : best row (tie-break: ARI then silhouette)
- ari_by_batch.csv / ari_by_patient.csv (if columns exist)
- contingency.csv                   : crosstab(best_cluster, label_col)
- cluster_purity.csv                : per-cluster purity against label_col
- adata_with_scvi_best.h5ad         : AnnData with X_scVI and obs['leiden_best']
- (optional) scvi_model/            : saved scVI model (if --save-model)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 0):
    import torch, random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def silhouette_safe(X, labels, metric="euclidean", max_n=5000, seed=0):
    """Compute silhouette on a random subset (for scalability)."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    n = X.shape[0]
    if n > max_n:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=max_n, replace=False)
        Xs = X[sel]
        ys = labels[sel]
    else:
        Xs, ys = X, labels
    # If < 2 clusters, silhouette is undefined; return NaN
    if len(np.unique(ys)) < 2:
        return float("nan")
    return float(silhouette_score(Xs, ys, metric=metric))

def ari_by(ad, group_col, ref_col, clu_col):
    rows = []
    if group_col not in ad.obs.columns:
        return pd.DataFrame(columns=[group_col, "ARI", "n_cells", "n_ref", "n_clu"])
    for g, idx in ad.obs.groupby(group_col, observed=False).groups.items():
        sub = ad.obs.loc[idx, [ref_col, clu_col]].dropna()
        if sub.empty:
            continue
        if sub[ref_col].nunique() > 1 and sub[clu_col].nunique() > 1:
            ari = adjusted_rand_score(sub[ref_col].astype(str), sub[clu_col].astype(str))
            rows.append((g, ari, len(sub), sub[ref_col].nunique(), sub[clu_col].nunique()))
    return pd.DataFrame(rows, columns=[group_col, "ARI", "n_cells", "n_ref", "n_clu"]).sort_values("ARI", ascending=False)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Leiden grid on scVI latent with ARI/Silhouette evaluation.")
    ap.add_argument("--input", required=True, help=".h5ad file with counts layer and labels")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--layer", default="counts", help="Layer with raw counts for scVI (default: counts)")
    ap.add_argument("--batch-col", default="Batch", help="Batch column in .obs (default: Batch)")
    ap.add_argument("--label-col", default="predicted", help="Reference labels column (default: predicted)")
    ap.add_argument("--n-latent", type=int, default=30, help="scVI latent size (default: 30)")
    ap.add_argument("--epochs", type=int, default=100, help="scVI training epochs (default: 100)")
    ap.add_argument("--neighbors", type=int, nargs="+", default=[10, 15, 30, 50], help="k for KNN graph")
    ap.add_argument("--res", type=float, nargs="+", default=[round(x,1) for x in np.arange(0.2, 3.2, 0.2)], help="Leiden resolutions")
    ap.add_argument("--metrics", type=str, nargs="+", default=["cosine", "euclidean"], help="Neighbor metrics")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--save-model", action="store_true", help="Save scVI model directory")
    ap.add_argument("--sil-max-n", type=int, default=5000, help="Max cells for silhouette subsampling")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    set_seed(args.seed)

    # Load data
    ad = sc.read_h5ad(args.input)
    if args.layer not in ad.layers:
        raise ValueError(f"Layer '{args.layer}' not found. Available: {list(ad.layers.keys())}")
    if args.batch_col not in ad.obs.columns:
        raise ValueError(f"Batch column '{args.batch_col}' not found in .obs")
    if args.label_col not in ad.obs.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in .obs")

    # Train scVI
    scvi.model.SCVI.setup_anndata(ad, layer=args.layer, batch_key=args.batch_col)
    model = scvi.model.SCVI(ad, n_latent=args.n_latent)
    model.train(max_epochs=args.epochs, plan_kwargs={"weight_decay": 0.0},
                check_val_every_n_epoch=None, enable_progress_bar=False)
    ad.obsm["X_scVI"] = model.get_latent_representation()

    if args.save_model:
        model_dir = os.path.join(args.outdir, "scvi_model")
        ensure_dir(model_dir)
        model.save(model_dir, overwrite=True)

    rep = "X_scVI"
    results = []

    y_ref = ad.obs[args.label_col].astype(str)

    # Grid
    for metric in args.metrics:
        for k in args.neighbors:
            sc.pp.neighbors(ad, use_rep=rep, n_neighbors=int(k), metric=metric)
            for r in args.res:
                key = f"leiden_{metric}_{k}_{r:.1f}"
                sc.tl.leiden(ad, resolution=float(r), key_added=key)
                y_clu = ad.obs[key].astype(str)
                ari = float(adjusted_rand_score(y_ref, y_clu))
                sil = silhouette_safe(ad.obsm[rep], y_clu, metric="euclidean", max_n=args.sil_max_n, seed=args.seed)
                n_clu = int(y_clu.nunique())
                results.append({
                    "metric": metric,
                    "k": int(k),
                    "resolution": float(r),
                    "ARI": ari,
                    "silhouette_euclid_on_scVI": sil,
                    "n_clusters": n_clu,
                    "key": key
                })

    # Save grid
    grid_df = pd.DataFrame(results).sort_values(["ARI", "silhouette_euclid_on_scVI"], ascending=[False, False])
    grid_csv = os.path.join(args.outdir, "grid_metrics.csv")
    grid_df.to_csv(grid_csv, index=False)

    # Pick best (by ARI, tie-break by silhouette)
    best_row = grid_df.iloc[0].to_dict()
    best_key = best_row["key"]
    ad.obs["leiden_best"] = ad.obs[best_key].astype(str)

    # ARI by batch/patient (if present)
    ari_batch = ari_by(ad, args.batch_col, args.label_col, "leiden_best")
    ari_batch.to_csv(os.path.join(args.outdir, "ari_by_batch.csv"), index=False)

    if "Patient" in ad.obs.columns:
        ari_patient = ari_by(ad, "Patient", args.label_col, "leiden_best")
        ari_patient.to_csv(os.path.join(args.outdir, "ari_by_patient.csv"), index=False)

    # Contingency and purity
    ct = pd.crosstab(ad.obs["leiden_best"], ad.obs[args.label_col])
    ct.to_csv(os.path.join(args.outdir, "contingency.csv"))
    purity = (ct.max(1) / ct.sum(1)).rename("purity").to_frame()
    purity.to_csv(os.path.join(args.outdir, "cluster_purity.csv"))

    # Save best config and AnnData
    with open(os.path.join(args.outdir, "best_config.json"), "w") as fh:
        json.dump(best_row, fh, indent=2)

    out_h5ad = os.path.join(args.outdir, "adata_with_scvi_best.h5ad")
    ad.write(out_h5ad, compression="gzip")

    print("\n=== DONE ===")
    print(f"Best: metric={best_row['metric']}, k={best_row['k']}, res={best_row['resolution']:.1f}, "
          f"ARI={best_row['ARI']:.3f}, silhouette={best_row['silhouette_euclid_on_scVI']:.3f}")
    print(f"Saved:\n  {grid_csv}\n  best_config.json\n  ari_by_batch.csv\n  ari_by_patient.csv (if present)\n"
          f"  contingency.csv\n  cluster_purity.csv\n  {out_h5ad}")

if __name__ == "__main__":
    main()
