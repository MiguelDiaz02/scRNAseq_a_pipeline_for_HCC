nohup python ARI_clustering_analysis.py \
  --input /home/mdiaz/HCC_project/hcc_adata/1_5T_unintegrated_CellTypeFromPred.h5ad \
  --outdir /home/mdiaz/HCC_project/hcc_adata/ \
  --layer counts \
  --batch-col Batch \
  --label-col predicted \
  --n-latent 30 \
  --epochs 100 \
  --neighbors 10 15 30 50 \
  --metrics cosine euclidean \
  --res 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 \
  --save-model \
  --seed 0 \
  > /home/mdiaz/HCC_project/hcc_adata/run.log 2>&1 &
# Ver progreso:
tail -f /home/mdiaz/HCC_project/hcc_adata/run.log
