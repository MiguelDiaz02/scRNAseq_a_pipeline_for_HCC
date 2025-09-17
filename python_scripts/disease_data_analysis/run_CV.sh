python UPDATED_Diseased_cross_validation.py \
  --adata_path /home/mdiaz/HCC_project/hcc_adata/1_5T_unintegrated_CellTypeFromPred.h5ad \
  --outdir /home/mdiaz/HCC_project/HCC_analysis/Diseased_CV \
  --label_col CellType \
  --batch_col Batch \
  --group_col Sample \
  --canon_map /home/mdiaz/HCC_project/HCC_analysis/canonical_map_hcc.json \
  --n_splits 4 \
  --seed 0
