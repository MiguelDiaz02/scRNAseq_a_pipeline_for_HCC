#!/usr/bin/env bash
set -euo pipefail

# Activate conda env
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate new_scKernel

INPUT_H5AD="./HCC_project/healthy_adata/1_5C_unintegrated_countsX.h5ad"
OUT_DIR="./CV_healthy_data"

LABEL_COL="predicted"    
BATCH_COL="Batch"
SAMPLE_COL="Sample"
CANONICAL_MAP="./canonical_map_healthy.json" 

N_SPLITS=5
SEED=0
N_LATENT=30
EPOCHS_SCVI=200
EPOCHS_SCANVI=20
N_SAMPLES_PER_LABEL=100
SCORE_THRESHOLD=0.8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/cross_validation.py"

mkdir -p "${OUT_DIR}"
STAMP="$(date +'%Y%m%d_%H%M%S')"
LOGFILE="${OUT_DIR}/run_${STAMP}.log"

echo "[INFO] Starting CV run at ${STAMP}" | tee -a "${LOGFILE}"

python "${PY_SCRIPT}" \
  --input-h5ad "${INPUT_H5AD}" \
  --out-dir "${OUT_DIR}" \
  --label-col "${LABEL_COL}" \
  --out-label-col CellType \
  --batch-col "${BATCH_COL}" \
  --sample-col "${SAMPLE_COL}" \
  --canonical-map-json "${CANONICAL_MAP}" \
  --n-splits "${N_SPLITS}" \
  --seed "${SEED}" \
  --n-latent "${N_LATENT}" \
  --max-epochs-scvi "${EPOCHS_SCVI}" \
  --max-epochs-scanvi "${EPOCHS_SCANVI}" \
  --n-samples-per-label "${N_SAMPLES_PER_LABEL}" \
  --score-threshold "${SCORE_THRESHOLD}" 2>&1 | tee -a "${LOGFILE}"

echo "[INFO] Finished. Logs: ${LOGFILE}"
