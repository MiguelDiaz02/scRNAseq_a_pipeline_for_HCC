#!/usr/bin/env bash
# Run ARI clustering grid search in the background.
# Override defaults by setting environment variables before calling this script:
#   INPUT_H5AD  — path to input .h5ad  (default below)
#   OUTPUT_DIR  — output directory      (default below)
#   LOG_FILE    — path for nohup log    (default: OUTPUT_DIR/run.log)

INPUT_H5AD="${INPUT_H5AD:-/home/mdiaz/HCC_project/hcc_adata/1_5T_unintegrated_CellTypeFromPred.h5ad}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/mdiaz/HCC_project/hcc_adata}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/run.log}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nohup python "${SCRIPT_DIR}/ari_clustering.py" \
  --input "${INPUT_H5AD}" \
  --outdir "${OUTPUT_DIR}" \
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
  > "${LOG_FILE}" 2>&1 &

echo "Job started (PID $!). Follow progress with:"
echo "  tail -f ${LOG_FILE}"
