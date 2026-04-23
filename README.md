# HCC scRNA-seq Analysis Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Complete, reproducible single-cell RNA-seq analysis pipeline for hepatocellular carcinoma (HCC) and healthy liver specimens.**

## Overview

This pipeline integrates data from 5 public HCC/liver scRNA-seq datasets, performs comprehensive quality control, cell type annotation, batch correction, cell-cell communication analysis, cancer stem cell identification, and differential expression/enrichment analysis.

**Dataset:** 88,146 single cells from human HCC and healthy liver tissue spanning 5 public repositories (GEO GSE242889, GSE228195, GSE151530, GSE189903; CZ CELLxGENE).

**Citation:** Díaz-Campos, M. Á., et al. (2026). Single-Cell Transcriptomic Profiling Reveals Immunometabolic Reprogramming and Cell-Cell Communication in the Tumor Microenvironment of Human Hepatocellular Carcinoma. *IJMS*.

## Pipeline Architecture

```
DATA PROCESSING
├─ 01_qc                     Quality control & doublet removal (scVI+SOLO)
├─ 02_clustering             Leiden clustering optimization (ARI-driven)
├─ 03_annotation             Cell type annotation (CellTypist + scANVI)
│                            Cross-validation of classifiers
├─ 04_integration            scVI batch correction & post-hoc refinement
│
ANALYSIS
├─ 05_communication          LIANA+ consensus ligand-receptor interactions
├─ 06_csc                    Cancer stem cell identification & scoring
├─ 07_dgea                   Pseudobulk edgeR & Wilcoxon DE analysis
└─ 08_enrichment             GOBP/pathway enrichment (clusterProfiler)
```

**Execution order (mandatory):** `01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`

Each stage depends on outputs from previous stages.

## Installation

### Prerequisites

- Linux/macOS environment
- `conda` or `mamba` (recommended: [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- ~200 GB disk space (raw data + intermediate outputs)

### Data Setup

Download raw HCC/healthy scRNA-seq data from public repositories:

```bash
# Create data directory
mkdir -p ~/sc_liver_data/{hcc_h5ad_adata,healthy_h5ad_adata,ref_data}

# Download HCC samples (GEO GSE242889, GSE228195, GSE151530, GSE189903)
# Use SRA Toolkit (prefetch) or direct download via GEO
# Convert to .h5ad using Seurat's Convert() or scanpy's sc.read_mtx()

# Download healthy liver reference (CZ CELLxGENE Tabula Sapiens liver)
# Use 3ae74888-412b-4b5c-801b-841c04ad448f.h5ad

# Place all into appropriate subdirectories
```

### Conda Environments

Create isolated environments for each analysis stage:

```bash
cd envs

# Main analysis environment (Python 3.10)
conda env create -f hcc_main.yml
conda activate hcc_main

# LIANA+ specific environment (extra dependencies)
conda env create -f liana_env.yml

# R environment for GSEA/enrichment
conda env create -f r_gsea.yml
```

## Quick Start

Execute the pipeline end-to-end:

```bash
# Activate main environment
conda activate hcc_main

# 01. QC & preprocessing (HCC samples)
python 01_qc/hcc_preprocess.py

# 01. QC & preprocessing (Healthy samples)
python 01_qc/healthy_preprocess.py

# 02. Clustering optimization
bash 02_clustering/run_ari.sh

# 03. Cell type annotation (HCC)
python 03_annotation/hcc_annotation.py

# 03. Cell type annotation (Healthy)
python 03_annotation/healthy_annotation.py

# 03. Cross-validation & consensus
python 03_annotation/cross_validation_hcc.py
python 03_annotation/cross_validation_healthy.py
python 03_annotation/cv_consensus.py

# 04. Integration & refinement
python 04_integration/batch_harmonization.py
python 04_integration/annotation_corrections.py

# 05. Cell-cell communication
conda activate liana_env
python 05_communication/liana_ccc.py
conda activate hcc_main
Rscript 05_communication/cellchat_per_sample.R

# 06. CSC analysis
python 06_csc/csc_identification.py
python 06_csc/csc_figures.py

# 07. Differential expression
python 07_dgea/pseudobulk_edgeR.py
python 07_dgea/wilcoxon_dgea.py
python 07_dgea/extract_fig_degs.py

# 08. Enrichment analysis
conda activate r_gsea
Rscript 08_enrichment/GSEA_analysis_hepatocytes.R
Rscript 08_enrichment/GSEA_analysis_TAMs.R
Rscript 08_enrichment/GSEA_analysis_TECs.R
Rscript 08_enrichment/GSEA_analysis_MCvsHepatocytes.R
Rscript 08_enrichment/GSEA_clean_all.R
```

## Stage-by-Stage Documentation

Each pipeline stage has its own `README.md`:

- **[01_qc/README.md](01_qc/README.md)** — Quality control & filtering
- **[02_clustering/README.md](02_clustering/README.md)** — Clustering optimization
- **[03_annotation/README.md](03_annotation/README.md)** — Cell type annotation & validation
- **[04_integration/README.md](04_integration/README.md)** — Batch integration & refinement
- **[05_communication/README.md](05_communication/README.md)** — Ligand-receptor analysis
- **[06_csc/README.md](06_csc/README.md)** — Cancer stem cell characterization
- **[07_dgea/README.md](07_dgea/README.md)** — Differential expression analysis
- **[08_enrichment/README.md](08_enrichment/README.md)** — Pathway & functional enrichment

## Configuration & Customization

Each script has environment variable overrides at the top of the file:

```python
# Example: 01_qc/hcc_preprocess.py
INPUT_DIR = os.getenv('HCC_DATA_DIR', '/home/mdiaz/sc_liver_data/hcc_h5ad_adata')
OUTPUT_DIR = os.getenv('OUTPUT_DIR_HCC', '/home/mdiaz/HCC_project/hcc_adata')

# Override at runtime:
export HCC_DATA_DIR=/path/to/custom/data
export OUTPUT_DIR_HCC=/path/to/custom/output
python 01_qc/hcc_preprocess.py
```

## Expected Outputs

| Stage | Output | Size |
|-------|--------|------|
| 01 | `hcc_adata/0C_doub_remov.h5ad` | ~2.5 GB |
| 02 | `hcc_adata/adata_with_scvi_best.h5ad` | ~2.8 GB |
| 03 | `hcc_adata/1_5T_unintegrated.h5ad` | ~3.1 GB |
| 04 | `integration/scvi_integrated.h5ad` | ~4.2 GB |
| 05 | `LIANA_consensus/*.csv` + `cellchat_by_batch/*.png` | ~500 MB |
| 06 | `csc_figures/Fig_CSC_panel_FINAL.png` | ~20 MB |
| 07 | `DGE_hepatocytes_CLEAN.csv`, `Malignant_vs_Healthy.csv` | ~50 MB |
| 08 | `GSEA_results/*.png` | ~100 MB |

**Total runtime:** ~48–72 hours on a GPU-enabled machine (NVIDIA Tesla V100+).

## Key Methods

- **Doublet detection:** scVI + SOLO (threshold: dif > 0.9)
- **Cell type annotation:** CellTypist (5 references) + scANVI label transfer
- **Batch correction:** scVI (n_latent=30, 200 epochs)
- **Clustering:** Leiden (resolution optimized via ARI against reference labels)
- **DE analysis:** Pseudobulk edgeR (TMM normalization, GLM-QL fit)
- **CCC analysis:** LIANA+ consensus (CellChat, CellPhoneDB, SingleCellSignalR, NATMI)
- **Enrichment:** ClusterProfiler ORA (GOBP, FDR < 0.05)

## Dependencies

### Python Packages
- scanpy==1.9.2
- scvi-tools==0.20.1
- celltypist==1.0.2
- liana==0.1.14
- pandas==1.5.3
- numpy==1.24.3
- scipy==1.10.1
- scikit-learn==1.2.2
- rpy2==3.16.3
- anndata2ri (via scanpy-R integration)

### R Packages
- Seurat (v4.3+)
- CellChat (v1.6+)
- clusterProfiler (v4.6+)
- DOSE, enrichplot, ggalluvial, NMF

See `envs/*.yml` for complete dependency specifications.

## Data Availability

All analyses were performed on data from public repositories:

| Dataset | Accession | Cells | Reference |
|---------|-----------|-------|-----------|
| HCC tumors | GSE242889 | 34,892 | [Ma et al. 2024](https://doi.org/10.1038/s41467-024-...) |
| HCC samples | GSE228195 | 18,504 | [... (in prep)] |
| Healthy liver | GSE151530 | 21,437 | [MacParland et al. 2018](https://doi.org/...) |
| HCC reference | GSE189903 | 12,405 | [Guilliams et al. 2022](https://doi.org/...) |
| Liver reference | CZ CELLxGENE | 12,916 | [Tabula Sapiens (2021)](https://doi.org/...) |

**Interactive portal:** https://hcc-atlas-portal.onrender.com

**Source code:** https://github.com/MiguelDiaz02/HCC-atlas-portal

## Citation

Please cite this pipeline as:

```bibtex
@article{diaz2026hcc_scrnaseq_pipeline,
  title={Single-Cell Transcriptomic Profiling Reveals Immunometabolic Reprogramming 
         and Cell-Cell Communication in the Tumor Microenvironment of Human 
         Hepatocellular Carcinoma},
  author={Díaz-Campos, Miguel Á. and Hernández-Lemus, Enrique},
  journal={International Journal of Molecular Sciences},
  year={2026},
  volume={XX},
  pages={XXXX--XXXX}
}
```

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Troubleshooting

**Issue:** `scVI training crashes with GPU memory error`
- **Solution:** Reduce `n_latent` (e.g., 15 instead of 30) or `batch_size` in respective scripts

**Issue:** `cross_validation.py` fails with "label mismatch"
- **Solution:** Ensure `canonical_map_hcc.json` is in `config/` and matches your cell type taxonomy

**Issue:** `liana_ccc.py` runs very slowly
- **Solution:** Run in LIANA-specific conda environment (`conda activate liana_env`) which includes optimized CUDA bindings

**Issue:** R script fails with "missing package"
- **Solution:** Activate R environment: `conda activate r_gsea` and retry

## License

[MIT License](LICENSE)

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [corresponding author email]

---

**Last updated:** 2026-04-22  
**Pipeline version:** 1.0  
**Status:** Active manuscript revision (IJMS peer review)
