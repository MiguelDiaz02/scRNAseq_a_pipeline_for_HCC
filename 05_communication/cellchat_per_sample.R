#!/usr/bin/env Rscript
# =============================================================================
# 05_communication/cellchat_per_sample.R
# =============================================================================
# Per-sample CellChat analysis on the HCC unintegrated AnnData.
# Runs the full CellChat pipeline for every sample (batch) present in the SCE
# object, saving circle plots, chord diagrams, pathway rankings, bubble plots,
# embeddings, and the serialized CellChat RDS per sample.
#
# Configuration
# -------------
# Set shell environment variables to override default paths:
#   INPUT_H5AD  -- path to the HCC unintegrated h5ad
#   OUTPUT_DIR  -- root output directory (sub-dirs created automatically)
#   PYTHON_BIN  -- path to the Python binary for reticulate
#   WORKERS     -- number of parallel workers for future (default: 12)
# =============================================================================

# --- Environment variable config block ---
INPUT_H5AD  <- Sys.getenv("INPUT_H5AD",
                  "/datos/home/mdiaz/HCC_project/hcc_adata/1_5T_unintegrated.h5ad")
OUTPUT_DIR  <- Sys.getenv("OUTPUT_DIR",
                  file.path(getwd(), "cellchat_by_batch"))
PYTHON_BIN  <- Sys.getenv("PYTHON_BIN",
                  "/datos/home/mdiaz/.cache/R/basilisk/1.21.5/zellkonverter/1.19.2/zellkonverterAnnDataEnv-0.11.4/bin/python")
WORKERS     <- as.integer(Sys.getenv("WORKERS", "12"))

# --- Python bridge ---
library(reticulate)
use_python(PYTHON_BIN, required = TRUE)
py_config()
stopifnot(py_module_available("umap"))
py_run_string("import numpy, numba, umap; print(numpy.__version__, numba.__version__, umap.__version__)")

# --- R libs ---
library(ggplot2)
library(dplyr)
library(CellChat)
library(patchwork)
library(anndata)
library(zellkonverter)
library(SingleCellExperiment)
library(presto)
library(Matrix)
library(NMF)
library(ggalluvial)
options(stringsAsFactors = FALSE)

# =============================================================================
# 1) Load .h5ad as SCE and ensure 'logcounts'
# =============================================================================
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("[INFO] Loading: %s\n", INPUT_H5AD))
sce <- zellkonverter::readH5AD(file = INPUT_H5AD)

# Ensure sparse format for assay 'X'
if (!inherits(assay(sce, "X"), "dgCMatrix")) {
  assay(sce, "X") <- as(assay(sce, "X"), "dgCMatrix")
}

# Build logcounts = CP10K + log1p (only if absent)
if (!"logcounts" %in% assayNames(sce)) {
  counts <- assay(sce, "X")
  library.size <- Matrix::colSums(counts)
  logcounts(sce) <- log1p(Matrix::t(Matrix::t(counts) / pmax(library.size, 1)) * 1e4)
}

# Minimum metadata
colData(sce)$samples <- sce$Sample
sce$predicted <- droplevels(factor(sce$predicted))

# =============================================================================
# 2) Harmonize 'predicted' labels
# =============================================================================
map_short <- c(
  "B cells"="B cells","CAFs"="CAFs","CD4-CD69-memory T cells"="CD4-Mem-CD69",
  "CD4-FOXP3-regulatory T cells"="CD4-Treg","CD4-IL7R-central memory T cells"="CD4-CM-IL7R",
  "CD4-KLRB1-T cells"="CD4-KLRB1","CD8-CD69-memory T cells"="CD8-Mem-CD69",
  "CD8-GZMH-effector T cells"="CD8-Eff-GZMH","CD8-GZMK-effector memory T cells"="CD8-EM-GZMK",
  "Cholangiocytes"="Cholangiocyte","Hepatocytes"="Hep","MAIT"="MAIT","Malignant cells"="Malignant",
  "NK-CD160-tissue resident"="NK-TR-CD160","NK-GNLY-circulatory"="NK-Circ-GNLY",
  "Plasma cells"="Plasma","T cells"="T cells","T cells-MKI67-proliferative"="T-MKI67",
  "TAM"="TAM","TAMs"="TAM","TEC"="TEC","TECs"="TEC","basophil"="Basophil",
  "c0-LUM-inflammatory CAF"="CAF-INF","c0-S100A8-Monocyte"="Mono-S100A8","c0-VWF-endothelial"="Endo-VWF",
  "c1-ANGPT2-endothelial"="Endo-ANGPT2","c1-CXCL10-M1 Macrophage"="M1-Macro","c1-MYH11-vascular CAF"="CAF-VASC",
  "c2-APOA1-hepatocyte like CAF"="CAF-HL","c2-CCL4L2-M2 Macrophage"="M2-Macro","c2-CRHBP-endothelial"="Endo-CRHBP",
  "c3-CCL5-endothelial"="Endo-CCL5","c3-TPSB2-Mast cells"="Mast","c4-RGS5-endothelial"="Endo-RGS5",
  "cholangiocyte"="Cholangiocyte","conventional dendritic cell"="cDC","endothelial cell"="Endo",
  "fibroblast"="Fibro","hepatocyte"="Hep","macrophage"="Macro","monocyte"="Mono","natural killer cell"="NK",
  "neutrophil"="Neutro","plasma cell"="Plasma","plasmacytoid dendritic cell"="pDC",
  "tumor"="Malignant","unclassified"="Unclassified","unspecified"="Unclassified"
)

orig_pred <- as.character(sce$predicted)
mapped    <- ifelse(orig_pred %in% names(map_short), map_short[orig_pred], orig_pred)
sce$predicted <- droplevels(factor(mapped))

unmapped <- setdiff(unique(orig_pred), names(map_short))
cat("Unmapped labels (if any):\n"); print(unmapped)
print(sort(table(sce$predicted), decreasing = TRUE)[1:20])


# =============================================================================
# Helper utilities
# =============================================================================
idx_of <- function(grps, cc = NULL) {
  # Returns integer indices of cell-type names in cellChat@idents.
  # cc: a CellChat object; if NULL, must be defined in calling scope.
  if (is.null(cc)) cc <- cellChat
  lvl <- levels(cc@idents)
  idx <- match(grps, lvl)
  idx[!is.na(idx)]
}

ensure_dir <- function(...) {
  # Create directory and return its path.
  d <- file.path(...)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

build_topN_matrix <- function(cc, sources, N = 5, pathways = NULL) {
  # Build a source x target probability matrix with only the top-N edges per source.
  df <- subsetCommunication(cc, signaling = pathways)
  df <- df[df$source %in% sources, , drop = FALSE]
  if (nrow(df) == 0) return(list(W = NULL, keep = character(0), df_top = df))
  df <- dplyr::arrange(df, source, dplyr::desc(prob))
  df_top <- df |>
    dplyr::group_by(source) |>
    dplyr::slice_max(order_by = prob, n = N, with_ties = FALSE) |>
    dplyr::ungroup()
  lvl <- levels(cc@idents)
  W   <- matrix(0, length(lvl), length(lvl), dimnames = list(lvl, lvl))
  idx <- cbind(match(df_top$source, lvl), match(df_top$target, lvl))
  W[idx] <- df_top$prob
  keep <- union(unique(df_top$source), unique(df_top$target))
  list(W = W[keep, keep, drop = FALSE], keep = keep, df_top = df_top)
}

prep_pair_by_interaction <- function(df) {
  # Build a data.frame(interaction_name=...) from an edge data.frame.
  col_int <- intersect(c("interaction_name","interaction","LR_pair","LRpair","name"),
                       colnames(df))[1]
  if (!is.na(col_int)) {
    un <- unique(stats::na.omit(as.character(df[[col_int]])))
    if (length(un) > 0)
      return(data.frame(interaction_name = un, stringsAsFactors = FALSE))
  }
  colL <- intersect(c("ligand","ligand_gene","ligand.symbol","ligand.name"), colnames(df))[1]
  colR <- intersect(c("receptor","receptor_gene","receptor.symbol","receptor.name"), colnames(df))[1]
  if (is.na(colL) || is.na(colR)) return(NULL)
  LR <- unique(stats::na.omit(paste0(as.character(df[[colL]]), "_", as.character(df[[colR]]))))
  if (length(LR) == 0) return(NULL)
  data.frame(interaction_name = LR, stringsAsFactors = FALSE)
}

make_pairLR_interaction <- function(df, col_int, col_lig, col_rec) {
  # Build pairLR.use from an edge data.frame; prefers interaction_name if present.
  if (!is.na(col_int)) {
    u <- unique(stats::na.omit(as.character(df[[col_int]])))
    if (length(u) > 0) return(data.frame(interaction_name = u, stringsAsFactors = FALSE))
  }
  if (!is.na(col_lig) && !is.na(col_rec)) {
    LR <- unique(stats::na.omit(paste0(as.character(df[[col_lig]]), "_", as.character(df[[col_rec]]))))
    if (length(LR) > 0) return(data.frame(interaction_name = LR, stringsAsFactors = FALSE))
  }
  NULL
}

.pick_k <- function(cc, pattern = c("outgoing", "incoming"), DIR_FIG_PAT, SAMPLE_ID) {
  # Safely choose NMF rank K for communication pattern identification.
  pattern <- match.arg(pattern)
  kmax <- min(6, max(2, length(cc@netP$pathways)))
  if (kmax < 2) return(0)
  png(file.path(DIR_FIG_PAT, sprintf("%s__selectK__%s.png", SAMPLE_ID, pattern)),
      width = 1600, height = 1200, res = 200)
  sel_ok <- TRUE
  tryCatch(selectK(cc, pattern = pattern), error = function(e) sel_ok <<- FALSE)
  dev.off()
  if (pattern == "outgoing") return(min(3, kmax))
  return(min(4, kmax))
}

select_topK_for_source <- function(tab, K = 3) {
  # Select top-K pathways per source with diversity constraints (category + family).
  adhesion_families <- c("CD99","PECAM1","PECAM2","ICAM","VCAM","ESAM","JAM","SELE","SELL",
                          "SELPLG","CLDN","OCLN","CDH","CDH1","CDH5","NECTIN","CADM","PTPRM","CD34")
  if (nrow(tab) == 0) return(tab[0, ])
  base <- tab |>
    dplyr::arrange(dplyr::desc(flow)) |>
    dplyr::mutate(family = dplyr::case_when(
      pathway %in% c("MHC-I","MHC-II") ~ "MHC",
      pathway %in% adhesion_families   ~ "ADHESION",
      TRUE                             ~ "OTHER"
    ))
  cand <- base |>
    dplyr::filter((is.na(n_targets) | n_targets >= 2), n_interactions >= 5)
  if (nrow(cand) == 0) cand <- base

  picked   <- cand[0, ]
  used_cat <- character(0)
  used_fam <- character(0)
  for (i in seq_len(nrow(cand))) {
    if (nrow(picked) >= K) break
    r <- cand[i, ]
    if (!is.na(r$category) && r$category %in% used_cat) next
    if (r$family %in% c("MHC","ADHESION") && r$family %in% used_fam) next
    picked   <- dplyr::bind_rows(picked, r)
    used_cat <- c(used_cat, r$category)
    used_fam <- c(used_fam, r$family)
  }
  if (nrow(picked) < K) {
    rest <- dplyr::anti_join(cand, picked, by = c("source","pathway"))
    for (i in seq_len(nrow(rest))) {
      if (nrow(picked) >= K) break
      r <- rest[i, ]
      if (r$family %in% c("MHC","ADHESION") && r$family %in% picked$family) next
      picked <- dplyr::bind_rows(picked, r)
    }
  }
  if (nrow(picked) < K) {
    rest2  <- dplyr::anti_join(cand, picked, by = c("source","pathway"))
    picked <- dplyr::bind_rows(picked, utils::head(rest2, K - nrow(picked)))
  }
  picked
}

plot_one_pathway <- function(cc, pathway, out_prefix, src = NULL) {
  # Render circle, chord, heatmap, and bubble plots for a single pathway.
  src_idx <- NULL
  if (!is.null(src)) {
    lvl     <- levels(cc@idents)
    src_idx <- match(src, lvl)
    if (is.na(src_idx)) src_idx <- NULL
  }
  png(paste0(out_prefix, "__circle.png"), width = 2000, height = 1600, res = 200)
  par(mfrow = c(1,1), mar = c(2,2,3,2), xpd = NA)
  netVisual_aggregate(cc, signaling = pathway, layout = "circle",
                      sources.use = src_idx)
  dev.off()
  png(paste0(out_prefix, "__chord.png"), width = 2000, height = 1600, res = 200)
  par(mfrow = c(1,1), mar = c(2,2,3,2), xpd = NA)
  netVisual_aggregate(cc, signaling = pathway, layout = "chord",
                      sources.use = src_idx, label.size = 0.8)
  dev.off()
  png(paste0(out_prefix, "__heatmap.png"), width = 1600, height = 1400, res = 200)
  par(mfrow = c(1,1))
  ok <- TRUE
  tryCatch(
    netVisual_heatmap(cc, signaling = pathway, color.heatmap = "Reds",
                      sources.use = src_idx),
    error = function(e) { ok <<- FALSE }
  )
  if (!ok) netVisual_heatmap(cc, signaling = pathway, color.heatmap = "Reds")
  dev.off()
  if (!is.null(src_idx)) {
    png(paste0(out_prefix, "__bubble.png"), width = 1800, height = 1600, res = 200)
    netVisual_bubble(cc, signaling = pathway, sources.use = src_idx, remove.isolate = TRUE)
    dev.off()
  }
  gg <- netAnalysis_contribution(cc, signaling = pathway)
  ggplot2::ggsave(filename = paste0(out_prefix, "__LR_contribution.pdf"),
                  plot = gg, width = 3.5, height = 2.5, units = "in")
}

expand_groups <- function(x, lvl) {
  # Expand generic group names (CAF, Endo, NK) to the subtype labels present in lvl.
  out <- unlist(lapply(x, function(g) {
    if (g == "CAF")  return(intersect(c("CAFs","CAF-INF","CAF-VASC","CAF-HL"), lvl))
    if (g == "Endo") return(intersect(c("Endo","Endo-ANGPT2","Endo-CCL5","Endo-CRHBP","Endo-RGS5"), lvl))
    if (g == "NK")   return(intersect(c("NK","NK-Circ-GNLY","NK-TR-CD160"), lvl))
    return(intersect(g, lvl))
  }))
  unique(out)
}


# =============================================================================
# Per-sample wrapper
# =============================================================================
run_single_batch <- function(
    SAMPLE_ID,
    FOCUS_SOURCES = c("TAM","M2-Macro","Hep","Malignant","TEC"),
    FILTER_METHOD = "topN",
    TOP_N         = 5,
    WORKERS       = 12,
    OUT_ROOT      = OUTPUT_DIR,
    MIN_CELLS     = 0
) {
  # ---- 3) Select sample and create per-sample CellChat object ----
  stopifnot(SAMPLE_ID %in% sce$Sample)
  sce_b <- sce[, sce$Sample == SAMPLE_ID]
  if (ncol(sce_b) < MIN_CELLS) {
    message(sprintf("Batch %s skipped: %d cells (< %d).", SAMPLE_ID, ncol(sce_b), MIN_CELLS))
    return(invisible(NULL))
  }
  sce_b$predicted <- droplevels(sce_b$predicted)
  cat(sprintf("Batch: %s  |  #cells: %d\n", SAMPLE_ID, ncol(sce_b)))

  # Output sub-directories
  OUTDIR  <- file.path(OUT_ROOT, SAMPLE_ID)
  DIR_FIG <- ensure_dir(OUTDIR, "figs")
  DIR_TAB <- ensure_dir(OUTDIR, "tables")

  cellChat <- createCellChat(object = sce_b, group.by = "predicted")

  # ---- 4) Interaction DB + preprocessing ----
  set.seed(123)
  CellChatDB     <- CellChatDB.human
  CellChatDB.use <- subsetDB(CellChatDB,
                             search = c("Secreted Signaling","ECM-Receptor","Cell-Cell Contact"),
                             key = "annotation")
  cellChat@DB <- CellChatDB.use

  lvl     <- levels(cellChat@idents)
  lvl_new <- c(FOCUS_SOURCES[FOCUS_SOURCES %in% lvl], setdiff(lvl, FOCUS_SOURCES))
  cellChat@idents <- factor(cellChat@idents, levels = lvl_new)
  cellChat@idents <- droplevels(cellChat@idents)

  cellChat <- subsetData(cellChat)

  future::plan("multisession", workers = WORKERS)
  on.exit({
    future::plan("sequential")
    try(future:::ClusterRegistry("stop"), silent = TRUE)
  }, add = TRUE)

  cellChat <- identifyOverExpressedGenes(cellChat)
  cellChat <- identifyOverExpressedInteractions(cellChat)

  # ---- 5) Communication probabilities ----
  cellChat <- computeCommunProb(cellChat, raw.use = TRUE)
  cellChat <- filterCommunication(cellChat, min.cells = 10)

  df.net <- subsetCommunication(cellChat)
  utils::write.csv(df.net,
                   file = file.path(DIR_TAB, sprintf("%s__subsetCommunication.csv", SAMPLE_ID)),
                   row.names = FALSE)

  cellChat <- computeCommunProbPathway(cellChat)
  cellChat <- aggregateNet(cellChat)

  # ---- 6) General summary figures ----
  groupSize <- as.numeric(table(cellChat@idents))

  png(file.path(DIR_FIG, sprintf("%s__circle_num_interactions.png", SAMPLE_ID)),
      width = 2200, height = 1800, res = 200)
  par(mfrow = c(1,1), mar = c(2,2,3,2), xpd = NA, cex.main = 0.9)
  netVisual_circle(cellChat@net$count, vertex.weight = groupSize, weight.scale = TRUE,
                   label.edge = FALSE, vertex.label.cex = 0.75,
                   title.name = "Number of interactions")
  dev.off()

  png(file.path(DIR_FIG, sprintf("%s__circle_interaction_strength.png", SAMPLE_ID)),
      width = 2200, height = 1800, res = 200)
  par(mfrow = c(1,1), mar = c(2,2,3,2), xpd = NA, cex.main = 0.9)
  netVisual_circle(cellChat@net$weight, vertex.weight = groupSize, weight.scale = TRUE,
                   label.edge = FALSE, vertex.label.cex = 0.75,
                   title.name = "Interaction strength")
  dev.off()

  # ---- 8) Top-N = 5 per source (focus sources) ----
  topN <- build_topN_matrix(cellChat, sources = FOCUS_SOURCES, N = TOP_N,
                            pathways = cellChat@netP$pathways)
  utils::write.csv(topN$df_top,
                   file = file.path(DIR_TAB, sprintf("%s__top%d_edges_per_source__focus.csv",
                                                     SAMPLE_ID, TOP_N)),
                   row.names = FALSE)

  if (!is.null(topN$W) && length(topN$keep) > 0) {
    gsub <- as.numeric(table(cellChat@idents)[topN$keep])
    png(file.path(DIR_FIG, sprintf("%s__circle_top%d_focus.png", SAMPLE_ID, TOP_N)),
        width = 2200, height = 1800, res = 200)
    par(mfrow = c(1,1), mar = c(2,2,3,2), xpd = NA, cex.main = 0.9)
    netVisual_circle(topN$W, vertex.weight = gsub, weight.scale = TRUE,
                     label.edge = FALSE, vertex.label.cex = 0.85,
                     title.name = sprintf("Top-%d edges per source", TOP_N))
    dev.off()
  }

  # ---- 9) Pathway selection: Top-3 per source ----
  K_TOP    <- 3
  DIR_FIG_SRC <- ensure_dir(DIR_FIG, "per_source")
  DIR_TAB_RNK <- ensure_dir(DIR_TAB, "rankings")
  DIR_TAB_SEL <- ensure_dir(DIR_TAB, "selections")

  dfp      <- subsetCommunication(cellChat)
  path_col <- intersect(c("pathway_name","pathway"), colnames(dfp))[1]
  stopifnot(!is.na(path_col))

  annot_map <- unique(cellChat@DB$interaction[, c("pathway_name","annotation")])
  colnames(annot_map) <- c("pathway","category")

  dfp2 <- dfp |>
    dplyr::mutate(pathway = .data[[path_col]]) |>
    dplyr::left_join(annot_map, by = "pathway")

  targets_stats <- dfp2 |>
    dplyr::group_by(source, pathway, target) |>
    dplyr::summarise(prob_t = sum(prob, na.rm = TRUE), .groups = "drop") |>
    dplyr::group_by(source, pathway) |>
    dplyr::summarise(
      n_targets        = dplyr::n(),
      top_target_share = max(prob_t) / sum(prob_t),
      .groups          = "drop"
    )

  metrics <- dfp2 |>
    dplyr::group_by(source, pathway) |>
    dplyr::summarise(
      n_interactions = dplyr::n(),
      flow           = sum(prob, na.rm = TRUE),
      category       = dplyr::first(.data[["category"]]),
      .groups        = "drop"
    ) |>
    dplyr::left_join(targets_stats, by = c("source","pathway")) |>
    dplyr::group_by(source) |>
    dplyr::mutate(flow_rel = ifelse(sum(flow) > 0, flow / sum(flow), 0)) |>
    dplyr::ungroup()

  utils::write.csv(metrics |> dplyr::arrange(source, dplyr::desc(flow)),
                   file = file.path(DIR_TAB_RNK,
                                    sprintf("%s__pathways_rank_metrics_by_source.csv", SAMPLE_ID)),
                   row.names = FALSE)

  present_sources <- intersect(FOCUS_SOURCES, levels(cellChat@idents))
  missing_sources <- setdiff(FOCUS_SOURCES, present_sources)
  if (length(missing_sources) > 0) {
    utils::write.csv(data.frame(source = missing_sources),
                     file = file.path(DIR_TAB_SEL,
                                      sprintf("%s__missing_sources.csv", SAMPLE_ID)),
                     row.names = FALSE)
  }

  sel_list <- lapply(present_sources, function(src) {
    select_topK_for_source(metrics[metrics$source == src, , drop = FALSE], K = K_TOP)
  })
  names(sel_list) <- present_sources

  selections <- dplyr::bind_rows(sel_list)
  utils::write.csv(selections,
                   file = file.path(DIR_TAB_SEL,
                                    sprintf("%s__selected_top%d_pathways_per_source.csv",
                                            SAMPLE_ID, K_TOP)),
                   row.names = FALSE)

  # Per-source, per-pathway figures
  for (src in present_sources) {
    tab_src <- selections[selections$source == src, , drop = FALSE]
    if (nrow(tab_src) == 0) next
    out_src <- ensure_dir(DIR_FIG_SRC, src)
    for (pw in tab_src$pathway) {
      out_prefix <- file.path(out_src, sprintf("%s__%s__%s", SAMPLE_ID, src, pw))
      plot_one_pathway(cellChat, pw, out_prefix, src = src)
    }
  }

  # ---- Bubble / Chord / Role / Gene expression figures ----
  DIR_FIG_BUB  <- ensure_dir(DIR_FIG, "bubbles")
  DIR_FIG_CHG  <- ensure_dir(DIR_FIG, "chord_gene")
  DIR_FIG_ROLE <- ensure_dir(DIR_FIG, "roles")
  DIR_FIG_GEXP <- ensure_dir(DIR_FIG, "gene_expression")
  DIR_TAB_AUD  <- ensure_dir(DIR_TAB, "audit_pairs_detail")

  lvl_all  <- levels(cellChat@idents)
  paths_av <- cellChat@netP$pathways

  src_names <- c("TAM","M2-Macro","Hep","Malignant","TEC")
  tgt_names <- lvl_all
  src_idx   <- idx_of(src_names, cellChat)
  tgt_idx   <- idx_of(tgt_names, cellChat)

  cand_paths <- intersect(c("CCL","CXCL","VEGF","EGF","MK"), paths_av)

  # (1) Bubble: all LR from sources -> all targets
  if (length(src_idx) > 0 && length(tgt_idx) > 0) {
    png(file.path(DIR_FIG_BUB, sprintf("%s__bubble__allLR__sources_focus_to_all.png", SAMPLE_ID)),
        width = 2200, height = 1800, res = 200)
    print(netVisual_bubble(cellChat, sources.use = src_idx, targets.use = tgt_idx,
                           remove.isolate = TRUE))
    dev.off()
  }

  # (2) Bubble for candidate pathways
  if (length(cand_paths) > 0 && length(src_idx) > 0) {
    png(file.path(DIR_FIG_BUB, sprintf("%s__bubble__%s__sources_focus.png",
                                       SAMPLE_ID, paste(cand_paths, collapse = "_"))),
        width = 2200, height = 1800, res = 200)
    print(netVisual_bubble(cellChat, sources.use = src_idx, targets.use = tgt_idx,
                           signaling = cand_paths, remove.isolate = TRUE))
    dev.off()
  }

  # (3) Bubble: pairLR from candidate paths
  col_lig_res <- intersect(c("ligand","ligand_gene","ligand.symbol","ligand.name"), colnames(dfp))[1]
  col_rec_res <- intersect(c("receptor","receptor_gene","receptor.symbol","receptor.name"), colnames(dfp))[1]
  col_pat_res <- intersect(c("pathway_name","pathway"), colnames(dfp))[1]

  src_idx_small <- idx_of(c("TAM","M2-Macro"), cellChat)
  if (length(src_idx_small) == 0) src_idx_small <- src_idx
  tgt_idx_small <- idx_of(c("Mono-S100A8","Macro","Hep","NK","NK-Circ-GNLY","CAF","CAFs","CD4-Treg"), cellChat)
  if (length(tgt_idx_small) == 0) tgt_idx_small <- tgt_idx
  src_names_small <- lvl_all[src_idx_small]
  tgt_names_small <- lvl_all[tgt_idx_small]

  plot_one_bubble <- function(edges_df, src_idx_use, tgt_idx_use, tag) {
    pr <- prep_pair_by_interaction(edges_df)
    if (is.null(pr) || nrow(pr) == 0) {
      message(sprintf("[%s] no 'interaction_name' after prep; skipping.", tag))
      return(invisible(FALSE))
    }
    png(file.path(DIR_FIG_BUB, sprintf("%s__bubble__pairLR__%s__%s.png",
                                       SAMPLE_ID, paste(cand_paths, collapse = "_"), tag)),
        width = 2200, height = 1800, res = 200)
    ok <- TRUE
    tryCatch({
      print(netVisual_bubble(cellChat, sources.use = src_idx_use, targets.use = tgt_idx_use,
                             pairLR.use = pr, remove.isolate = TRUE))
    }, error = function(e) { ok <<- FALSE; message(sprintf("[%s] bubble skipped: %s", tag, conditionMessage(e))) })
    dev.off()
    invisible(ok)
  }

  res_A <- dfp[dfp[[col_pat_res]] %in% cand_paths & dfp$source %in% src_names_small & dfp$target %in% tgt_names_small, , drop = FALSE]
  res_B <- dfp[dfp[[col_pat_res]] %in% cand_paths & dfp$source %in% src_names_small, , drop = FALSE]
  res_C <- dfp[dfp[[col_pat_res]] %in% cand_paths, , drop = FALSE]

  done <- FALSE
  if (nrow(res_A) > 0 && length(src_idx_small) > 0 && length(tgt_idx_small) > 0)
    done <- plot_one_bubble(res_A, src_idx_small, tgt_idx_small, "A_srcFocus_tgtEndoTEC")
  if (!isTRUE(done) && nrow(res_B) > 0 && length(src_idx_small) > 0)
    done <- plot_one_bubble(res_B, src_idx_small, tgt_idx, "B_srcFocus_tgtAll")
  if (!isTRUE(done) && nrow(res_C) > 0)
    done <- plot_one_bubble(res_C, src_idx, tgt_idx, "C_srcAll_tgtAll")
  if (!isTRUE(done))
    message("No plottable L-R pairs in cand_paths for this batch.")

  # (3bis) Audit of pairs of interest
  pairs_interest <- data.frame(
    ligand   = c("HLA-E","HLA-E","DLL1","CEACAM1","APP","TF","AREG",
                 "APOE","APOB","APOB","APOA1","ALB","C3","CCL5","PTPRC"),
    receptor = c("VSIR","KLRC1","NOTCH1","CD209","CD74","TFR2","EGFR",
                 "TREM2","TREM2","APOBR","ABCA1","FCGRT","C3AR1","CCR1","MRC1"),
    stringsAsFactors = FALSE
  )
  synonyms <- list(
    FCGRT = c("FCGRT","FcRn","FcRn-complex","FCGRT-complex"),
    TREM2 = c("TREM2","TREM2-receptor"),
    HLAE  = c("HLA-E","HLAE")
  )

  db          <- cellChat@DB$interaction
  dfp         <- subsetCommunication(cellChat)
  col_lig_db  <- intersect(c("ligand","ligand_gene"), colnames(db))[1]
  col_rec_db  <- intersect(c("receptor","receptor_gene"), colnames(db))[1]
  col_pat_db  <- intersect(c("pathway_name","pathway"), colnames(db))[1]
  col_lig_res <- intersect(c("ligand","ligand_gene"), colnames(dfp))[1]
  col_rec_res <- intersect(c("receptor","receptor_gene"), colnames(dfp))[1]
  col_pat_res <- intersect(c("pathway_name","pathway"), colnames(dfp))[1]

  mk_pat <- function(x) {
    key  <- gsub("-", "", x)
    alts <- unique(c(x, unlist(synonyms[[key]])))
    alts <- gsub("\\+", "\\\\+", alts)
    alts <- gsub("\\s+", "[-_ ]*", alts)
    paste0("(^|[ ,;:/\\(\\)])(", paste(alts, collapse="|"), ")([ ,;:/\\)\\[]|$)")
  }

  audit_rows <- lapply(seq_len(nrow(pairs_interest)), function(i) {
    L_q <- pairs_interest$ligand[i]; R_q <- pairs_interest$receptor[i]
    pL  <- mk_pat(L_q); pR <- mk_pat(R_q)
    in_db_direct  <- grepl(pL, db[[col_lig_db]], ignore.case=TRUE) & grepl(pR, db[[col_rec_db]], ignore.case=TRUE)
    in_db_reverse <- grepl(pL, db[[col_rec_db]], ignore.case=TRUE) & grepl(pR, db[[col_lig_db]], ignore.case=TRUE)
    db_hits       <- db[in_db_direct | in_db_reverse, c(col_lig_db, col_rec_db, col_pat_db), drop=FALSE]
    in_res_direct  <- grepl(pL, dfp[[col_lig_res]], ignore.case=TRUE) & grepl(pR, dfp[[col_rec_res]], ignore.case=TRUE)
    in_res_reverse <- grepl(pL, dfp[[col_rec_res]], ignore.case=TRUE) & grepl(pR, dfp[[col_lig_res]], ignore.case=TRUE)
    res_hits       <- dfp[in_res_direct | in_res_reverse,
                          c("source","target",col_pat_res,col_lig_res,col_rec_res,"prob"), drop=FALSE]
    data.frame(
      query        = paste(L_q, R_q, sep="--"),
      in_database  = nrow(db_hits) > 0,
      n_db_records = nrow(db_hits),
      detected     = nrow(res_hits) > 0,
      n_edges      = nrow(res_hits),
      top_pathways = if (nrow(res_hits)>0) paste(unique(res_hits[[col_pat_res]])[1:min(3,length(unique(res_hits[[col_pat_res]])))], collapse=", ") else NA_character_,
      stringsAsFactors = FALSE
    ) |> dplyr::mutate(details_db = I(list(db_hits)), details_res = I(list(res_hits)))
  })
  audit_tbl <- dplyr::bind_rows(audit_rows)
  utils::write.csv(audit_tbl[, c("query","in_database","n_db_records","detected","n_edges","top_pathways")],
                   file = file.path(DIR_TAB, sprintf("%s__audit_pairs_interest__summary.csv", SAMPLE_ID)),
                   row.names = FALSE)
  for (i in seq_len(nrow(audit_tbl))) {
    q    <- gsub("[^A-Za-z0-9._-]", "_", audit_tbl$query[i])
    dbd  <- audit_tbl$details_db[[i]];  if (!is.null(dbd)  && nrow(dbd)>0)  utils::write.csv(dbd,  file.path(DIR_TAB_AUD, paste0(q,"__DB_hits.csv")),  row.names=FALSE)
    resd <- audit_tbl$details_res[[i]]; if (!is.null(resd) && nrow(resd)>0) utils::write.csv(resd, file.path(DIR_TAB_AUD, paste0(q,"__RES_hits.csv")), row.names=FALSE)
  }

  # Bubble for detected pairs with focus sources
  det_pairs     <- audit_tbl$details_res[ audit_tbl$detected ]
  src_idx_focus <- idx_of(src_names, cellChat)
  if (length(det_pairs) > 0 && length(src_idx_focus) > 0) {
    res_all   <- dplyr::bind_rows(det_pairs)
    res_focus <- res_all[ res_all[["source"]] %in% src_names, , drop = FALSE ]
    if (nrow(res_focus) > 0) {
      col_int <- intersect(c("interaction_name","interaction","LR_pair","LRpair","name"), colnames(res_focus))[1]
      pairLR_focus <- if (!is.na(col_int)) {
        data.frame(interaction_name = unique(as.character(res_focus[[col_int]])), stringsAsFactors = FALSE)
      } else {
        pr <- unique(res_focus[, c(col_lig_res, col_rec_res), drop = FALSE])
        pr <- stats::na.omit(pr)
        data.frame(interaction_name = unique(paste0(as.character(pr[[1]]), "_", as.character(pr[[2]]))), stringsAsFactors = FALSE)
      }
      if (nrow(pairLR_focus) > 0) {
        png(file.path(DIR_FIG_BUB, sprintf("%s__bubble__pairs_interest__focus_sources.png", SAMPLE_ID)),
            width = 2200, height = 1800, res = 200)
        ok <- TRUE
        tryCatch({
          print(netVisual_bubble(cellChat, sources.use = src_idx_focus,
                                 pairLR.use = pairLR_focus, remove.isolate = TRUE))
        }, error = function(e) { ok <<- FALSE; message("Bubble (pairs_interest) skipped: ", conditionMessage(e)) })
        dev.off()
        if (!isTRUE(ok)) {
          png(file.path(DIR_FIG_BUB, sprintf("%s__bubble__pairs_interest__ALL_sources.png", SAMPLE_ID)),
              width = 2200, height = 1800, res = 200)
          tryCatch({
            print(netVisual_bubble(cellChat, pairLR.use = pairLR_focus, remove.isolate = TRUE))
          }, error = function(e) message("Fallback bubble skipped: ", conditionMessage(e)))
          dev.off()
        }
      }
    }
  }

  # (4) Bubble sorted by target
  tgt_order <- intersect(c("Macro","M2-Macro","M1-Macro","TEC","TAM"), lvl_all)
  if (length(tgt_order) > 0) {
    df_tgt   <- dfp[dfp$target %in% tgt_order, , drop = FALSE]
    col_int  <- intersect(c("interaction_name","interaction","LR_pair","LRpair","name"), colnames(df_tgt))[1]
    pairLR_s <- if (!is.na(col_int)) {
      unique(stats::na.omit(as.character(df_tgt[[col_int]])))
    } else {
      colL <- intersect(c("ligand","ligand_gene","ligand.symbol","ligand.name"), colnames(df_tgt))[1]
      colR <- intersect(c("receptor","receptor_gene","receptor.symbol","receptor.name"), colnames(df_tgt))[1]
      if (!is.na(colL) && !is.na(colR))
        unique(stats::na.omit(paste0(as.character(df_tgt[[colL]]), "_", as.character(df_tgt[[colR]]))))
      else
        character(0)
    }
    if (length(pairLR_s) > 0) {
      pairLR_s <- data.frame(interaction_name = pairLR_s, stringsAsFactors = FALSE)
      png(file.path(DIR_FIG_BUB, sprintf("%s__bubble__sorted_by_target.png", SAMPLE_ID)),
          width = 4200, height = 3800, res = 200)
      print(netVisual_bubble(cellChat, targets.use = tgt_order, pairLR.use = pairLR_s,
                             remove.isolate = TRUE, sort.by.target = TRUE))
      dev.off()
    }
  }

  # (5-6) Bubble sorted by source and by both
  src_order_req <- c("TAM","M2-Macro","TEC","Malignant")
  tgt_order_req <- c("Mono-S100A8","Macro","Hep","NK","NK-Circ-GNLY","CAF","CAFs","CD4-Treg")
  src_order <- expand_groups(src_order_req, lvl_all)
  tgt_order <- expand_groups(tgt_order_req, lvl_all)

  col_int <- intersect(c("interaction_name","interaction","LR_pair","LRpair","name"), colnames(dfp))[1]
  col_lig <- intersect(c("ligand","ligand_gene","ligand.symbol","ligand.name"), colnames(dfp))[1]
  col_rec <- intersect(c("receptor","receptor_gene","receptor.symbol","receptor.name"), colnames(dfp))[1]

  df_st <- dfp[dfp$source %in% src_order & dfp$target %in% tgt_order, , drop = FALSE]
  if (nrow(df_st) == 0) {
    message("No edges src_order->tgt_order; trying src_order only.")
    df_st     <- dfp[dfp$source %in% src_order, , drop = FALSE]
    tgt_order <- intersect(unique(df_st$target), lvl_all)
  }

  pairLR.use <- make_pairLR_interaction(df_st, col_int, col_lig, col_rec)
  if (length(src_order) > 0 && !is.null(pairLR.use) && nrow(pairLR.use) > 0) {
    png(file.path(DIR_FIG_BUB, sprintf("%s__bubble__sorted_by_source.png", SAMPLE_ID)),
        width = 3200, height = 2800, res = 200)
    print(netVisual_bubble(cellChat, sources.use = src_order, pairLR.use = pairLR.use,
                           remove.isolate = TRUE, sort.by.source = TRUE))
    dev.off()
    if (length(tgt_order) > 0) {
      png(file.path(DIR_FIG_BUB, sprintf("%s__bubble__sorted_by_both.png", SAMPLE_ID)),
          width = 3200, height = 2800, res = 200)
      print(netVisual_bubble(cellChat, sources.use = src_order, targets.use = tgt_order,
                             pairLR.use = pairLR.use, remove.isolate = TRUE,
                             sort.by.source = TRUE, sort.by.target = TRUE,
                             sort.by.source.priority = TRUE))
      dev.off()
    }
  }

  # (9) Chord gene
  m2_inf_idx <- idx_of("M2-Macro", cellChat)
  tgt_some   <- idx_of(c("Macro","Malignant","TEC","TAM","Hep"), cellChat)
  if (length(m2_inf_idx) == 1 && length(tgt_some) > 0) {
    png(file.path(DIR_FIG_CHG, sprintf("%s__chord_gene__M2_to_specific_pop.png", SAMPLE_ID)),
        width = 2200, height = 2000, res = 200)
    netVisual_chord_gene(cellChat, sources.use = m2_inf_idx, targets.use = tgt_some,
                         lab.cex = 0.6, legend.pos.y = 30)
    dev.off()
  }
  tec_idx  <- idx_of("TEC", cellChat)
  src_some <- idx_of(c("Macro","Malignant","M2-Macro","TAM","Hep"), cellChat)
  if (length(tec_idx) == 1) {
    png(file.path(DIR_FIG_CHG, sprintf("%s__chord_gene_TEC_to_specific_pop.png", SAMPLE_ID)),
        width = 2200, height = 2000, res = 200)
    netVisual_chord_gene(cellChat, sources.use = src_some, targets.use = tec_idx, legend.pos.x = 15)
    dev.off()
    if (length(cand_paths) > 0) {
      png(file.path(DIR_FIG_CHG, sprintf("%s__chord_gene__to_TEC__%s.png",
                                         SAMPLE_ID, paste(cand_paths, collapse = "_"))),
          width = 2200, height = 2000, res = 200)
      netVisual_chord_gene(cellChat, sources.use = src_some, targets.use = tec_idx,
                           signaling = cand_paths, legend.pos.x = 8)
      dev.off()
    }
  }

  # (10) Centrality / roles
  cellChat <- netAnalysis_computeCentrality(cellChat, slot.name = "netP")
  paths_sel <- unique(selections$pathway)
  if (length(paths_sel) == 0) paths_sel <- paths_av

  png(file.path(DIR_FIG_ROLE, sprintf("%s__signalingRole_network.png", SAMPLE_ID)),
      width = 3200, height = 800, res = 200)
  print(netAnalysis_signalingRole_network(cellChat, signaling = paths_sel,
                                          width = 16, height = 4, font.size = 8))
  dev.off()

  gg1 <- netAnalysis_signalingRole_scatter(cellChat)
  gg2 <- netAnalysis_signalingRole_scatter(cellChat, signaling = intersect(c("CXCL","CCL"), paths_av))
  ggsave(file.path(DIR_FIG_ROLE, sprintf("%s__signalingRole_scatter.pdf", SAMPLE_ID)),
         plot = gg1 + gg2, width = 10, height = 4)

  png(file.path(DIR_FIG_ROLE, sprintf("%s__signalingRole_heatmap_out_in.png", SAMPLE_ID)),
      width = 2400, height = 1600, res = 200)
  ht1 <- netAnalysis_signalingRole_heatmap(cellChat, pattern = "outgoing")
  ht2 <- netAnalysis_signalingRole_heatmap(cellChat, pattern = "incoming")
  print(ht1 + ht2)
  dev.off()

  paths_small <- intersect(c("ANGPT","VEGF","CCL","CXCL","TGFb","ApoA","ApoB","ApoE"), paths_av)
  if (length(paths_small) > 0) {
    png(file.path(DIR_FIG_ROLE, sprintf("%s__signalingRole.png", SAMPLE_ID)),
        width = 1800, height = 1200, res = 200)
    ht <- netAnalysis_signalingRole_heatmap(cellChat, signaling = paths_small)
    print(ht); dev.off()
  }

  # (11) Gene expression per pathway
  safe <- function(x) gsub("[^A-Za-z0-9._-]+", "_", x)
  dfp2      <- subsetCommunication(cellChat)
  pcol      <- intersect(c("pathway_name","pathway"), colnames(dfp2))[1]
  plot_paths <- sort(intersect(paths_av, unique(dfp2[[pcol]])))
  annot_map  <- unique(cellChat@DB$interaction[, c("pathway_name","annotation")])
  colnames(annot_map) <- c("pathway","annotation")
  plot_paths <- intersect(plot_paths,
    annot_map$pathway[annot_map$annotation %in% c("Secreted Signaling","ECM-Receptor","Cell-Cell Contact")])
  plot_paths <- intersect(plot_paths, c("ANGPT","VEGF","CCL","CXCL","TGFb","ApoA","ApoB","ApoE"))
  for (pw in plot_paths) {
    try({ p <- plotGeneExpression(cellChat, signaling = pw, enriched.only = TRUE, type = "violin")
          png(file.path(DIR_FIG_GEXP, sprintf("%s__plotGeneExpression__%s__enriched.png", SAMPLE_ID, safe(pw))),
              width = 2400, height = 1600, res = 200); print(p); dev.off() }, silent = TRUE)
    try({ p <- plotGeneExpression(cellChat, signaling = pw, enriched.only = FALSE)
          png(file.path(DIR_FIG_GEXP, sprintf("%s__plotGeneExpression__%s__all.png", SAMPLE_ID, safe(pw))),
              width = 2400, height = 1600, res = 200); print(p); dev.off() }, silent = TRUE)
  }

  # Communication patterns and embeddings
  DIR_FIG_PAT <- ensure_dir(DIR_FIG, "patterns")
  DIR_FIG_EMB <- ensure_dir(DIR_FIG, "embeddings")

  K_out <- .pick_k(cellChat, "outgoing", DIR_FIG_PAT, SAMPLE_ID)
  if (K_out >= 2) {
    cellChat <- tryCatch(identifyCommunicationPatterns(cellChat, pattern = "outgoing", k = K_out),
                         error = function(e) { message("Skipping outgoing patterns: ", e$message); cellChat })
    png(file.path(DIR_FIG_PAT, sprintf("%s__river__outgoing.png", SAMPLE_ID)), width=2000, height=1400, res=200)
    print(netAnalysis_river(cellChat, pattern = "outgoing")); dev.off()
    png(file.path(DIR_FIG_PAT, sprintf("%s__dot__outgoing.png", SAMPLE_ID)), width=2000, height=1400, res=200)
    print(netAnalysis_dot(cellChat, pattern = "outgoing")); dev.off()
  }

  K_in <- .pick_k(cellChat, "incoming", DIR_FIG_PAT, SAMPLE_ID)
  if (K_in >= 2) {
    cellChat <- tryCatch(identifyCommunicationPatterns(cellChat, pattern = "incoming", k = K_in),
                         error = function(e) { message("Skipping incoming patterns: ", e$message); cellChat })
    png(file.path(DIR_FIG_PAT, sprintf("%s__river__incoming.png", SAMPLE_ID)), width=2000, height=1400, res=200)
    print(netAnalysis_river(cellChat, pattern = "incoming")); dev.off()
    png(file.path(DIR_FIG_PAT, sprintf("%s__dot__incoming.png", SAMPLE_ID)), width=2000, height=1400, res=200)
    print(netAnalysis_dot(cellChat, pattern = "incoming")); dev.off()
  }

  for (sim_type in c("functional","structural")) {
    cellChat <- tryCatch(computeNetSimilarity(cellChat, type = sim_type),
                         error = function(e) { message(sim_type, " similarity skip: ", e$message); cellChat })
    cellChat <- tryCatch(netEmbedding(cellChat, type = sim_type),
                         error = function(e) { message(sim_type, " embedding skip: ", e$message); cellChat })
    cellChat <- tryCatch(netClustering(cellChat, type = sim_type),
                         error = function(e) { message(sim_type, " clustering skip: ", e$message); cellChat })
    png(file.path(DIR_FIG_EMB, sprintf("%s__embedding__%s.png", SAMPLE_ID, sim_type)),
        width=1800, height=1400, res=200)
    tryCatch(print(netVisual_embedding(cellChat, type = sim_type, label.size = 3.5)),
             error = function(e) message(sim_type, " embedding plot skip: ", e$message))
    dev.off()
    png(file.path(DIR_FIG_EMB, sprintf("%s__embeddingZoomIn__%s.png", SAMPLE_ID, sim_type)),
        width=2000, height=1600, res=200)
    tryCatch(print(netVisual_embeddingZoomIn(cellChat, type = sim_type, nCol = 2)),
             error = function(e) message(sim_type, " zoom-in skip: ", e$message))
    dev.off()
  }

  # Save serialized CellChat object
  saveRDS(cellChat, file = file.path(OUTDIR, sprintf("%s__cellchat_object.rds", SAMPLE_ID)))
  message(sprintf("[DONE] Batch %s complete. Output: %s", SAMPLE_ID, OUTDIR))
}


# =============================================================================
# MAIN: run all batches
# =============================================================================
all_samples <- levels(droplevels(sce$Sample))
message(sprintf("Running %d batches: %s", length(all_samples), paste(all_samples, collapse = ", ")))

for (sid in all_samples) {
  message("\n==============================")
  message(">>> Batch: ", sid)
  message("==============================")
  try({
    run_single_batch(
      SAMPLE_ID     = sid,
      FOCUS_SOURCES = c("TAM","M2-Macro","Hep","Malignant","TEC"),
      FILTER_METHOD = "topN",
      TOP_N         = 5,
      WORKERS       = WORKERS,
      OUT_ROOT      = OUTPUT_DIR,
      MIN_CELLS     = 0
    )
    gc()
  }, silent = FALSE)
}
