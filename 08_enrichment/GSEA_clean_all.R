#!/usr/bin/env Rscript
# =============================================================================
# GSEA_clean_all.R
# =============================================================================
# Genera dotplot + barplot de enriquecimiento GOBP (compareCluster UP vs DOWN)
# para cuatro tipos celulares, con estética uniforme y p.adjust bien formateado.
#
# Tipos celulares y CSVs de entrada:
#   Hepatocytes  → DGE_hepatocytes_CLEAN_v2.csv     (referencia limpia)
#   MC           → Malignant_vs_Healthy_Hepatocytes_CLEAN.csv
#   TAMs         → TAMs_vs_Macrophages.csv
#   TECs         → TECs_vs_Endothelial_cells.csv
#
# Output (new_figures/):
#   {type}_dotplot_CLEAN.png  y  {type}_barplot_CLEAN.png
# =============================================================================

rm(list = ls(all.names = TRUE)); gc()
options(scipen = 999, stringsAsFactors = FALSE, dplyr.summarise.inform = FALSE)

suppressMessages({
  library(tidyverse)
  library(clusterProfiler)
  library(enrichplot)
  library(ggplot2)
  library(scales)
})

cat(sprintf("clusterProfiler %s | enrichplot %s\n",
            as.character(packageVersion("clusterProfiler")),
            as.character(packageVersion("enrichplot"))))

# ── Rutas fijas ───────────────────────────────────────────────────────────────
DGE_DIR <- "/home/mdiaz/HCC_project/MERGED_analysis/DGE_csv_files"
BG_PATH <- "/home/mdiaz/sc_liver_data/enrichment_pathways_in_R/geneSets_pathways"
OUT_DIR <- "/home/mdiaz/manuscript_revision/new_figures"

PADJ_CUT  <- 0.05
MIN_GENES <- 5         # mínimo de genes en el pathway
TOP_N     <- 10        # categorías a mostrar

# ── Función principal ─────────────────────────────────────────────────────────
run_gsea <- function(csv_file, label, title_str) {
  cat(sprintf("\n=== %s ===\n", label))

  # 1. Leer y anotar
  df <- read.csv(csv_file) %>%
    filter(!duplicated(gene_symbol), !grepl("\\.", gene_symbol)) %>%
    mutate(diffexpressed = case_when(
      log2fc > 0 & padj < 0.05 ~ "UP",
      log2fc < 0 & padj < 0.05 ~ "DOWN",
      TRUE ~ "NO"
    ))

  cat(sprintf("    Genes: %d | UP: %d | DOWN: %d\n",
              nrow(df),
              sum(df$diffexpressed == "UP"),
              sum(df$diffexpressed == "DOWN")))

  # 2. Background GOBP (filtrado a genes del dataset)
  bg <- readRDS(paste0(BG_PATH, "/go.bp.RDS"))
  bg <- bg[bg$gene %in% df$gene_symbol, ]

  # 3. Gene clusters
  df_sig <- df %>% filter(diffexpressed != "NO")
  if (sum(df_sig$diffexpressed == "UP")   < 5 ||
      sum(df_sig$diffexpressed == "DOWN") < 5) {
    warning(sprintf("[%s] Muy pocos genes significativos — omitiendo.", label))
    return(invisible(NULL))
  }

  gene_clusters <- list(
    UP   = df_sig$gene_symbol[df_sig$diffexpressed == "UP"],
    DOWN = df_sig$gene_symbol[df_sig$diffexpressed == "DOWN"]
  )

  # 4. compareCluster
  cc_res <- compareCluster(
    geneClusters = gene_clusters,
    fun          = "enricher",
    TERM2GENE    = bg,
    pvalueCutoff = PADJ_CUT,
    minGSSize    = MIN_GENES
  )

  if (is.null(cc_res) || nrow(as.data.frame(cc_res)) == 0) {
    warning(sprintf("[%s] Sin pathways significativos.", label))
    return(invisible(NULL))
  }

  # Limpiar prefijos GO en Description y reemplazar guiones bajos
  cc_res@compareClusterResult$Description <-
    gsub("^GO[^_]*_", "", cc_res@compareClusterResult$Description)
  cc_res@compareClusterResult$Description <-
    gsub("_", " ", cc_res@compareClusterResult$Description)
  cc_res@compareClusterResult$Description <-
    str_to_sentence(tolower(cc_res@compareClusterResult$Description))

  n_pw <- nrow(as.data.frame(cc_res))
  cat(sprintf("    Pathways significativos: %d\n", n_pw))

  # ── Dotplot ─────────────────────────────────────────────────────────────────
  # enrichplot dotplot para compareClusterResult:
  #   x = GeneRatio, size = Count, color = p.adjust  (comportamiento por defecto)
  # Solo corregimos el formato del color (p.adjust).
  p_dot <- dotplot(cc_res, showCategory = TOP_N) +
    scale_color_gradient(
      low    = "red",
      high   = "blue",
      name   = "p.adjust",
      labels = function(x) sprintf("%.2e", x)
    ) +
    scale_size_continuous(name = "Count", range = c(2, 8)) +
    theme_bw(base_size = 12) +
    theme(
      axis.text.x      = element_text(size = 10, face = "bold"),
      axis.text.y      = element_text(size = 9,  face = "bold"),
      axis.title.x     = element_text(size = 12, face = "bold"),
      axis.title.y     = element_text(size = 12, face = "bold"),
      strip.text       = element_text(size = 12, face = "bold"),
      plot.title       = element_text(size = 14, face = "bold"),
      plot.subtitle    = element_text(size = 10, face = "bold"),
      legend.title     = element_text(size = 10, face = "bold"),
      legend.text      = element_text(size = 9,  face = "bold"),
      panel.spacing    = unit(1.5, "lines"),
      plot.margin      = margin(20, 20, 20, 20),
      legend.position  = "right"
    ) +
    labs(
      title    = title_str,
      subtitle = "GO Biological Process — enricher (UP vs DOWN)",
      x        = "Gene Ratio",
      y        = "GO Biological Process"
    )

  dot_file <- file.path(OUT_DIR, paste0(label, "_dotplot_CLEAN.png"))
  ggsave(dot_file, plot = p_dot, width = 13, height = 10, dpi = 300)
  cat(sprintf("    Guardado: %s\n", basename(dot_file)))

  # ── Barplot (manual con ggplot2; barplot.compareClusterResult roto en 1.29.x)
  bar_data <- as.data.frame(cc_res) %>%
    group_by(Cluster) %>%
    arrange(p.adjust) %>%
    slice_head(n = TOP_N) %>%
    ungroup() %>%
    mutate(
      qscore      = -log10(p.adjust),
      Description = str_wrap(Description, width = 40),
      Description = reorder(Description, qscore)
    )

  p_bar <- ggplot(bar_data, aes(x = qscore, y = Description, fill = Cluster)) +
    geom_bar(stat = "identity", width = 0.75) +
    facet_wrap(~ Cluster, scales = "free_y") +
    scale_fill_manual(values = c(UP = "#d73027", DOWN = "#4575b4")) +
    theme_bw(base_size = 12) +
    theme(
      axis.text.x      = element_text(size = 10, face = "bold"),
      axis.text.y      = element_text(size = 8.5, face = "bold"),
      axis.title.x     = element_text(size = 12, face = "bold"),
      strip.text       = element_text(size = 12, face = "bold"),
      plot.title       = element_text(size = 14, face = "bold"),
      plot.subtitle    = element_text(size = 10, face = "bold"),
      panel.spacing    = unit(1.5, "lines"),
      plot.margin      = margin(20, 20, 20, 20),
      legend.position  = "none"
    ) +
    labs(
      title    = title_str,
      subtitle = sprintf("GO Biological Process — top %d per direction", TOP_N),
      x        = expression(bold(-log[10](p.adjust))),
      y        = NULL
    )

  bar_file <- file.path(OUT_DIR, paste0(label, "_barplot_CLEAN.png"))
  ggsave(bar_file, plot = p_bar, width = 14, height = 8, dpi = 300)
  cat(sprintf("    Guardado: %s\n", basename(bar_file)))

  # Imprimir top pathways
  cat("    TOP DOWN:\n")
  as.data.frame(cc_res) %>%
    filter(Cluster == "DOWN") %>% arrange(p.adjust) %>%
    select(Description, Count, p.adjust) %>% head(5) %>%
    mutate(p.adjust = sprintf("%.2e", p.adjust)) %>%
    print()
  cat("    TOP UP:\n")
  as.data.frame(cc_res) %>%
    filter(Cluster == "UP") %>% arrange(p.adjust) %>%
    select(Description, Count, p.adjust) %>% head(5) %>%
    mutate(p.adjust = sprintf("%.2e", p.adjust)) %>%
    print()

  invisible(cc_res)
}

# ── Ejecutar los 4 tipos celulares ────────────────────────────────────────────

run_gsea(
  csv_file  = file.path(DGE_DIR, "DGE_hepatocytes_CLEAN_v2.csv"),
  label     = "Hepatocytes",
  title_str = "Hepatocytes: Tumor vs Healthy"
)

run_gsea(
  csv_file  = file.path(DGE_DIR, "Malignant_vs_Healthy_Hepatocytes_CLEAN.csv"),
  label     = "MC",
  title_str = "Malignant Cells vs Healthy Hepatocytes"
)

run_gsea(
  csv_file  = file.path(DGE_DIR, "TAMs_vs_Macrophages.csv"),
  label     = "TAMs",
  title_str = "TAMs vs Macrophages"
)

run_gsea(
  csv_file  = file.path(DGE_DIR, "TECs_vs_Endothelial_cells.csv"),
  label     = "TECs",
  title_str = "TECs vs Endothelial Cells"
)

cat("\n[OK] Todos los análisis completados.\n")
cat(sprintf("Figuras guardadas en: %s\n", OUT_DIR))
