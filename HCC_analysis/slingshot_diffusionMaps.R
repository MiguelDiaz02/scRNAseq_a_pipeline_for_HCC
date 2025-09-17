setwd("/home/mdiaz/sc_liver_data/R_scripts/slingshot_results/")

options(zellkonverter.use.basilisk = FALSE)

library(reticulate)
use_condaenv("zellenv", required = TRUE)

py_config()  #

suppressPackageStartupMessages({
  library(scran)
  library(scater)
  library(igraph)
  library(Seurat)
  library(SeuratDisk)
  library(zellkonverter)
  library(SingleCellExperiment)
  library(SeuratWrappers)
  library(cowplot)
  library(slingshot)
  library(rafalib)
  library(plotly)
  options(rgl.printRglwidget = TRUE)
  library(Matrix)
  library(sparseMatrixStats)
  library(tradeSeq)
  library(patchwork)
})


# sceasy::convertFormat(
#   "/home/mdiaz/sc_liver_data/pseudotime_checkpoints/adata_macrophages.h5ad",
#   from = "anndata",
#   to = "seurat",
#   outFile = "adata_mono_macro.rds"
# )

seurat_obj <- readRDS("/home/mdiaz/sc_liver_data/R_scripts/adata_hep_malign.rds")

# Verifica contenido
seurat_obj

head(seurat_obj@meta.data)


# Normalización de datos
seurat_obj <- NormalizeData(object = seurat_obj, verbose = FALSE)

# Selección de características variables
seurat_obj <- FindVariableFeatures(object = seurat_obj, nfeatures = 2000, verbose = FALSE, selection.method = 'vst')

# Escalado de datos
seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)

# Análisis de componentes principales (PCA)
seurat_obj <- RunPCA(seurat_obj, npcs = 30, verbose = FALSE)

# Encontrar vecinos para la construcción del gráfico KNN
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30, verbose = FALSE)

# Reducción de dimensionalidad con UMAP
seurat_obj <- RunUMAP(seurat_obj, reduction = "pca", dims = 1:30, verbose = FALSE)

# Visualización inicial
DimPlot(seurat_obj, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()


DefaultAssay(seurat_obj) <- "RNA"

# Filtrar tipos celulares con al menos 50 células
cell_counts <- table(seurat_obj$cell_type)
valid_cell_types <- names(cell_counts[cell_counts >= 50])
seurat_obj <- subset(seurat_obj, subset = cell_type %in% valid_cell_types)

#############################################
#############################################

pal <- c(scales::hue_pal()(8),RColorBrewer::brewer.pal(9,"Set1"),RColorBrewer::brewer.pal(8,"Set2") )
set.seed(1); pal <- rep( sample( pal , length(pal) ) , 200 )

#Add graph to the base R graphics plot
draw_graph <- function( layout , graph , lwd = 0.2 , col = "grey" ){
  res <- rep(x = 1:(length(graph@p)-1) , times = (graph@p[-1] - graph@p[-length(graph@p)]) )
  segments(x0 = layout[graph@i+1,1], x1=layout[res,1],
           y0 = layout[graph@i+1,2], y1=layout[res,2], lwd=lwd , col=col )}

# Visualización de agrupamientos/metadatos clave
vars <- c("Batch", "dataset", "leiden", "cell_type")
pl <- list()
for(i in vars){
  pl[[i]] <- DimPlot(seurat_obj, group.by = i, label = TRUE) + theme_void() + NoLegend()
}
plot_grid(plotlist = pl)

# Tabla de conteo por cluster leiden
table(seurat_obj$leiden)

# Extracción de embeddings y estructuras necesarias
NORM_COUNTS      <- seurat_obj@assays$RNA@data
UMAP2            <- seurat_obj@reductions$umap@cell.embeddings


NORM_COUNTS      <- seurat_obj@assays$RNA@data
UMAP2            <- seurat_obj@reductions$umap@cell.embeddings
HARMONY <- seurat_obj@reductions$pca_harmony@cell.embeddings
PCA              <- seurat_obj@reductions$pca@cell.embeddings
PCA_loadings     <- seurat_obj@reductions$pca@feature.loadings
clustering       <- factor(seurat_obj$leiden)
KNN              <- seurat_obj@graphs$RNA_snn  # o verifica con names(seurat_obj@graphs)

# Calcular centroides de clusters para visualización
mm <- sparse.model.matrix(~ 0 + clustering)
colnames(mm) <- levels(clustering)

centroids2d <- as.matrix(t(t(UMAP2) %*% mm) / Matrix::colSums(mm))


stopifnot(identical(rownames(UMAP2), rownames(KNN)))  # Validación

# Vector lógico para verificar que cada célula tiene vecinos
valid_cells <- Matrix::rowSums(KNN) > 0

# Inicializar matriz vacía
neighbor_avg <- matrix(NA, nrow = nrow(UMAP2), ncol = ncol(UMAP2))
rownames(neighbor_avg) <- rownames(UMAP2)

# Celdas válidas
valid_idx <- which(valid_cells)

# Multiplicación matriz por matriz
num <- KNN[valid_idx, ] %*% UMAP2  # matriz (n_valid x 2)

# Normalización explícita por fila
rsums <- Matrix::rowSums(KNN[valid_idx, ])
neighbor_avg_valid <- num / rsums  # se recicla por columna

# Asegurar que sigue siendo matriz
neighbor_avg_valid <- as.matrix(neighbor_avg_valid)
rownames(neighbor_avg_valid) <- rownames(UMAP2)[valid_idx]

# Insertar en matriz general
neighbor_avg <- matrix(NA, nrow = nrow(UMAP2), ncol = ncol(UMAP2),
                       dimnames = list(rownames(UMAP2), colnames(UMAP2)))
neighbor_avg[valid_idx, ] <- neighbor_avg_valid

d <- rowSums((neighbor_avg - UMAP2)^2)^0.5
cutoff <- mean(d, na.rm = TRUE) + 5 * sd(d, na.rm = TRUE)
to_keep <- d < cutoff

# Reemplazo robusto
new_UMAP2 <- UMAP2
outlier_idx <- which(!to_keep)
new_UMAP2[outlier_idx, ] <- neighbor_avg[outlier_idx, , drop = FALSE]

# Validar que las dimensiones coincidan
stopifnot(nrow(new_UMAP2[outlier_idx, ]) == nrow(neighbor_avg[outlier_idx, , drop = FALSE]))

# Asignación robusta
new_UMAP2[outlier_idx, ] <- neighbor_avg[outlier_idx, , drop = FALSE]

# Recalcular centroides con el nuevo embedding
new_centroids2d <- as.matrix(t(t(new_UMAP2) %*% mm) / Matrix::colSums(mm))

# Paleta de colores para los clusters
pal <- scater:::.get_palette("tableau10medium")
pal <- rep(pal, length.out = length(levels(clustering)))

# Graficar y guardar
png("umap_graph_filtered.png", width = 1200, height = 1000, res = 150)

par(mar = c(4, 4, 4, 4))
plot(new_UMAP2, type = "n", xlab = "UMAP1", ylab = "UMAP2", main = "Filtered UMAP with Graph Overlay")
plot(KNN, layout = new_UMAP2, add = TRUE, edge.color = "grey80", edge.width = 0.5)

points(
  new_UMAP2,
  cex = ifelse(!to_keep, 1, 0.3),
  lwd = ifelse(!to_keep, 2, 0),
  bg = pal[clustering],
  pch = 21
)

text(new_centroids2d, labels = rownames(new_centroids2d), cex = 1, font = 2)
dev.off()

#############
#Diff maps###
#############

pcs <- Embeddings(seurat_obj, "pca")[, 1:30]
dm <- DiffusionMap(pcs)
dm_coords <- as.matrix(dm@eigenvectors)
diffmap <- CreateDimReducObject(embeddings = dm_coords,
                                key = "DM_",
                                assay = DefaultAssay(seurat_obj))
seurat_obj[["diffmap"]] <- diffmap

DM_embed <- Embeddings(seurat_obj, "diffmap")[, 1:2]
stopifnot(identical(rownames(DM_embed), rownames(KNN)))

# Matriz modelo para clusters
mm <- sparse.model.matrix(~ 0 + clustering)
colnames(mm) <- levels(clustering)

# Centroides
centroids2d <- as.matrix(t(t(DM_embed) %*% mm) / Matrix::colSums(mm))

# Celdas con vecinos
valid_cells <- Matrix::rowSums(KNN) > 0
valid_idx <- which(valid_cells)

# Vecinos promedio
num <- KNN[valid_idx, ] %*% DM_embed
rsums <- Matrix::rowSums(KNN[valid_idx, ])
neighbor_avg_valid <- num / rsums
neighbor_avg_valid <- as.matrix(neighbor_avg_valid)
rownames(neighbor_avg_valid) <- rownames(DM_embed)[valid_idx]

neighbor_avg <- matrix(NA, nrow = nrow(DM_embed), ncol = ncol(DM_embed),
                       dimnames = list(rownames(DM_embed), colnames(DM_embed)))
neighbor_avg[valid_idx, ] <- neighbor_avg_valid

# Outliers
d <- sqrt(rowSums((neighbor_avg - DM_embed)^2))
cutoff <- mean(d, na.rm = TRUE) + 5 * sd(d, na.rm = TRUE)
to_keep <- d < cutoff

# Sustituir outliers
new_DM_embed <- DM_embed
outlier_idx <- which(!to_keep)
new_DM_embed[outlier_idx, ] <- neighbor_avg[outlier_idx, , drop = FALSE]

# Nuevos centroides
new_dm_centroids2d <- as.matrix(t(t(new_DM_embed) %*% mm) / Matrix::colSums(mm))

# Graficar
png("diffmap_graph_filtered.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(new_DM_embed, type = "n", xlab = "DC1", ylab = "DC2", main = "Filtered Diffusion Map with Graph Overlay")
points(new_DM_embed,
       cex = ifelse(!to_keep, 1, 0.3),
       lwd = ifelse(!to_keep, 2, 0),
       bg = pal[clustering],
       pch = 21)
text(new_dm_centroids2d, labels = rownames(new_dm_centroids2d), cex = 1, font = 2)
dev.off()

##################################
## TRAJECTORY INFERENCE WITH SLINGSHOT (DIFFUSION MAP)
#######################################
library(slingshot)
library(distances)

# Validar consistencia de nombres
stopifnot(identical(rownames(new_DM_embed), names(clustering)))

# Matriz de centroides por cluster usando Diffusion Map filtrado
cluster_ids <- levels(clustering)
centroids <- sapply(cluster_ids, function(cl) {
  cells <- which(clustering == cl)
  colMeans(new_DM_embed[cells, , drop = FALSE])
}, simplify = "array")

centroids <- t(centroids)
rownames(centroids) <- cluster_ids

# Verificar NaNs en matriz de distancias
dist_mat <- as.matrix(dist(centroids))
which(is.na(dist_mat), arr.ind = TRUE)

# Filtrar clusters con al menos 5 células
valid_clusters <- names(which(table(clustering) >= 5))
valid_cells <- clustering %in% valid_clusters

# Subconjuntos válidos
new_DM_valid <- new_DM_embed[valid_cells, ]
clustering_valid <- droplevels(clustering[valid_cells])
KNN_valid <- KNN[valid_cells, valid_cells]
to_keep_valid <- to_keep[valid_cells]

# Ejecutar Slingshot
set.seed(1)
lineages <- getLineages(
  data = new_DM_valid,
  clusterLabels = clustering_valid
)
lineages <- as.SlingshotDataSet(lineages)

# Visualizar y guardar
png("slingshot_trajectory_diffmap.png", width = 1200, height = 1000, res = 150)
par(mfrow = c(1, 2), mar = c(4, 4, 4, 4))

# Plot A: trayectorias sobre Diffusion Map filtrado
plot(new_DM_valid, col = pal[clustering_valid], cex = 0.5, pch = 16,
     main = "Lineages on filtered Diffusion Map")
lines(lineages, lwd = 2, col = "black", cex = 3)
text(new_dm_centroids2d, labels = rownames(new_dm_centroids2d),
     cex = 0.8, font = 2, col = "white")

# Plot B: grafo con outliers marcados
plot(new_DM_valid, type = "n", main = "Diffusion Map + KNN Graph + Outliers")
draw_graph(layout = new_DM_valid, graph = KNN_valid)
points(new_DM_valid,
       cex = ifelse(!to_keep_valid, 1, 0.3),
       lwd = ifelse(!to_keep_valid, 2, 0),
       bg = pal[clustering_valid], pch = 21)
text(new_dm_centroids2d, labels = rownames(new_dm_centroids2d),
     cex = 0.8, font = 2)

dev.off()

# Imprimir objeto resultante
print(lineages)

library(dplyr)

# 1. Obtener expresión media por gen en el cluster 11
cluster_id <- "11"
cells_cluster_11 <- colnames(seurat_obj)[seurat_obj$leiden == cluster_id]

expr_matrix <- GetAssayData(seurat_obj, assay = "RNA", slot = "data")
mean_expr_cluster11 <- Matrix::rowMeans(expr_matrix[, cells_cluster_11, drop = FALSE])

# 2. Genes más expresados en cluster 11
top_genes <- sort(mean_expr_cluster11, decreasing = TRUE)[1:20]
print(top_genes)

# 3. Expresión media de MALAT1 por cluster
meta <- seurat_obj@meta.data
meta$MALAT1 <- Seurat::FetchData(seurat_obj, vars = "MALAT1")[, 1]

cluster_means <- meta %>%
  group_by(leiden) %>%
  summarise(mean_expr = mean(MALAT1)) %>%
  arrange(desc(mean_expr))

print(cluster_means)

# 4. Inferir trayectoria desde el cluster 11 usando Diffusion Map
set.seed(1)
lineages <- as.SlingshotDataSet(
  getLineages(
    data = new_DM_valid,
    clusterLabels = clustering_valid,
    start.clus = "11"
  )
)

# Asegurar que el embedding esté bien asignado
lineages@reducedDim <- new_DM_valid

# 5. Graficar trayectorias desde cluster MALAT1-alto
png("slingshot_trajectory-MALAT1-marker_DM.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(new_DM_valid, col = pal[clustering_valid], cex = 0.5, pch = 16,
     main = "Slingshot trajectories from MALAT1-high cluster (Diffusion Map)")
lines(lineages, lwd = 2, col = 'black')
text(new_dm_centroids2d, labels = rownames(new_dm_centroids2d), cex = 0.8, font = 2, col = "white")
dev.off()

# 6. Graficar Diffusion Map completo con etiquetas de clusters
pal <- rainbow(length(levels(clustering)))

png("full_DM_with_graph.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(new_DM_embed, col = pal[clustering], cex = 0.5, pch = 16,
     main = "Diffusion Map con todos los clusters")
draw_graph(layout = new_DM_embed, graph = KNN)
text(new_dm_centroids2d, labels = rownames(new_dm_centroids2d), cex = 1, font = 2)
dev.off()

library(Matrix)
library(pheatmap)
library(RColorBrewer)

# Paleta para clusters originales
pal <- colorRampPalette(brewer.pal(12, "Set3"))(length(levels(clustering)))

# Matriz modelo (1-hot) para los clusters
mm <- sparse.model.matrix(~ 0 + clustering)
colnames(mm) <- levels(clustering)

# Calcular conexiones entre clusters en base al grafo de vecinos (KNN)
d <- t(KNN %*% mm) %*% mm /
  (t(t(colSums(mm))) %*% t(colSums(mm)))^(1/2)

diag(d) <- 0
d <- drop0(d)  # eliminar ceros explícitos (matriz dispersa)

# Visualizar mapa de calor de conexiones
pheatmap(d, clustering_method = "ward.D2", main = "Conexiones entre clusters (Diffusion Map)")

# Histograma de valores de conexión
hist(d@x, breaks = 50, main = "Histograma de valores de conexión")
cutoff <- 1.2 * (sum((d@x^2)) / (length(d@x) - 1))^(1/2)
abline(v = cutoff, col = "red", xpd = FALSE)

# Identificar pares de clusters altamente conectados
to_merge <- drop0((d > cutoff) * 1)
to_merge <- to_merge * lower.tri(to_merge)
diag(to_merge)[rowSums(to_merge) == 0] <- 1

# Visualizar qué clusters se agruparán
pheatmap(to_merge, cluster_rows = FALSE, cluster_cols = FALSE, main = "Mapeo de merges (DM)")

# Generar tabla de mapeo de clusters
mappings <- cbind(
  from = colnames(to_merge)[to_merge@i + 1],
  to   = colnames(to_merge)[rep(1:(length(to_merge@p) - 1), times = diff(to_merge@p))]
)

# Aplicar merge al clustering original
merged_clustering <- factor(mappings[match(clustering, mappings[, "from"]), "to"])

# Matriz modelo para clusters mergeados
merged_mm <- sparse.model.matrix(~ 0 + merged_clustering)
colnames(merged_mm) <- levels(merged_clustering)

# Calcular nuevos centroides sobre Diffusion Map
merged_centroids2d <- as.matrix(t(t(new_DM_embed) %*% merged_mm) / Matrix::colSums(merged_mm))

# Graficar clusters mergeados sobre Diffusion Map
png("merged_clusters_dm.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(new_DM_embed, type = "n", main = "Clusters mergeados (Diffusion Map)")
draw_graph(layout = new_DM_embed, graph = KNN)
points(new_DM_embed, col = pal[merged_clustering], cex = 0.5, pch = 16)
text(merged_centroids2d, labels = rownames(merged_centroids2d), cex = 1, font = 1)
dev.off()

library(ggplot2)

# Extraer embedding diffusion map (asegúrate que el nombre "diffmap" es correcto en tu objeto)
diff_embed <- Embeddings(seurat_obj, "diffmap")
df_embed <- as.data.frame(diff_embed)

# Agregar etiqueta 'label' que contiene condición (healthy, HCC, etc)
df_embed$label <- seurat_obj$label  # o seurat_obj@meta.data$label

# Graficar con ggplot2
p <- ggplot(df_embed, aes(x = DM_1, y = DM_2, color = label)) +
  geom_point(size = 1) +
  theme_minimal() +
  labs(title = "Diffusion Map coloreado por condición (label)", color = "Condición")

ggsave("diffusion_map_por_condicion.png", plot = p, width = 8, height = 6, dpi = 300)

# Filtrar células según tus criterios previos
obj_new <- seurat_obj[, cell_to_keep]

# Actualizar el grafo KNN filtrado
obj_new@graphs <- list(RNA_snn = filt_KNN)  # usar el nombre correcto del grafo en Seurat

# Asignar clusters mergeados filtrados
obj_new$clusters_use <- factor(merged_filt_clustering)

saveRDS(obj_new, file = "/home/mdiaz/sc_liver_data/pseudotime_checkpoints/trajectory_seurat_DiffusionMaps_Hepatocytes.rds")



#A continuación necesitamos ver los top marcadores por condicióny por rama presente en el pseudotiempo

Idents(obj_new) <- obj_new$clusters_use

clusters_target <- c("9", "1", "18", "13", "2", "11")
marker_list <- list()

for (cl in clusters_target) {
  marker_list[[cl]] <- FindMarkers(
    object = obj_new,
    ident.1 = cl,
    ident.2 = NULL,  # compara contra todos los demás
    only.pos = TRUE, # solo genes sobreexpresados en el cluster
    min.pct = 0.25,
    logfc.threshold = 0.25
  )
  marker_list[[cl]]$gene <- rownames(marker_list[[cl]])
  marker_list[[cl]]$cluster <- cl
}

all_markers <- bind_rows(marker_list)


# Contar en cuántos clusters aparece cada gen
gene_counts <- all_markers %>%
  group_by(gene) %>%
  summarise(n_clusters = n())

# Filtrar los genes que aparecen solo en un cluster
unique_genes <- gene_counts %>%
  filter(n_clusters == 1)

# Unir con la tabla original para quedarte con solo esos genes
unique_markers <- all_markers %>%
  filter(gene %in% unique_genes$gene)

top1_markers <- unique_markers %>%
  group_by(cluster) %>%
  top_n(n = 1, wt = avg_log2FC) %>%  # usa avg_logFC si estás en Seurat v3
  arrange(cluster)

print(top1_markers[, c("cluster", "gene", "avg_log2FC", "p_val_adj")])

#Marcadores por cluster de interés
# Cluster 1 (healthy): MTRNR2L12
#Cluster 18 (HCC): AKR1B10
# Cluster 11 (HCC): POLR2J3 
# Cluster 2 (HCC): DANCR

# Paleta de colores (ajusta el número si tienes más clusters)
# pal <- rainbow(length(levels(obj_new$clusters_use)))
# 
# # Extraer el embedding del Diffusion Map
# new_DM_embed <- Embeddings(obj_new, "diffmap")
# 
# # Grafo
# KNN <- obj_new@graphs$KNN
# 
# # Clustering mergeado
# merged_clustering <- obj_new$clusters_use
# 
# # Matriz modelo para clusters mergeados
# library(Matrix)
# merged_mm <- sparse.model.matrix(~ 0 + merged_clustering)
# colnames(merged_mm) <- levels(merged_clustering)
# 
# # Calcular centroides en espacio Diffusion Map
# merged_centroids2d <- as.matrix(t(t(new_DM_embed) %*% merged_mm) / Matrix::colSums(merged_mm))
# 
# # Función para graficar el grafo
# draw_graph <- function(layout, graph, lwd = 0.2, col = "grey") {
#   res <- rep(x = 1:(length(graph@p) - 1), times = (graph@p[-1] - graph@p[-length(graph@p)]))
#   segments(x0 = layout[graph@i + 1, 1], x1 = layout[res, 1],
#            y0 = layout[graph@i + 1, 2], y1 = layout[res, 2], lwd = lwd, col = col)
# }
# 
# # Guardar visualización
# png("merged_clusters_dm.png", width = 1200, height = 1000, res = 150)
# par(mar = c(4, 4, 4, 4))
# plot(new_DM_embed, type = "n", main = "Clusters mergeados (Diffusion Map)")
# draw_graph(layout = new_DM_valid, graph = KNN_valid)
# points(new_DM_embed, col = pal[merged_clustering], cex = 0.5, pch = 16)
# text(merged_centroids2d, labels = rownames(merged_centroids2d), cex = 1, font = 1)
# dev.off()

#####################
# Starting to plot ##
#####################
setwd( "/scratch/home/mdiaz/sc_liver_data/R_scripts/slingshot_results/difussion_plots/")

vars <- c("MTRNR2L12", "POLR2J3", "AKR1B10", "DANCR")
pl <- list()

# Paleta para condición
label_colors <- c("healthy" = "#08306B", "HCC" = "#E41A1C")

# Paleta para tipo celular (solo dos tipos)
cell_types <- unique(obj_new$cell_type)
stopifnot(length(cell_types) == 2)
cell_type_colors <- setNames(c("#4DAF4A", "#984EA3"), cell_types)

# Plot condición (label)
pl[["label"]] <- DimPlot(
  obj_new,
  group.by = "label",
  reduction = "diffmap",
  cols = label_colors
) + theme_void() + 
  ggtitle("Condición: HCC / Healthy") +
  theme(legend.position = "right")

# Plot tipo celular (cell_type)
pl[["cell_type"]] <- DimPlot(
  obj_new,
  group.by = "cell_type",
  reduction = "diffmap",
  cols = cell_type_colors
) + theme_void() + 
  ggtitle("Tipo celular") +
  theme(legend.position = "right")

# Plots para cada gen de interés, con gradiente color rojo
for (gene in vars) {
  expr_vals <- FetchData(obj_new, vars = gene)[, 1]
  df_plot <- as.data.frame(Embeddings(obj_new, reduction = "diffmap")[, 1:2])
  df_plot$expression <- expr_vals
  
  p <- ggplot(df_plot, aes(x = DM_1, y = DM_2)) +
    geom_point(aes(color = expression), size = 0.5) +
    scale_color_gradient(low = "lightgrey", high = "#E41A1C", na.value = "lightgrey") +
    theme_void() + NoLegend() +
    ggtitle(gene)
  
  pl[[gene]] <- p
}

# Armar layout
top_row <- pl[["label"]] | pl[["cell_type"]] | pl[["MTRNR2L12"]]
bottom_row <- pl[["POLR2J3"]] | pl[["AKR1B10"]] | pl[["DANCR"]]
combined_plot <- top_row / bottom_row

# Guardar
ggsave("marker_genes_diffmap_custom_colors.png", plot = combined_plot, width = 15, height = 10, dpi = 300)
ggsave("marker_genes_diffmap_custom_colors.pdf", plot = combined_plot, width = 15, height = 10)


# CLuster annotation:

# Crear vector con nombres nuevos solo para los clusters de interés
new_clust <- as.character(obj_new$clusters_use)

new_clust[new_clust == "1"]  <- "1-MTRNR2L12"  # healthy
new_clust[new_clust == "18"] <- "18-AKR1B10"   # HCC
new_clust[new_clust == "11"] <- "11-POLR2J3"   # HCC
new_clust[new_clust == "2"]  <- "2-DANCR"      # HCC

# Dejar el resto de clusters como están
obj_new$clust_annot <- factor(new_clust)

# Graficar sobre Diffusion Map
p <- DimPlot(obj_new, group.by = "clust_annot", reduction = "diffmap", label = TRUE) +
  theme_void() + NoLegend() +
  ggtitle("Clusters anotados por marcador")

# Guardar como imagen
ggsave("annotated_clusters_diffmap.png", plot = p, width = 8, height = 6, dpi = 300)

# LINAJES EN EL DATASET:

# Definir puntos inicial y finales
start_cluster <- "1-MTRNR2L12"
end_clusters <- c("2-DANCR", "11-POLR2J3", "18-AKR1B10")

# Extraer embedding de Diffusion Map
dm_embed <- obj_new@reductions$diffmap@cell.embeddings

# Definir lineajes con slingshot
set.seed(1)
lineages <- getLineages(
  data = dm_embed,
  clusterLabels = obj_new$clust_annot,
  start.clus = start_cluster,
  end.clus = end_clusters,
  dist.method = "mnn"
)
lineages <- as.SlingshotDataSet(lineages)

# Asignar nuevamente la reducción Diffusion Map (para visualización)
lineages@reducedDim <- dm_embed

# Calcular centroides por cluster anotado
cluster_ids <- levels(obj_new$clust_annot)
mm <- sparse.model.matrix(~ 0 + obj_new$clust_annot)
colnames(mm) <- levels(obj_new$clust_annot)
centroids2d <- as.matrix(t(t(dm_embed) %*% mm) / Matrix::colSums(mm))
rownames(centroids2d) <- cluster_ids

# Paleta
pal <- scales::hue_pal()(length(cluster_ids))
names(pal) <- cluster_ids

# Plot y guardar
png("trajectory_diffmap_lineages.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(dm_embed, col = pal[obj_new$clust_annot], cex = 0.5, pch = 16,
     main = "Trajectories over Diffusion Map")
lines(lineages, lwd = 1.5, col = "black", cex = 2)
text(centroids2d, labels = rownames(centroids2d), cex = 0.8, font = 2, col = "white")
dev.off()
#####################
# SUAVIZADO DE CURVAS
######################

# Define curves
curves <- as.SlingshotDataSet(getCurves(
  data          = lineages,
  thresh        = 1e-1,
  stretch       = 1e-1,
  allow.breaks  = F,
))

curves <- as.SlingshotDataSet(curves)

# Confirmar
print(curves)

# Reasignar reducción (por seguridad)
curves@reducedDim <- obj_new@reductions$diffmap@cell.embeddings

# Paleta de colores para clust_annot
cluster_ids <- levels(obj_new$clust_annot)
pal <- scales::hue_pal()(length(cluster_ids))
names(pal) <- cluster_ids

# Plot y guardar
png("trajectory_diffmap_curves.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(obj_new@reductions$diffmap@cell.embeddings,
     col = pal[obj_new$clust_annot], pch = 16,
     main = "Fitted curves over Diffusion Map")
lines(curves, lwd = 2, col = "black")
text(centroids2d, labels = rownames(centroids2d), cex = 1, font = 2)
dev.off()

############
#Cálculo de pseudotiempo
#####################

# Recalcular pseudotiempo si es necesario
pseudotime <- slingPseudotime(curves, na = FALSE)
x <- rowMeans(pseudotime, na.rm = TRUE)
x <- x / max(x, na.rm = TRUE)
o <- order(x)

# Etiquetas personalizadas por cluster
cluster_labels <- c("1" = "MTRNR2L12", "18" = "AKR1B10", "11" = "POLR2J3", "2" = "DANCR")
clust_ids <- names(cluster_labels)

# Calcular centroides por cluster
clust_vec <- as.character(obj_new$clusters_use)
centroids_sel <- sapply(clust_ids, function(cl) {
  which_cells <- which(clust_vec == cl)
  colMeans(obj_new@reductions$diffmap@cell.embeddings[which_cells, , drop = FALSE])
}, simplify = "array")
centroids_sel <- t(centroids_sel)
rownames(centroids_sel) <- unname(cluster_labels)

# Plot con pseudotiempo
png("trajectory_diffmap_pseudotime_custom_labels.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))

# Usar una paleta más suave
cols_soft <- colorRampPalette(c("#d0d1e6", "#74a9cf", "#0570b0", "#023858"))(99)

plot(
  obj_new@reductions$diffmap@cell.embeddings[o, ],
  main = "Diffusion Map con pseudotiempo y etiquetas específicas",
  pch = 16,
  cex = 0.4,
  axes = FALSE,
  xlab = "", ylab = "",
  col = cols_soft[x[o] * 98 + 1]
)

# Añadir puntos blancos donde van las etiquetas
points(centroids_sel, cex = 2.5, pch = 16, col = "#FFFFFFAA")

# Añadir texto de etiquetas específicas
text(centroids_sel, labels = rownames(centroids_sel), cex = 1.2, font = 2)

dev.off()


#####################
# SAVE & LOAD#######
####################

# # Guardar objeto Seurat con todos los embeddings y metadatos
# saveRDS(obj_new, file = "/home/mdiaz/sc_liver_data/pseudotime_checkpoints/obj_new_final.rds")
# 
# # Guardar SlingshotDataSet con curvas
# saveRDS(curves, file = "/home/mdiaz/sc_liver_data/pseudotime_checkpoints/slingshot_curves_final.rds")
# 
# # (Opcional) Guardar pseudotiempo y pesos de las curvas
# saveRDS(pseudotime, file = "/home/mdiaz/sc_liver_data/pseudotime_checkpoints/pseudotime_final.rds")
# saveRDS(cellWeights, file = "/home/mdiaz/sc_liver_data/pseudotime_checkpoints/curve_weights_final.rds")
# 
# 
# #load
# # Cargar objetos guardados
# obj_new <- readRDS("/home/mdiaz/sc_liver_data/pseudotime_checkpoints/obj_new_final.rds")
# curves <- readRDS("/home/mdiaz/sc_liver_data/pseudotime_checkpoints/slingshot_curves_final.rds")
# 
# # (Opcional) Si los guardaste también:
# pseudotime <- readRDS("/home/mdiaz/sc_liver_data/pseudotime_checkpoints/pseudotime_final.rds")
# cellWeights <- readRDS("/home/mdiaz/sc_liver_data/pseudotime_checkpoints/curve_weights_final.rds")
# 

#########################################
### Encontrando DEGs
##########################################

BiocParallel::register(BiocParallel::MulticoreParam())


