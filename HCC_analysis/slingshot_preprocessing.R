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

##################################
## TRAJECTORY INFERENCE WITH SLINGSHOT#
#######################################
library(slingshot)

# Asegúrate de que clustering y new_UMAP2 tengan nombres consistentes
stopifnot(identical(rownames(new_UMAP2), names(clustering)))




library(Slingshot)
library(distances)

# Matriz de centroides por cluster
cluster_ids <- levels(clustering)
centroids <- sapply(cluster_ids, function(cl) {
  cells <- which(clustering == cl)
  colMeans(new_UMAP2[cells, , drop = FALSE])
}, simplify = "array")

# Transponer para que filas sean clusters
centroids <- t(centroids)
rownames(centroids) <- cluster_ids

# Ver si hay NaNs en distancias
dist_mat <- as.matrix(dist(centroids))
which(is.na(dist_mat), arr.ind = TRUE)



# Filtrar clusters con al menos 5 células
valid_clusters <- names(which(table(clustering) >= 5))
valid_cells <- clustering %in% valid_clusters

# Filtrar datos
new_UMAP2_valid <- new_UMAP2[valid_cells, ]
clustering_valid <- droplevels(clustering[valid_cells])

# Ejecutar Slingshot
set.seed(1)
lineages <- getLineages(
  data = new_UMAP2_valid,
  clusterLabels = clustering_valid
)

lineages <- as.SlingshotDataSet(lineages)

# 2. (Opcional) Reasignar la reducción por claridad visual
#lineages@reducedDim <- new_UMAP2  # esto no es necesario si usaste new_UMAP2 ya arriba

# 3. Visualizar resultados y guardar
png("slingshot_trajectory.png", width = 1200, height = 1000, res = 150)
par(mfrow = c(1, 2), mar = c(4, 4, 4, 4))

# Plot A: línea sobre UMAP filtrado
plot(new_UMAP2, col = pal[clustering], cex = 0.5, pch = 16,
     main = "Lineages on filtered UMAP")
lines(lineages, lwd = 2, col = "black", cex = 3)
text(new_centroids2d, labels = rownames(new_centroids2d),
     cex = 0.8, font = 2, col = "white")

# Plot B: grafo con outliers marcados
plot(new_UMAP2, type = "n", main = "UMAP + KNN Graph + Outliers")
draw_graph(layout = new_UMAP2, graph = KNN)
points(new_UMAP2,
       cex = ifelse(!to_keep, 1, 0.3),
       lwd = ifelse(!to_keep, 2, 0),
       bg = pal[clustering], pch = 21)
text(new_centroids2d, labels = rownames(new_centroids2d),
     cex = 0.8, font = 2)

dev.off()

print(lineages)

# Obtener expresión media por gen en el cluster 16
cluster_id <- "16"

# Extraer células del cluster 16
cells_cluster_16 <- colnames(seurat_obj)[seurat_obj$leiden == cluster_id]

# Obtener la matriz de expresión normalizada
expr_matrix <- GetAssayData(seurat_obj, assay = "RNA", slot = "data")

# Calcular promedio por gen dentro del cluster
mean_expr_cluster16 <- Matrix::rowMeans(expr_matrix[, cells_cluster_16, drop = FALSE])

# Ordenar y mostrar los genes más expresados
top_genes <- sort(mean_expr_cluster16, decreasing = TRUE)[1:20]
print(top_genes)

# Calcular expresión media por cluster (usando 'leiden')
library(dplyr)
meta <- seurat_obj@meta.data
meta$APOA2 <- Seurat::FetchData(seurat_obj, vars = "APOA2")[,1]

cluster_means <- meta %>%
  group_by(leiden) %>%
  summarise(mean_expr = mean(APOA2)) %>%
  arrange(desc(mean_expr))

print(cluster_means)

set.seed(1)

lineages <- as.SlingshotDataSet(
  getLineages(
    data = new_UMAP2_valid,
    clusterLabels = clustering_valid,
    start.clus = "16"  # usa el número real aquí
  )
)

# Asignar el embedding corregido
lineages@reducedDim <- new_UMAP2_valid

png("slingshot_trajectory-APOA2-marker.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(new_UMAP2_valid, col = pal[clustering_valid], cex = 0.5, pch = 16,
     main = "Slingshot trajectories from APOA2-high cluster")
lines(lineages, lwd = 2, col = 'black')
text(new_centroids2d, labels = rownames(new_centroids2d), cex = 0.8, font = 2, col = "white")
dev.off()


# Paleta
library(RColorBrewer)
pal <- rainbow(length(levels(clustering)))
# Graficar el UMAP completo con etiquetas
png("full_umap_with_graph.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(new_UMAP2, col = pal[clustering], cex = 0.5, pch = 16,
     main = "UMAP con todos los clusters")
draw_graph(layout = new_UMAP2, graph = KNN)
text(new_centroids2d, labels = rownames(new_centroids2d), cex = 1, font = 2)
dev.off()

# Clusters a eliminar
clusters_to_remove <- c("40", "26", "8", "23", "3", "38", "28", "19", "0", "1", "7", "15", "9", "27", "29", "12")

# Filtrar celdas que no pertenecen a esos clusters
cell_to_keep <- !(clustering %in% clusters_to_remove)

# Filtrar objetos
filt_new_UMAP2       <- new_UMAP2[cell_to_keep, ]
filt_NORM_COUNTS     <- NORM_COUNTS[, cell_to_keep]
filt_PCA             <- PCA[cell_to_keep, ]
filt_HARMONY         <- HARMONY[cell_to_keep, ]
filt_KNN             <- KNN[cell_to_keep, cell_to_keep]
filt_clustering      <- factor(clustering[cell_to_keep])

# Recalcular centroides 2D con matriz modelo
filt_mm <- sparse.model.matrix(~ 0 + filt_clustering)
colnames(filt_mm) <- levels(filt_clustering)
filt_new_centroids2d <- as.matrix(t(t(filt_new_UMAP2) %*% filt_mm) / Matrix::colSums(filt_mm))

# Paleta para clusters filtrados
pal_filt <- rainbow(length(levels(clustering)))

# Graficar y guardar
png("filtered_umap_with_graph.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(filt_new_UMAP2, type = "n", main = "UMAP filtrado")
draw_graph(layout = filt_new_UMAP2, graph = filt_KNN)
points(filt_new_UMAP2, col = pal_filt[filt_clustering], cex = 0.5, pch = 16)
text(filt_new_centroids2d, labels = rownames(filt_new_centroids2d), cex = 1, font = 1)
dev.off()

library(pheatmap)
# Paleta para clusters filtrados
pal <- colorRampPalette(brewer.pal(12, "Set3"))(length(levels(filt_clustering)))

# Crear matriz modelo para clusters filtrados
filt_mm <- sparse.model.matrix(~ 0 + filt_clustering)
colnames(filt_mm) <- levels(filt_clustering)

# Calcular conexiones entre clusters basado en la matriz KNN
d <- t(filt_KNN %*% filt_mm) %*% filt_mm /
  (t(t(colSums(filt_mm))) %*% t(colSums(filt_mm)))^(1/2)

diag(d) <- 0
d <- drop0(d)  # elimina ceros explícitos

# Visualizar mapa de calor con pheatmap
pheatmap(d, clustering_method = "ward.D2", main = "Conexiones entre clusters")

# Identificar clusters similares para merge
hist(d@x, breaks = 50, main = "Histograma de valores de conexión")
cutoff <- 1.2 * (sum((d@x^2)) / (length(d@x) - 1))^(1/2)
abline(v = cutoff, col = "red", xpd = FALSE)

to_merge <- drop0((d > cutoff) * 1)
to_merge <- to_merge * lower.tri(to_merge)
diag(to_merge)[rowSums(to_merge) == 0] <- 1

# Visualizar mapa de calor del merge
pheatmap(to_merge, cluster_rows = FALSE, cluster_cols = FALSE, main = "Mapeo de merges")

# Generar mapeo para merge de clusters
mappings <- cbind(
  from = colnames(to_merge)[to_merge@i + 1],
  to = colnames(to_merge)[rep(1:(length(to_merge@p) - 1), times = diff(to_merge@p))]
)

# Aplicar merge a clustering filtrado
merged_filt_clustering <- factor(mappings[match(filt_clustering, mappings[, "from"]), "to"])

# Matriz modelo con clusters merged
merged_filt_mm <- sparse.model.matrix(~ 0 + merged_filt_clustering)
colnames(merged_filt_mm) <- levels(merged_filt_clustering)

# Calcular nuevos centroides
merged_filt_new_centroids2d <- as.matrix(t(t(filt_new_UMAP2) %*% merged_filt_mm) / Matrix::colSums(merged_filt_mm))

# Graficar clusters merged
png("merged_clusters_umap.png", width = 1200, height = 1000, res = 150)
par(mar = c(4, 4, 4, 4))
plot(filt_new_UMAP2, type = "n", main = "Clusters Mergeados")
draw_graph(layout = filt_new_UMAP2, graph = filt_KNN)
points(filt_new_UMAP2, col = pal[merged_filt_clustering], cex = 0.5, pch = 16)
text(merged_filt_new_centroids2d, labels = rownames(merged_filt_new_centroids2d), cex = 1, font = 1)
dev.off()


#############################################################################################################################
#############################################################################################################################
# CON PCA
#############################################################################################################################
# 
# seurat_obj <- readRDS("/home/mdiaz/sc_liver_data/R_scripts/adata_hep_malign.rds")
# 
# # Verifica contenido
# seurat_obj
# 
# head(seurat_obj@meta.data)
# 
# 
# # Normalización de datos
# seurat_obj <- NormalizeData(object = seurat_obj, verbose = FALSE)
# 
# # Selección de características variables
# seurat_obj <- FindVariableFeatures(object = seurat_obj, nfeatures = 2000, verbose = FALSE, selection.method = 'vst')
# 
# # Escalado de datos
# seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
# 
# # Análisis de componentes principales (PCA)
# seurat_obj <- RunPCA(seurat_obj, npcs = 30, verbose = FALSE)
# 
# # Encontrar vecinos para la construcción del gráfico KNN
# seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30, verbose = FALSE)
# 
# # Reducción de dimensionalidad con UMAP
# seurat_obj <- RunUMAP(seurat_obj, reduction = "pca", dims = 1:30, verbose = FALSE)
# 
# # Visualización inicial
# DimPlot(seurat_obj, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()
# 
# 
# DefaultAssay(seurat_obj) <- "RNA"
# 
# # Filtrar tipos celulares con al menos 50 células
# cell_counts <- table(seurat_obj$cell_type)
# valid_cell_types <- names(cell_counts[cell_counts >= 50])
# seurat_obj <- subset(seurat_obj, subset = cell_type %in% valid_cell_types)
# 
# #############################
# #############################
# # DOWNSTREAM
# ############################
# 
# pal <- c(scales::hue_pal()(8), RColorBrewer::brewer.pal(9, "Set1"), RColorBrewer::brewer.pal(8, "Set2"))
# set.seed(1)
# pal <- rep(sample(pal, length(pal)), 200)
# 
# # Función para añadir grafo sobre plot base
# draw_graph <- function(layout, graph, lwd = 0.2, col = "grey") {
#   res <- rep(x = 1:(length(graph@p) - 1), times = (graph@p[-1] - graph@p[-length(graph@p)]))
#   segments(x0 = layout[graph@i + 1, 1], x1 = layout[res, 1],
#            y0 = layout[graph@i + 1, 2], y1 = layout[res, 2], lwd = lwd, col = col)
# }
# 
# # Visualización agrupamientos y metadatos clave
# vars <- c("Batch", "dataset", "leiden", "cell_type")
# pl <- list()
# for (i in vars) {
#   pl[[i]] <- DimPlot(seurat_obj, group.by = i, label = TRUE) + theme_void() + NoLegend()
# }
# plot_grid(plotlist = pl)
# 
# # Tabla de conteo por cluster leiden
# table(seurat_obj$leiden)
# 
# # Extracción de datos necesarios
# NORM_COUNTS <- seurat_obj@assays$RNA@data
# 
# # Usamos las 2 primeras PCs para visualización
# PCA_embed <- seurat_obj@reductions$pca@cell.embeddings[, 1:2]
# 
# HARMONY <- seurat_obj@reductions$pca_harmony@cell.embeddings
# PCA_loadings <- seurat_obj@reductions$pca@feature.loadings
# clustering <- factor(seurat_obj$leiden)
# KNN <- seurat_obj@graphs$RNA_snn
# 
# # Matriz modelo para clusters
# mm <- sparse.model.matrix(~ 0 + clustering)
# colnames(mm) <- levels(clustering)
# 
# # Centroides en espacio PCA
# centroids2d <- as.matrix(t(t(PCA_embed) %*% mm) / Matrix::colSums(mm))
# 
# stopifnot(identical(rownames(PCA_embed), rownames(KNN)))
# 
# # Celdas con vecinos
# valid_cells <- Matrix::rowSums(KNN) > 0
# 
# # Matriz vacía para vecinos promedio
# neighbor_avg <- matrix(NA, nrow = nrow(PCA_embed), ncol = ncol(PCA_embed))
# rownames(neighbor_avg) <- rownames(PCA_embed)
# 
# # Índices válidos
# valid_idx <- which(valid_cells)
# 
# # Producto matriz vecinos por embedding PCA
# num <- KNN[valid_idx, ] %*% PCA_embed
# 
# # Normalización por fila
# rsums <- Matrix::rowSums(KNN[valid_idx, ])
# neighbor_avg_valid <- num / rsums
# 
# neighbor_avg_valid <- as.matrix(neighbor_avg_valid)
# rownames(neighbor_avg_valid) <- rownames(PCA_embed)[valid_idx]
# 
# neighbor_avg <- matrix(NA, nrow = nrow(PCA_embed), ncol = ncol(PCA_embed),
#                        dimnames = list(rownames(PCA_embed), colnames(PCA_embed)))
# neighbor_avg[valid_idx, ] <- neighbor_avg_valid
# 
# # Distancia euclídea entre embedding original y vecinos promedio
# d <- sqrt(rowSums((neighbor_avg - PCA_embed)^2))
# cutoff <- mean(d, na.rm = TRUE) + 5 * sd(d, na.rm = TRUE)
# to_keep <- d < cutoff
# 
# # Reemplazo robusto de outliers
# new_PCA_embed <- PCA_embed
# outlier_idx <- which(!to_keep)
# new_PCA_embed[outlier_idx, ] <- neighbor_avg[outlier_idx, , drop = FALSE]
# 
# stopifnot(nrow(new_PCA_embed[outlier_idx, ]) == nrow(neighbor_avg[outlier_idx, , drop = FALSE]))
# 
# # Recalcular centroides con el nuevo embedding corregido
# new_centroids2d <- as.matrix(t(t(new_PCA_embed) %*% mm) / Matrix::colSums(mm))
# 
# # Paleta de colores para los clusters
# pal <- scater:::.get_palette("tableau10medium")
# pal <- rep(pal, length.out = length(levels(clustering)))
# 
# # Graficar y guardar
# png("pca_graph_filtered.png", width = 1200, height = 1000, res = 150)
# 
# par(mar = c(4, 4, 4, 4))
# plot(new_PCA_embed, type = "n", xlab = "PC1", ylab = "PC2", main = "Filtered PCA with Graph Overlay")
# #plot(KNN, layout = new_PCA_embed, add = TRUE, edge.color = "grey80", edge.width = 0.5)
# 
# points(
#   new_PCA_embed,
#   cex = ifelse(!to_keep, 1, 0.3),
#   lwd = ifelse(!to_keep, 2, 0),
#   bg = pal[clustering],
#   pch = 21
# )
# 
# text(new_centroids2d, labels = rownames(new_centroids2d), cex = 1, font = 2)
# dev.off()

##################################
## TRAJECTORY INFERENCE WITH SLINGSHOT
#######################################
# library(slingshot)
# 
# # Validar que clustering y embedding tengan nombres consistentes
# stopifnot(identical(rownames(new_PCA_embed), names(clustering)))
# 
# library(distances)
# 
# # Matriz de centroides por cluster usando nuevo embedding PCA
# cluster_ids <- levels(clustering)
# centroids <- sapply(cluster_ids, function(cl) {
#   cells <- which(clustering == cl)
#   colMeans(new_PCA_embed[cells, , drop = FALSE])
# }, simplify = "array")
# 
# # Transponer para que filas sean clusters
# centroids <- t(centroids)
# rownames(centroids) <- cluster_ids
# 
# # Revisar NaNs en matriz de distancias
# dist_mat <- as.matrix(dist(centroids))
# which(is.na(dist_mat), arr.ind = TRUE)
# 
# # Filtrar clusters con al menos 5 células
# valid_clusters <- names(which(table(clustering) >= 5))
# valid_cells <- clustering %in% valid_clusters
# 
# # Filtrar datos
# new_PCA_valid <- new_PCA_embed[valid_cells, ]
# clustering_valid <- droplevels(clustering[valid_cells])
# 
# # Ejecutar Slingshot
# set.seed(1)
# lineages <- getLineages(
#   data = new_PCA_valid,
#   clusterLabels = clustering_valid
# )
# 
# lineages <- as.SlingshotDataSet(lineages)
# 
# # Visualizar resultados y guardar
# png("slingshot_trajectory_pca.png", width = 1200, height = 1000, res = 150)
# par(mfrow = c(1, 2), mar = c(4, 4, 4, 4))
# 
# # Plot A: línea sobre PCA filtrado
# plot(new_PCA_valid, col = pal[clustering_valid], cex = 0.5, pch = 16,
#      main = "Lineages on filtered PCA")
# lines(lineages, lwd = 2, col = "black", cex = 3)
# text(new_centroids2d, labels = rownames(new_centroids2d),
#      cex = 0.8, font = 2, col = "white")
# 
# # Plot B: grafo con outliers marcados
# KNN_valid <- KNN[valid_cells, valid_cells]
# plot(new_PCA_valid, type = "n", main = "PCA + KNN Graph + Outliers")
# draw_graph(layout = new_PCA_valid, graph = KNN_valid)
# 
# points(new_PCA_valid,
#        cex = ifelse(!to_keep, 1, 0.3),
#        lwd = ifelse(!to_keep, 2, 0),
#        bg = pal[clustering_valid], pch = 21)
# text(new_centroids2d, labels = rownames(new_centroids2d),
#      cex = 0.8, font = 2)
# 
# dev.off()
# 
# print(lineages)
# 
# # Obtener expresión media por gen en el cluster 20
# cluster_id <- "20"
# 
# # Extraer células del cluster 20
# cells_cluster_20 <- colnames(seurat_obj)[seurat_obj$leiden == cluster_id]
# 
# # Obtener la matriz de expresión normalizada
# expr_matrix <- GetAssayData(seurat_obj, assay = "RNA", slot = "data")
# 
# # Calcular promedio por gen dentro del cluster
# mean_expr_cluster20 <- Matrix::rowMeans(expr_matrix[, cells_cluster_20, drop = FALSE])
# 
# # Ordenar y mostrar los genes más expresados
# top_genes <- sort(mean_expr_cluster20, decreasing = TRUE)[1:20]
# print(top_genes)
# 
# # Calcular expresión media por cluster (usando 'leiden')
# library(dplyr)
# meta <- seurat_obj@meta.data
# meta$FTL <- Seurat::FetchData(seurat_obj, vars = "FTL")[, 1]
# 
# cluster_means <- meta %>%
#   group_by(leiden) %>%
#   summarise(mean_expr = mean(FTL)) %>%
#   arrange(desc(mean_expr))
# 
# print(cluster_means)
# 
# # Trajectoria desde el cluster FTL-alto
# set.seed(1)
# lineages <- as.SlingshotDataSet(
#   getLineages(
#     data = new_PCA_valid,
#     clusterLabels = clustering_valid,
#     start.clus = "20"
#   )
# )
# 
# # Asignar PCA corregido como embedding
# lineages@reducedDim <- new_PCA_valid
# 
# # Graficar trayectorias
# png("slingshot_trajectory-FTL-marker_PCA.png", width = 1200, height = 1000, res = 150)
# par(mar = c(4, 4, 4, 4))
# plot(new_PCA_valid, col = pal[clustering_valid], cex = 0.5, pch = 16,
#      main = "Slingshot trajectories from FTL-high cluster (PCA)")
# lines(lineages, lwd = 2, col = 'black')
# text(new_centroids2d, labels = rownames(new_centroids2d), cex = 0.8, font = 2, col = "white")
# dev.off()
# 
# # Paleta
# pal <- rainbow(length(levels(clustering)))
# 
# # Graficar el PCA completo con etiquetas
# png("full_PCA_with_graph.png", width = 1200, height = 1000, res = 150)
# par(mar = c(4, 4, 4, 4))
# plot(new_PCA_embed, col = pal[clustering], cex = 0.5, pch = 16,
#      main = "PCA con todos los clusters")
# draw_graph(layout = new_PCA_embed, graph = KNN)
# text(new_centroids2d, labels = rownames(new_centroids2d), cex = 1, font = 2)
# dev.off()
# 
# # Clusters a eliminar
# clusters_to_remove <- c("40", "8")
# 
# # Filtrar celdas que no pertenecen a esos clusters
# cell_to_keep <- !(clustering %in% clusters_to_remove)
# 
# # Filtrar objetos
# filt_new_PCA         <- new_PCA_embed[cell_to_keep, ]
# filt_NORM_COUNTS     <- NORM_COUNTS[, cell_to_keep]
# filt_PCA             <- PCA[cell_to_keep, ]
# filt_HARMONY         <- HARMONY[cell_to_keep, ]
# filt_KNN             <- KNN[cell_to_keep, cell_to_keep]
# filt_clustering      <- factor(clustering[cell_to_keep])
# 
# # Recalcular centroides 2D con matriz modelo (usando PCA filtrado)
# filt_mm <- sparse.model.matrix(~ 0 + filt_clustering)
# colnames(filt_mm) <- levels(filt_clustering)
# filt_new_centroids2d <- as.matrix(t(t(filt_new_PCA) %*% filt_mm) / Matrix::colSums(filt_mm))
# 
# # Paleta para clusters filtrados
# pal_filt <- rainbow(length(levels(filt_clustering)))
# 
# # Graficar y guardar
# png("filtered_PCA_with_graph.png", width = 1200, height = 1000, res = 150)
# par(mar = c(4, 4, 4, 4))
# plot(filt_new_PCA, type = "n", main = "PCA filtrado")
# draw_graph(layout = filt_new_PCA, graph = filt_KNN)
# points(filt_new_PCA, col = pal_filt[filt_clustering], cex = 0.5, pch = 16)
# text(filt_new_centroids2d, labels = rownames(filt_new_centroids2d), cex = 1, font = 1)
# dev.off()
# 
# 
# library(pheatmap)
# library(Matrix)
# library(RColorBrewer)
# # Paleta para clusters filtrados
# pal <- colorRampPalette(brewer.pal(12, "Set3"))(length(levels(filt_clustering)))
# 
# # Crear matriz modelo para clusters filtrados
# filt_mm <- sparse.model.matrix(~ 0 + filt_clustering)
# colnames(filt_mm) <- levels(filt_clustering)
# 
# # Calcular conexiones entre clusters basado en la matriz KNN
# d <- t(filt_KNN %*% filt_mm) %*% filt_mm /
#   (t(t(colSums(filt_mm))) %*% t(colSums(filt_mm)))^(1/2)
# 
# diag(d) <- 0
# d <- drop0(d)  # eliminar ceros explícitos (sparse matrix)
# 
# # Visualizar conexiones entre clusters
# pheatmap(d, clustering_method = "ward.D2", main = "Conexiones entre clusters (PCA)")
# 
# # Histograma de valores de conexión
# hist(d@x, breaks = 50, main = "Histograma de valores de conexión")
# cutoff <- 1.2 * (sum((d@x^2)) / (length(d@x) - 1))^(1/2)
# abline(v = cutoff, col = "red", xpd = FALSE)
# 
# # Identificar pares de clusters con alta conexión
# to_merge <- drop0((d > cutoff) * 1)
# to_merge <- to_merge * lower.tri(to_merge)
# diag(to_merge)[rowSums(to_merge) == 0] <- 1
# 
# # Visualizar mapa de merges
# pheatmap(to_merge, cluster_rows = FALSE, cluster_cols = FALSE, main = "Mapeo de merges")
# 
# # Generar tabla de mapeo para hacer merge
# mappings <- cbind(
#   from = colnames(to_merge)[to_merge@i + 1],
#   to   = colnames(to_merge)[rep(1:(length(to_merge@p) - 1), times = diff(to_merge@p))]
# )
# 
# # Aplicar merge al clustering original
# merged_filt_clustering <- factor(mappings[match(filt_clustering, mappings[, "from"]), "to"])
# 
# # Nueva matriz modelo con clusters mergeados
# merged_filt_mm <- sparse.model.matrix(~ 0 + merged_filt_clustering)
# colnames(merged_filt_mm) <- levels(merged_filt_clustering)
# 
# # Calcular centroides usando el embedding filtrado en PCA
# merged_filt_new_centroids2d <- as.matrix(t(t(filt_new_PCA) %*% merged_filt_mm) / Matrix::colSums(merged_filt_mm))
# 
# # Graficar resultado
# png("merged_clusters_pca.png", width = 1200, height = 1000, res = 150)
# par(mar = c(4, 4, 4, 4))
# plot(filt_new_PCA, type = "n", main = "Clusters mergeados (PCA)")
# draw_graph(layout = filt_new_PCA, graph = filt_KNN)
# points(filt_new_PCA, col = pal[merged_filt_clustering], cex = 0.5, pch = 16)
# text(merged_filt_new_centroids2d, labels = rownames(merged_filt_new_centroids2d), cex = 1, font = 1)
# dev.off()



