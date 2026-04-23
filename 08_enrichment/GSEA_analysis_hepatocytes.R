setwd("/home/mdiaz/HCC_project/MERGED_analysis/enrichment_analysis")


# Setting up environment ===================================================

# Clean environment
rm(list = ls(all.names = TRUE)) # will clear all objects including hidden objects
gc() # free up memory and report the memory usage
options(max.print = .Machine$integer.max, scipen = 999, stringsAsFactors = F, dplyr.summarise.inform = F) # avoid truncated output in R console and scientific notation


# Loading relevant libraries 
library(tidyverse) # includes ggplot2, for data visualisation. dplyr, for data manipulation.
library(RColorBrewer) # for a colourful plot
library(pheatmap)
library(clusterProfiler) # for PEA analysis
library('org.Hs.eg.db')
library(DOSE)
library(enrichplot) # for visualisations
library(ggupset) # for visualisations

# Set input path
in_path <- "/home/mdiaz/HCC_project/MERGED_analysis/DGE_csv_files" # input path, where your data is located
out_path <- "/home/mdiaz/HCC_project/MERGED_analysis/GSEA_results/hepatocytes" # output path, where you want your results exported to
bg_path <- "/datos/home/mdiaz/sc_liver_data/enrichment_pathways_in_R/geneSets_pathways"
# Functions ------------------------------------------------
## Function: Adjacency matrix to list ####
matrix_to_list <- function(pws){
  pws.l <- list()
  for (pw in colnames(pws)) {
    pws.l[[pw]] <- rownames(pws)[as.logical(pws[, pw])]
  }
  return(pws.l)
}

# Read in data ===================================================
list.files(in_path)
df <- read.csv(paste0(in_path, '/DGE_hepatocytes.csv'))

df <- df[!duplicated(df$gene_symbol), ]

df <- df[!grepl("\\.", df$gene_symbol), ]

# Verificar si hay genes duplicados
if (any(duplicated(df$gene_symbol))) {
  warning("Hay genes duplicados en la columna 'gene_symbol'. Se eliminarán duplicados.")
  df <- df[!duplicated(df$gene_symbol), ] # Eliminar duplicados
}

head(df)

# Annotate according to differential expression
df <- df %>% mutate(diffexpressed = case_when(
  log2fc > 0 & padj < 0.05 ~ 'UP',
  log2fc < 0 & padj < 0.05 ~ 'DOWN',
  padj > 0.05 ~ 'NO'
))

genes_in_data <- df$gene_symbol


# SQUIDTIP! If you want to parse several .gmt files at once, you can use a loop:
output_path <- "/home/mdiaz/HCC_project/MERGED_analysis/enrichment_analysis/geneSets_pathways"
gmt_files <- list.files(path = bg_path, pattern = '.gmt', full.names = TRUE)
for (file in gmt_files){
  file <- gmt_files[1]
  pwl2 <- read.gmt(file)
  pwl2 <- pwl2[pwl2$gene %in% genes_in_data,]
  filename <- file.path(output_path, paste0(gsub('c.\\.', '', gsub('.v2024.*$', '', basename(file))), '.RDS'))
  saveRDS(pwl2, filename)
}

# Remove non-significant genes
df <- df[df$diffexpressed != 'NO', ]
# Substitute names so they are annotated nicely in the heatmap later
df$diffexpressed <- gsub('DOWN', 'Down Regulated', gsub('UP', 'Up Regulated', df$diffexpressed))
unique(df$diffexpressed)
# Split the dataframe into a list of sub-dataframes: upregulated, downregulated genes
deg_results_list <- split(df, df$diffexpressed)


## Run ClusterProfiler -----------------------------------------------

# Settings
name_of_comparison <- 'UPvsDOWN' # for our filename
background_genes <- 'GOBP' # for our filename
bg_genes <- readRDS(paste0(bg_path, '/go.bp.RDS')) # read in the background genes
padj_cutoff <- 0.05 # p-adjusted threshold, used to filter out pathways
genecount_cutoff <- 5 # minimum number of genes in the pathway, used to filter out pathways
filename <- paste0(out_path, 'clusterProfiler/', name_of_comparison, '_', background_genes) # filename of our PEA results

upregulated_genes <- deg_results_list$`Up Regulated`$gene_symbol

x <- enricher(upregulated_genes, TERM2GENE = bg_genes)

# Ejecutamos el análisis para cada sub-dataframe
res <- lapply(names(deg_results_list), function(pattern) {
  # Extraemos los genes para cada patrón
  genes <- deg_results_list[[pattern]]$gene_symbol
  # Ejecutamos el análisis de enriquecimiento
  enricher(genes, TERM2GENE = bg_genes)
})

# Asignamos nombres a los resultados (Up vs Down)
names(res) <- names(deg_results_list)

# Convertimos los resultados de enriquecimiento a un dataframe
res_df <- lapply(names(res), function(name) {
  # Combinamos los resultados de cada patrón (Up Regulated o Down Regulated)
  rbind(res[[name]]@result)
})

# Asignamos nombres a las listas resultantes
names(res_df) <- names(res)

# Unimos todos los resultados en un solo dataframe
res_df <- do.call(rbind, res_df)

# Mostramos las primeras filas del dataframe resultante
head(res_df)

# Agregar columnas adicionales para hacer los resultados más legibles
res_df <- res_df %>% mutate(
  minuslog10padj = -log10(p.adjust),  # log transform de p ajustado
  diffexpressed = gsub('\\.GOBP.*$|\\.KEGG.*$|\\.REACTOME.*$', '', rownames(res_df))  # Simplificar los nombres de las rutas
)

# Filtrar por umbral de p-valor y número de genes
target_pws <- unique(res_df$ID[res_df$p.adjust < padj_cutoff & res_df$Count > genecount_cutoff])
res_df <- res_df[res_df$ID %in% target_pws, ]

# Verifica si el directorio existe, si no, lo crea
dir.create(dirname(filename), showWarnings = FALSE, recursive = TRUE)

# Verifica si el dataframe res_df no está vacío antes de guardarlo
if (nrow(res_df) > 0) {
  print('Saving clusterprofiler results')
  write.csv(res_df, paste0(filename, '_resclusterp.csv'), row.names = FALSE)
} else {
  print('No significant pathways found to save.')
}

########################################
##  Visualization  #####################
########################################

# Read in the data
#res_df <- read.csv(paste0(out_path, 'clusterProfiler/', 'UPvsDOWN_GOBP_resclusterp.csv'))
#bg_genes <- readRDS(paste0(bg_path, '/go.bp.RDS'))

# Convert clusterProfiler object to a new "enrichResult" object
# Select only upregulated genes

res_df_up <- res_df %>% 
  filter(diffexpressed == 'Up Regulated') %>% 
  dplyr::select(!c('minuslog10padj', 'diffexpressed')) 

rownames(res_df_up) <- res_df_up$ID

enrichres_up <- new("enrichResult",
                    readable = FALSE,
                    result = res_df_up,
                    pvalueCutoff = 0.05,
                    pAdjustMethod = "BH",
                    qvalueCutoff = 0.2,
                    organism = "human",
                    ontology = "UNKNOWN",
                    gene = df$gene_symbol,
                    keytype = "UNKNOWN",
                    universe = unique(bg_genes$gene),
                    gene2Symbol = character(0),
                    geneSets = bg_genes)

enrichres_up_sim <- pairwise_termsim(enrichres_up)

class(enrichres_up_sim)

#Barplot
barplot(enrichres_up, showCategory = 20) 

#Dotplot
dotplot(enrichres_up, showCategory = 10)


# Cnetplot
cnetplot(enrichres_up)
#This way we easily identify clusters of genes 
#that may share common biological functions 
# or participate in the same molecular pathway


# Upsetplot
upsetplot(enrichres_up)
#An upsetplot is an alternative to cnetplot. It emphasizes the gene overlapping among different gene sets.
#In an upsetplot, each gene set or pathway is represented by a column, and the height of the column indicates 
#the number of genes associated with that set or term. The intersections between the gene sets or terms are shown 
#as connected lines or bars.


# Heatplot
heatplot(enrichres_up, showCategory = 7)

#A heatplot is similar to cnetplot. In this case, the x axis shows the list of genes 
#we have (in this case, our upregulated genes). In the y axis, we have our pathways. 
#The bars show which genes participated in which pathways. It is useful when there is
#a large number of significant pathways, as it makes it easier to visualize.

# Treeplot
enrichres_up_sim <- pairwise_termsim(enrichres_up) # calculate pairwise similarities of the enriched terms using Jaccard’s similarity index
#treeplot(enrichres_up_sim)
#A treeplot function as a hierarchical clustering of enriched pathways
#The treeplot() function from the enrichplot package takes the enrichment 
#results and the hierarchical ontology as inputs. It creates a tree-like plot 
#where each pathway is represented by a node, and the parent-child relationships 
#between terms are shown by connecting lines.
#In this case, the node colour represents the significance of enrichment. 
#The node size number of genes associated with each term. 


# Enrichment map 
emapplot(enrichres_up_sim)
#An enrichment map organizes enriched pathways into a network with edges connecting overlapping gene sets or pathways. 
#In this way, mutually overlapping gene sets are tend to cluster together, making it easy to identify functional module.




################
#Down regulated genes 

res_df_down <- res_df %>% 
  filter(diffexpressed == 'Down Regulated') %>% 
  dplyr::select(!c('minuslog10padj', 'diffexpressed')) 

rownames(res_df_down) <- res_df_down$ID

enrichres_down <- new("enrichResult",
                      readable = FALSE,
                      result = res_df_down,
                      pvalueCutoff = 0.05,
                      pAdjustMethod = "BH",
                      qvalueCutoff = 0.2,
                      organism = "human",
                      ontology = "UNKNOWN",
                      gene = df$gene_symbol,
                      keytype = "UNKNOWN",
                      universe = unique(bg_genes$gene),
                      gene2Symbol = character(0),
                      geneSets = bg_genes)

class(enrichres_down)

#Barplot
barplot(enrichres_down, showCategory = 20) 


# Cnetplot
cnetplot(enrichres_down)
#This way we easily identify clusters of genes 
#that may share common biological functions 
# or participate in the same molecular pathway


# Upsetplot
upsetplot(enrichres_down)
#An upsetplot is an alternative to cnetplot. It emphasizes the gene overlapping among different gene sets.
#In an upsetplot, each gene set or pathway is represented by a column, and the height of the column indicates 
#the number of genes associated with that set or term. The intersections between the gene sets or terms are shown 
#as connected lines or bars.


# Heatplot
heatplot(enrichres_down, showCategory = 7)

#A heatplot is similar to cnetplot. In this case, the x axis shows the list of genes 
#we have (in this case, our upregulated genes). In the y axis, we have our pathways. 
#The bars show which genes participated in which pathways. It is useful when there is
#a large number of significant pathways, as it makes it easier to visualize.

# Treeplot
enrichres_down_sim <- pairwise_termsim(enrichres_down) # calculate pairwise similarities of the enriched terms using Jaccard’s similarity index
treeplot(enrichres_down_sim)
#A treeplot function as a hierarchical clustering of enriched pathways
#The treeplot() function from the enrichplot package takes the enrichment 
#results and the hierarchical ontology as inputs. It creates a tree-like plot 
#where each pathway is represented by a node, and the parent-child relationships 
#between terms are shown by connecting lines.
#In this case, the node colour represents the significance of enrichment. 
#The node size number of genes associated with each term. 


# Enrichment map 
#emapplot(enrichres_down_sim)
#An enrichment map organizes enriched pathways into a network with edges connecting overlapping gene sets or pathways. 
#In this way, mutually overlapping gene sets are tend to cluster together, making it easy to identify functional module.

# 
# emapplot(enrichres2, showCategory = 15) + ggrepel::geom_text_repel(max.overlaps = 50)
# options(repr.plot.width = 10, repr.plot.height = 8)  # Ajusta según necesidad
# emapplot(enrichres2, showCategory = 30)




#######
# Up vs Down regulated genes

# Combinar genes UP y DOWN regulados
all_genes <- c(deg_results_list[["Up Regulated"]]$gene_symbol, 
               deg_results_list[["Down Regulated"]]$gene_symbol)


# Extraer la información de regulación de los nombres de las filas
res_df <- res_df %>%
  rownames_to_column(var = "term") %>%  # Convertir rownames a una columna
  separate(term, into = c(".sign", "term"), sep = "\\.", extra = "merge")  # Separar en .sign y término

# Simplificar los valores de .sign
res_df <- res_df %>%
  mutate(.sign = case_when(
    .sign == "Up Regulated" ~ "UP",
    .sign == "Down Regulated" ~ "DOWN"
  ))

res_df <- res_df %>%
  mutate(term = str_replace(term, "^GO.*_", ""))

# Recrear el objeto enrichResult con los nombres de pathways actualizados
enrichres <- new("enrichResult",
                 readable = FALSE,
                 result = res_df,
                 pvalueCutoff = 0.05,
                 pAdjustMethod = "BH",
                 qvalueCutoff = 0.2,
                 organism = "human",
                 ontology = "BP",  # Adaptado para GO:BP
                 gene = all_genes,  # Usar todos los genes (UP y DOWN)
                 keytype = "UNKNOWN",
                 universe = unique(bg_genes$gene),
                 gene2Symbol = character(0),
                 geneSets = bg_genes)

# Generar el dotplot comparativo
dotplot_BP <- dotplot(enrichres, showCategory = 10, split = ".sign") + 
  facet_grid(. ~ .sign) + 
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))

# Mostrar el dotplot
print(dotplot_BP)


# Barplot
# Crear el barplot comparativo
barplot_comparativo <- barplot(enrichres, showCategory = 10, split = ".sign") + 
  facet_grid(. ~ .sign, scales = "free") +  # Dividir en paneles UP y DOWN
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm")) + 
  labs(x = "qscore (-log10 p.adjust)", y = "Pathway")  # Etiquetas de los ejes

# Mostrar el barplot
print(barplot_comparativo)


# In this barplots red pathways are motre significantly enriched than blue pathways
# The X-axis refers to number of (up-regulated) genes associated to that pathway

#################
#SAVING PLOTS####
#################
output_dir <- "/home/mdiaz/HCC_project/MERGED_analysis/GSEA_results/hepatocytes"
save_plot <- function(plot_expr, filename, width=10, height=8, path="/home/mdiaz/HCC_project/MERGED_analysis/GSEA_results/hepatocytes") {
  png(file.path(path, filename), width=width, height=height, units="in", res=300)
  print(plot_expr)
  dev.off()
}
#UP REGULATED GENES 
save_plot(barplot(enrichres_up, showCategory = 17), "barplot_up.png")
save_plot(dotplot(enrichres_up, showCategory = 17), "dotplot_up.png")
save_plot(cnetplot(enrichres_up), "cnetplot_up.png")
save_plot(upsetplot(enrichres_up), "upsetplot_up.png")
save_plot(heatplot(enrichres_up, showCategory = 7), "heatplot_up.png")
save_plot(treeplot(enrichres_up_sim), "treeplot_up.png")
save_plot(emapplot(enrichres_up_sim), "emapplot_up.png")
save_plot(emapplot(enrichres_up_sim, showCategory = 10), "emapplot_top30_up.png")

#DOWN REGULATED GENES 
save_plot(barplot(enrichres_down, showCategory = 15), "barplot_down.png")
save_plot(cnetplot(enrichres_down), "cnetplot_down.png")
save_plot(upsetplot(enrichres_down), "upsetplot_down.png")
save_plot(heatplot(enrichres_down, showCategory = 7), "heatplot_down.png")
save_plot(treeplot(enrichres_down_sim), "treeplot_down.png")
save_plot(emapplot(enrichres_down_sim), "emapplot_down.png")
save_plot(emapplot(enrichres_down_sim, showCategory = 10), "emapplot_top30_down.png")

############################
## SAVING COMPARATIVE PLOTS#
############################
# Dotplot con ajustes
dotplot_BP_clean <- dotplot_BP +
  theme(
    strip.text = element_text(size = 12),
    axis.text.y = element_text(size = 9),
    axis.text.x = element_text(size = 9),
    panel.spacing = unit(2, "lines"),  # más espacio entre paneles
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave(
  filename = file.path(output_dir, "dotplot_up_vs_down.png"),
  plot = dotplot_BP_clean,
  width = 14, height = 10, dpi = 300
)

# Barplot con ajustes
barplot_comparativo_clean <- barplot_comparativo +
  theme(
    strip.text = element_text(size = 12),
    axis.text.y = element_text(size = 9),
    axis.text.x = element_text(size = 9),
    panel.spacing = unit(2, "lines"),
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave(
  filename = file.path(output_dir, "barplot_up_vs_down.png"),
  plot = barplot_comparativo_clean,
  width = 14, height = 10, dpi = 300
)

###################
# TREE PLOTS#######
###################
treeplot_up_clean <- treeplot(enrichres_up_sim, label_format = 25) + 
  theme(
    plot.margin = margin(30, 30, 30, 30),
    text = element_text(size = 4),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 4)
  )

treeplot_down_clean <- treeplot(enrichres_down_sim, label_format = 25) + 
  theme(
    plot.margin = margin(30, 30, 30, 0.1),
    text = element_text(size = 1),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 1)
  )

ggsave(
  filename = file.path(output_dir, "treeplot_up.png"),
  plot = treeplot_up_clean,
  width = 16, height = 14, dpi = 300
)

ggsave(
  filename = file.path(output_dir, "treeplot_down.png"),
  plot = treeplot_down_clean,
  width = 16, height = 14, dpi = 300
)


