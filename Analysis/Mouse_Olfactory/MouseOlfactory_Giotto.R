rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)
library(Giotto)
library(dplyr)


BASE_DIR <- "D:/Work/MCGAE project/MCGAE-master"
file_path <- file.path(BASE_DIR, "benchmark", "Mouse_Olfactory", "raw_data")
save_dir <- file.path(BASE_DIR, "benchmark", "Mouse_Olfactory", "result_new")

counts <- read.csv(file.path(file_path, "RNA_counts.tsv"), sep = "\t", row.names = 1)
position <- read.csv(file.path(file_path, "position.tsv"), sep = "\t")
used_barcodes <- read.csv(file.path(file_path, "used_barcodes.txt"), sep = "\t", header = FALSE, stringsAsFactors = FALSE)
colnames(counts) <- gsub("^X", "Spot_", colnames(counts))
rownames(position) <- paste0('Spot_', position$label)
position <- position %>% select(x, y)
print("Column names in counts:")
print(colnames(counts))
print("Used barcodes:")
print(used_barcodes$V1)
missing_barcodes <- setdiff(used_barcodes$V1, colnames(counts))
if (length(missing_barcodes) > 0) {
  print("Missing barcodes:")
  print(missing_barcodes)
} else {
  print("All barcodes are matched.")
}
if (all(used_barcodes$V1 %in% colnames(counts))) {
  counts <- counts[, used_barcodes$V1]
} else {
  stop("Some barcodes in used_barcodes are not found in counts columns.")
}
counts_filtered <- counts[, used_barcodes$V1, drop = FALSE]

position_filtered <- position[used_barcodes$V1, ]
adata <- CreateSeuratObject(counts = counts_filtered, assay = "Spatial")
adata <- AddMetaData(adata, metadata = position_filtered, col.name = c("x", "y"))
libd_mousebrain <- adata

# pre-filter for generate simulated data quickly
libd_mousebrain <- SCTransform(libd_mousebrain, assay = "Spatial")
libd_mousebrain <- RunPCA(libd_mousebrain, assay = "SCT", verbose = FALSE)
libd_mousebrain <- FindNeighbors(libd_mousebrain, reduction = "pca", dims = 1:30)
n_cluster <- 7
save_obj <- data.frame(ID = colnames(libd_mousebrain))
counts <- libd_mousebrain@assays$Spatial$counts
numCluster = n_cluster
myinst  <- createGiottoInstructions(save_plot=F, show_plot=T, save_dir=save_dir,
                                    python_path="D:\\Software\\Anaconda\\envs\\py3.9")
coldata <- data.frame(row = position_filtered$x, col = position_filtered$y)
obj         <- createGiottoObject(raw_exprs = counts, spatial_locs = coldata[,1:2],instructions = myinst)
go_obj      <- filterGiotto(gobject = obj, expression_threshold = 0,gene_det_in_min_cells = 50, min_det_genes_per_cell = 2,expression_values = c('raw'),verbose = T)
go_obj      <- normalizeGiotto(gobject = go_obj, scalefactor = 6000, verbose = T)
## add gene & cell statistics
go_obj      <- addStatistics(gobject = go_obj)
## adjust expression matrix for technical or known variables
go_obj      <- adjustGiottoMatrix(gobject = go_obj, expression_values = c('normalized'),batch_columns = NULL, covariate_columns = c('nr_genes', 'total_expr'),return_gobject = TRUE,update_slot = c('custom'))
go_obj      <- calculateHVG(gobject = go_obj, method = 'cov_loess',show_plot=FALSE,save_plot=FALSE, difference_in_cov = 0.1)
## select genes based on HVG and gene statistics, both found in gene metadata
gene_metadata <- fDataDT(go_obj)
featgenes <- gene_metadata[hvg == 'yes' & perc_cells > 2 & mean_expr_det > 0.5]$gene_ID
save_obj <- data.frame(ID = colnames(libd_mousebrain))
for (i in seq(1,1)){
  go_obj <- runPCA(gobject = go_obj, scale_unit = F, center=T, method="factominer")
  go_obj <- createNearestNetwork(gobject = go_obj, type = "sNN", dimensions_to_use = 1:15, k = 15)
  ## k-means clustering
  go_obj <- doKmeans(gobject = go_obj, dim_reduction_to_use = 'pca', centers = numCluster, seed_number=i)
  comp_df <- data.frame(cluster = go_obj@cell_metadata$kmeans)
  save_obj <- cbind(save_obj, comp_df)
}

write.csv(save_obj, file = paste0(save_dir, "//Giotto_ClusterOutput.csv"),
          row.names = F)
