rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)
library(reticulate)
library(SeuratDisk)
library(dplyr)
library(Giotto)

# 设置数据路径
BASE_DIR <- "D:/Work/MCGAE project/MCGAE-master"
file_path <- file.path(BASE_DIR, "benchmark", "STARmap")
dir_path <- file.path(file_path, "raw_data")
csv_file_path <- file.path(file_path, "cluster_metric", "STARmapSum.csv")
save_dir <- file.path(file_path, "cluster_metric")
# 将 h5ad 文件转换为 h5Seurat 格式
# Convert(file.path(dir_path, "STARmap_20180505_BY3_1k.h5ad"), dest = "h5seurat", overwrite = TRUE)
# 读取 h5Seurat 文件
adata <- LoadH5Seurat(file.path(dir_path, "STARmap_20180505_BY3_1k.h5Seurat"))

label <- adata@meta.data$label
libd_mousebrain <- adata
# pre-filter for generate simulated data quickly
libd_mousebrain <- SCTransform(libd_mousebrain, assay = "RNA")
libd_mousebrain <- RunPCA(libd_mousebrain, assay = "SCT", verbose = FALSE)
libd_mousebrain <- FindNeighbors(libd_mousebrain, reduction = "pca", dims = 1:30)
n_cluster <- length(unique(label))
save_obj <- data.frame(ID = colnames(libd_mousebrain))
counts <- libd_mousebrain@assays$RNA@counts
numCluster = n_cluster
myinst  <- createGiottoInstructions(save_plot=F, show_plot=T, save_dir=save_dir,
                                    python_path="D:\\Software\\Anaconda\\envs\\py3.9")
# coldata <- data.frame(row = libd_mousebrain@images$slice1@coordinates$col, col = libd_mousebrain@images$slice1@coordinates$row)
coldata <- data.frame(row = libd_mousebrain$X, col = libd_mousebrain$Y)

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
for (i in seq(1,10)){
  go_obj <- runPCA(gobject = go_obj, scale_unit = F, center=T, method="factominer")
  go_obj <- createNearestNetwork(gobject = go_obj, type = "sNN", dimensions_to_use = 1:15, k = 15)
  ## k-means clustering
  go_obj <- doKmeans(gobject = go_obj, dim_reduction_to_use = 'pca', centers = numCluster, seed_number=i)
  comp_df <- data.frame(cluster = go_obj@cell_metadata$kmeans)
  save_obj <- cbind(save_obj, comp_df)
}

write.csv(save_obj, file = file.path(save_dir, "STARmap_Giotto_ClusterOutput.csv"), 
          row.names = F)
