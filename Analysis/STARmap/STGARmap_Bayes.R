rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)
library(reticulate)
library(SeuratDisk)
library(dplyr)
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
# coldata <- data.frame(row = libd_mousebrain@images$slice1@coordinates$col, col = libd_mousebrain@images$slice1@coordinates$row)
coldata <- data.frame(row = libd_mousebrain$X, col = libd_mousebrain$Y)
sce <- SingleCellExperiment(assays=list(counts=as(counts, "dgCMatrix")),
                              colData=coldata[,1:2])
bs_obj <- sce
for (i in seq(1,10)){
    set.seed(i)
    bs_obj <- spatialPreprocess(bs_obj, platform="Visium", n.HVGs=2000, n.PCs=30, log.normalize = T)
    set.seed(i)
    bs_obj <- spatialCluster(bs_obj, q=numCluster, platform="Visium",
                             init.method="mclust", model="t",
                             nrep=10000, burn.in=1000,
                             save.chain=TRUE)
    comp_df <- as.data.frame(colData(bs_obj))
    save_obj <- cbind(save_obj,  cluster= comp_df[, 5])
}

write.csv(save_obj, file = file.path(save_dir, "STARmap_Bayes_ClusterOutput.csv"),
          row.names = F)

