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

res <- seq(0.4,1.5,0.1)
for (reso in res){
  libd_mousebrain <- FindClusters(libd_mousebrain, verbose = FALSE,resolution=reso,random.seed=1)
  if (length(unique(libd_mousebrain@meta.data$seurat_clusters)) == n_cluster){
    cat("Final set resolution is: ", reso)
    break}
  else {
    next  
  }}
if (length(unique(libd_mousebrain@meta.data$seurat_clusters)) != n_cluster){
  reso =  0.3
  cat("manually set resolution is: ", reso)
}
save_obj <- data.frame(ID = colnames(libd_mousebrain))
for (i in seq(1,10)){
  set.seed(i)
  libd_mousebrain <- FindClusters(libd_mousebrain, verbose = FALSE,resolution=reso,random.seed=i)
  seurat_output <- data.frame(cluster = libd_mousebrain@meta.data$seurat_clusters)
  save_obj <- cbind(save_obj,  cluster= seurat_output)
}
write.csv(save_obj, file = file.path(save_dir, "STARmap_Seurat_ClusterOutput.csv"),
          row.names = F)



