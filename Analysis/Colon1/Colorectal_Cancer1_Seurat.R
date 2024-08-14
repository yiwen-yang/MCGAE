rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)

file_name <- "ST-colon1"
BASE_DIR <- "D:/Work/MCGAE project/MCGAE-master"
rawdata_path <- file.path(BASE_DIR, "benchmark", "Colorectal Cancer", file_name)
print(rawdata_path)  

save_dir <- file.path(rawdata_path, "result_new")
libd_mousebrain <- Load10X_Spatial(data.dir = paste0(rawdata_path, "\\raw_data"),
                                   filename = "filtered_feature_bc_matrix.h5",
                                   assay = "Spatial",
                                   filter.matrix = T)
# label <- read.table(file = paste0(rawdata_path, "\\raw_data\\", "metadata.tsv"),sep="\t", header=TRUE)

# pre-filter for generate simulated data quickly
libd_mousebrain <- SCTransform(libd_mousebrain, assay = "Spatial")
libd_mousebrain <- RunPCA(libd_mousebrain, assay = "SCT", verbose = FALSE)
libd_mousebrain <- FindNeighbors(libd_mousebrain, reduction = "pca", dims = 1:30)
n_cluster <- 20

save_obj <- data.frame(ID = colnames(libd_mousebrain))
set.seed(1)
libd_mousebrain <- FindClusters(libd_mousebrain, verbose = FALSE,resolution=1.745,random.seed=1)
cat("Final cluster is: ", length(unique(libd_mousebrain@meta.data$seurat_clusters)))
seurat_output <- data.frame(cluster = libd_mousebrain@meta.data$seurat_clusters)
save_obj <- cbind(save_obj,  cluster= seurat_output)

write.csv(save_obj, file = paste0(save_dir, "//Seurat_ClusterOutput.csv"),
          row.names = F)



