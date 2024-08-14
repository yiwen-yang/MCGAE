rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)

file_name <- "ST-colon1"
BASE_DIR <- "D:/Work/MCGAE project/MCGAE-master"
rawdata_path <- file.path(BASE_DIR, "benchmark", "Colorectal Cancer", file_name)


save_dir <- file.path(rawdata_path, "result_new")
libd_mousebrain <- Load10X_Spatial(data.dir = paste0(rawdata_path, "\\raw_data"),
                               filename = "filtered_feature_bc_matrix.h5",
                               assay = "Spatial",
                               filter.matrix = T)

# pre-filter for generate simulated data quickly
libd_mousebrain <- SCTransform(libd_mousebrain, assay = "Spatial")
libd_mousebrain <- RunPCA(libd_mousebrain, assay = "SCT", verbose = FALSE)
libd_mousebrain <- FindNeighbors(libd_mousebrain, reduction = "pca", dims = 1:30)
n_cluster <- 20
save_obj <- data.frame(ID = colnames(libd_mousebrain))
counts <- libd_mousebrain@assays$Spatial$counts
numCluster = n_cluster
coldata <- data.frame(row = libd_mousebrain@images$slice1@coordinates$col, col = libd_mousebrain@images$slice1@coordinates$row)
sce <- SingleCellExperiment(assays=list(counts=as(counts, "dgCMatrix")),
                              colData=coldata[,1:2])
bs_obj <- sce
for (i in seq(1,2)){
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

write.csv(save_obj, file = paste0(save_dir, "//Bayes_ClusterOutput.csv"),
          row.names = F)

