rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)
rawdata_path <- "D:\\Work\\MCGAE project\\MCGAE-master\\benchmark\\Human_Breast_Cancer"
save_dir <- "D:\\Work\\MCGAE project\\MCGAE-master\\benchmark\\Human_Breast_Cancer\\cluster_output\\"
libd_mousebrain <- Load10X_Spatial(data.dir = paste0(rawdata_path, "\\raw_data"),
                               filename = "filtered_feature_bc_matrix.h5",
                               assay = "Spatial",
                               filter.matrix = T)
label <- read.table(file = paste0(rawdata_path, "\\raw_data\\", "metadata.tsv"),sep="\t", header=TRUE)

# pre-filter for generate simulated data quickly
libd_mousebrain <- SCTransform(libd_mousebrain, assay = "Spatial")
libd_mousebrain <- RunPCA(libd_mousebrain, assay = "SCT", verbose = FALSE)
libd_mousebrain <- FindNeighbors(libd_mousebrain, reduction = "pca", dims = 1:30)
n_cluster <- length(unique(label$ground_truth))
save_obj <- data.frame(ID = colnames(libd_mousebrain))
counts <- libd_mousebrain@assays$Spatial@counts
numCluster = n_cluster
coldata <- data.frame(row = libd_mousebrain@images$slice1@coordinates$col, col = libd_mousebrain@images$slice1@coordinates$row)
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

write.csv(save_obj, file = paste0(save_dir, "HumanBreastCancer_Bayes_ClusterOutput.csv"),
          row.names = F)

