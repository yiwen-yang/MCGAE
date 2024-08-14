rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)
rawdata_path <- "D:\\Work\\MCGAE project\\MCGAE-master\\benchmark\\DLPFC\\slide151507\\raw_data\\"
save_dir <- "D:\\Work\\MCGAE project\\MCGAE-master\\benchmark\\DLPFC\\slide151507\\cluster_output\\"
libd_151507 <- Load10X_Spatial(data.dir = paste0(rawdata_path, "151507"),
                               filename = "filtered_feature_bc_matrix.h5",
                               assay = "Spatial",
                               slice = "151507",
                               filter.matrix = T)
label <- read.table(file = paste0(rawdata_path, "151507\\", "metadata.tsv"))
label <- label[!is.na(label$layer_guess_reordered),]
libd_151507 <- libd_151507[, colnames(libd_151507) %in% label$barcode]
# pre-filter for generate simulated data quickly
libd_151507 <- SCTransform(libd_151507, assay = "Spatial")
libd_151507 <- RunPCA(libd_151507, assay = "SCT", verbose = FALSE)
libd_151507 <- FindNeighbors(libd_151507, reduction = "pca", dims = 1:30)
n_cluster <- length(unique(label$layer_guess_reordered))
save_obj <- data.frame(ID = label$barcode)
counts <- libd_151507@assays$Spatial@counts
numCluster = n_cluster
coldata <- data.frame(row = libd_151507@images$X151507@coordinates$col, col = libd_151507@images$X151507@coordinates$row)
sce <- SingleCellExperiment(assays=list(counts=as(counts, "dgCMatrix")),
                              colData=coldata[,1:2])
bs_obj <- sce
for (i in seq(1,10)){
    set.seed(i)
    bs_obj <- spatialPreprocess(bs_obj, platform="Visium", n.HVGs=3000, n.PCs=30, log.normalize = T)
    set.seed(i)
    bs_obj <- spatialCluster(bs_obj, q=numCluster, platform="Visium",
                             init.method="mclust", model="t",
                             nrep=10000, burn.in=1000,
                             save.chain=TRUE)
    comp_df <- as.data.frame(colData(bs_obj))
    save_obj <- cbind(save_obj,  cluster= comp_df[, 5])
}

write.csv(save_obj, file = paste0(save_dir, "DLPFC_Slide151507_Bayes_ClusterOutput.csv"),
          row.names = F)

