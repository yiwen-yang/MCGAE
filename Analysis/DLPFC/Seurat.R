rm(list = ls())
library(ggplot2)
library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)

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

res <- seq(0.4,1.5,0.1)
for (reso in res){
  libd_151507 <- FindClusters(libd_151507, verbose = FALSE,resolution=reso,random.seed=1)
  if (length(unique(libd_151507@meta.data$seurat_clusters)) == n_cluster){
    cat("Final set resolution is: ", reso)
    break}
  else {
    next  
  }}
if (length(unique(libd_151507@meta.data$seurat_clusters)) != n_cluster){
  reso =  0.3
  cat("manually set resolution is: ", reso)
}
save_obj <- data.frame(ID = label$barcode)
for (i in seq(1,10)){
  libd_151507 <- FindClusters(libd_151507, verbose = FALSE,resolution=reso,random.seed=i)
  seurat_output <- data.frame(cluster = libd_151507@meta.data$seurat_clusters)
  save_obj <- cbind(save_obj,  cluster= seurat_output)
}
write.csv(save_obj, file = paste0(save_dir, "DLPFC_Slide151507_Seurat_clusteroutput.csv"), 
          row.names = F)

