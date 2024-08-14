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
write.csv(save_obj, file = paste0(save_dir, "HumanBreastCancer_Seurat_ClusterOutput.csv"), 
          row.names = F)



