rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)
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
# label <- read.table(file = paste0(rawdata_path, "\\raw_data\\", "metadata.tsv"),sep="\t", header=TRUE)

# pre-filter for generate simulated data quickly
libd_mousebrain <- SCTransform(libd_mousebrain, assay = "Spatial")
libd_mousebrain <- RunPCA(libd_mousebrain, assay = "SCT", verbose = FALSE)
libd_mousebrain <- FindNeighbors(libd_mousebrain, reduction = "pca", dims = 1:30)
n_cluster <- 7
save_obj <- data.frame(ID = colnames(libd_mousebrain))
set.seed(1)
libd_mousebrain <- FindClusters(libd_mousebrain, verbose = FALSE,resolution=0.2,random.seed=1)
cat("Final cluster is: ", length(unique(libd_mousebrain@meta.data$seurat_clusters)))
seurat_output <- data.frame(cluster = libd_mousebrain@meta.data$seurat_clusters)
save_obj <- cbind(save_obj,  cluster= seurat_output)

write.csv(save_obj, file = paste0(save_dir, "//Seurat_ClusterOutput.csv"),
          row.names = F)
