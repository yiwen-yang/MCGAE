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

all(rownames(position) %in% Cells(adata))


counts_filtered <- counts[, used_barcodes$V1, drop = FALSE]

position_filtered <- position[used_barcodes$V1, ]

adata <- CreateSeuratObject(counts = counts_filtered, assay = "Spatial")


adata <- AddMetaData(adata, metadata = position_filtered, col.name = c("x", "y"))

libd_mousebrain <- adata

# pre-filter for generate simulated data quickly
libd_mousebrain <- SCTransform(libd_mousebrain, assay = "Spatial")
libd_mousebrain <- RunPCA(libd_mousebrain, assay = "SCT", verbose = FALSE)
libd_mousebrain <- FindNeighbors(libd_mousebrain, reduction = "pca", dims = 1:30)
n_cluster <- 7
save_obj <- data.frame(ID = colnames(libd_mousebrain))
counts <- libd_mousebrain@assays$Spatial$counts
numCluster = n_cluster
coldata <- data.frame(row = position_filtered$x, col = position_filtered$y)
sce <- SingleCellExperiment(assays=list(counts=as(counts, "dgCMatrix")),
                              colData=coldata[,1:2])
bs_obj <- sce
for (i in seq(1,1)){
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

