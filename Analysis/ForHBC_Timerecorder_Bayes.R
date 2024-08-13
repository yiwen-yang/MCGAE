rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)
library(pryr)
rawdata_path <- "D:\\Work\\MCGAE project\\MCGAE-master\\benchmark\\Human_Breast_Cancer"
save_dir <- "D:\\Work\\MCGAE project\\MCGAE-master\\benchmark\\ForTimeRecord\\cluster_metric\\"
libd_mousebrain <- Load10X_Spatial(data.dir = paste0(rawdata_path, "\\raw_data"),
                                   filename = "filtered_feature_bc_matrix.h5",
                                   assay = "Spatial",
                                   filter.matrix = T)

libd_mousebrain <- merge(libd_mousebrain,y=libd_mousebrain,add.cell.ids=c("8k","8k2"), project="16k")
libd_mousebrain <- merge(libd_mousebrain,y=libd_mousebrain,add.cell.ids=c("8k","8k2"), project="16k")
libd_mousebrain <- merge(libd_mousebrain,y=libd_mousebrain,add.cell.ids=c("8k","8k2"), project="16k")

n_cluster <- 5


# 保存结果的列表
results <- list()

# 测试不同数量的细胞
for (n_cells in seq(1000, 30000, 2000)) {
  libd_mousebrain_new <- libd_mousebrain[, 1:n_cells]
  # 记录初始时间和内存使用
  initial_time <- Sys.time()
  initial_memory <- pryr::mem_used() / 1024 ** 2  # 转换为MB
  # pre-filter for generate simulated data quickly
  libd_mousebrain_new <- SCTransform(libd_mousebrain_new, assay = "Spatial")
  libd_mousebrain_new <- RunPCA(libd_mousebrain_new, assay = "SCT", verbose = FALSE)
  libd_mousebrain_new <- FindNeighbors(libd_mousebrain_new, reduction = "pca", dims = 1:30)
  counts <- libd_mousebrain_new@assays$Spatial$counts.1.1.1
  numCluster = n_cluster
  coldata <- data.frame(row = libd_mousebrain_new@images$slice1@coordinates$col, col = libd_mousebrain_new@images$slice1@coordinates$row)
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
  }
  # 记录结束时间和内存使用
  final_time <- Sys.time()
  final_memory <- pryr::mem_used() / 1024 ** 2  # 转换为MB
  
  # 计算时间消耗和内存消耗
  time_cost_sec <- as.numeric(difftime(final_time, initial_time, units = "secs"))
  memory_usage_mb <- final_memory - initial_memory
  
  # 保存结果
  results <- rbind(results, data.frame(n_cells = n_cells, time_cost_sec = time_cost_sec, memory_usage_mb = memory_usage_mb))
}

# Write the dataframe to a CSV file
write.csv(results, file = paste0(save_dir, "//Bayes_resource_usage.csv"), row.names = FALSE)

