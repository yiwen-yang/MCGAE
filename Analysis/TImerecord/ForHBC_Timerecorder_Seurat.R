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
# pre-filter for generate simulated data quickly
libd_mousebrain <- SCTransform(libd_mousebrain, assay = "Spatial")
libd_mousebrain <- RunPCA(libd_mousebrain, assay = "SCT", verbose = FALSE)

# 保存结果的列表
results <- list()
# 测试不同数量的细胞
for (n_cells in seq(1000, 30000, 2000)) {
  libd_mousebrain_new <- libd_mousebrain[, 1:n_cells]

  # 记录初始时间和内存使用
  initial_time <- Sys.time()
  initial_memory <- pryr::mem_used() / 1024 ** 2  # 转换为MB
  

  libd_mousebrain_new <- FindNeighbors(libd_mousebrain_new, reduction = "pca", dims = 1:30)
  counts <- libd_mousebrain_new@assays$Spatial$counts
  numCluster = n_cluster
  libd_mousebrain_new <- FindClusters(libd_mousebrain_new, verbose = FALSE,resolution=0.2)

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
write.csv(results, file = paste0(save_dir, "//Seurat_resource_usage.csv"), row.names = FALSE)
