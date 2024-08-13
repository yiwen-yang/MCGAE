rm(list=ls())
library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Matrix)
library(Seurat)
library(Giotto)
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

# 保存结果的列表
results <- list()
# 测试不同数量的细胞
for (n_cells in seq(1000, 30000, 2000)) {
  libd_mousebrain_new <- libd_mousebrain[, 1:n_cells]
  # 记录初始时间和内存使用
  initial_time <- Sys.time()
  initial_memory <- pryr::mem_used() / 1024 ** 2  # 转换为MB

libd_mousebrain_new <- FindNeighbors(libd_mousebrain_new, reduction = "pca", dims = 1:30)
save_obj <- data.frame(ID = colnames(libd_mousebrain_new))
counts <- libd_mousebrain_new@assays$Spatial$counts.1.1.1
numCluster = n_cluster
myinst  <- createGiottoInstructions(save_plot=F, show_plot=T, save_dir=save_dir,
                                    python_path="D:\\Software\\Anaconda\\envs\\py3.9")
coldata <- data.frame(row = libd_mousebrain_new@images$slice1@coordinates$col, col = libd_mousebrain_new@images$slice1@coordinates$row)
obj         <- createGiottoObject(raw_exprs = counts, spatial_locs = coldata[,1:2],instructions = myinst)
go_obj      <- filterGiotto(gobject = obj, expression_threshold = 0,gene_det_in_min_cells = 50, min_det_genes_per_cell = 2,expression_values = c('raw'),verbose = T)
go_obj      <- normalizeGiotto(gobject = go_obj, scalefactor = 6000, verbose = T)
## add gene & cell statistics
go_obj      <- addStatistics(gobject = go_obj)
## adjust expression matrix for technical or known variables
go_obj      <- adjustGiottoMatrix(gobject = go_obj, expression_values = c('normalized'),batch_columns = NULL, covariate_columns = c('nr_genes', 'total_expr'),return_gobject = TRUE,update_slot = c('custom'))
go_obj      <- calculateHVG(gobject = go_obj, method = 'cov_loess',show_plot=FALSE,save_plot=FALSE, difference_in_cov = 0.1)
## select genes based on HVG and gene statistics, both found in gene metadata
gene_metadata <- fDataDT(go_obj)
featgenes <- gene_metadata[hvg == 'yes' & perc_cells > 2 & mean_expr_det > 0.5]$gene_ID

for (i in seq(1,1)){
  go_obj <- runPCA(gobject = go_obj, scale_unit = F, center=T, method="factominer")
  go_obj <- createNearestNetwork(gobject = go_obj, type = "sNN", dimensions_to_use = 1:15, k = 15)


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
write.csv(results, file = paste0(save_dir, "//Giotto_resource_usage.csv"), row.names = FALSE)