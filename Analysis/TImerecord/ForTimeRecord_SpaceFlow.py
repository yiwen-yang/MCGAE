# -*- coding:utf-8 -*-
# use spaceflow_env environment
import os
import pandas as pd
import numpy as np
import scanpy as sc
import psutil
import gc
import torch
import time
import warnings
from tqdm import tqdm
from SpaceFlow import SpaceFlow

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "ForTimeRecord")

# read h5ad
counts = pd.read_csv(os.path.join(file_path, "raw_data", "RNA_counts.tsv"), sep="\t", index_col=0)
position = pd.read_csv(os.path.join(file_path, "raw_data", "position.tsv"), sep="\t")
counts.columns = ['Spot_'+str(x) for x in counts.columns]
position.index = position['label'].map(lambda x: 'Spot_'+str(x))
position = position.loc[:, ['x', 'y']]
adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
position = position.loc[adata.obs_names, ["y", "x"]]
adata.obsm["spatial"] = position.to_numpy()
used_barcode = pd.read_csv(os.path.join(file_path, "raw_data", "used_barcodes.txt"), sep="\t", header=None)
adata = adata[used_barcode[0], :]
adata.X = torch.from_numpy(adata.X)
sc.pp.filter_genes(adata, min_cells=50)

# 原始数据集的过采样
original_adata = adata.copy()
oversampling_factor = 2
new_n_obs = original_adata.n_obs * oversampling_factor
indices = np.random.choice(original_adata.n_obs, new_n_obs, replace=True)
oversampled_adata = original_adata[indices].copy()
n_clusters = 7
save_obj_z = pd.DataFrame()
results = []

# 测试不同数量的细胞
for n_cells in tqdm(range(1000, 20001, 2000), desc="Processing", position=0, leave=True):
    if n_cells > oversampled_adata.shape[0]:
        break

    subset_adata = oversampled_adata[:n_cells, :]

    # 强制进行垃圾回收，确保测量的准确性
    gc.collect()
    torch.cuda.empty_cache()

    # 初始化 RAM 和 VRAM 测量
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 ** 2  # in MB
    initial_vram_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
    initial_vram_reserved = torch.cuda.memory_reserved() / 1024 ** 2    # in MB

    # 记录开始时间
    begin = time.time()

    sfobj = SpaceFlow.SpaceFlow(subset_adata,
                                spatial_locs=subset_adata.obsm["spatial"])
    sfobj.preprocessing_data(n_top_genes=2000)
    sfobj.train(spatial_regularization_strength=0.1,
                embedding_save_filepath=os.path.join(file_path, "data_temp", "SpaceFlow_EmbeddingFile.tsv"),
                z_dim=50,
                lr=1e-3,
                epochs=500,
                max_patience=50,
                min_stop=100,
                random_seed=1234,
                gpu=0,
                regularization_acceleration=True,
                edge_subset_sz=1000000)

    sfobj.segmentation(domain_label_save_filepath=os.path.join(file_path, "data_temp", "SpaceFlow_EmbeddingFile.tsv"),
                        n_neighbors=50,
                        resolution=0.5)
    pred_clusters = np.array(sfobj.domains).astype(int)
    
    # 记录结束时间
    end = time.time()
    elapsed_time = end - begin

    # 强制进行垃圾回收，确保测量的准确性
    gc.collect()
    torch.cuda.empty_cache()

    # 测量结束时的 RAM 和 VRAM
    final_memory = process.memory_info().rss / 1024 ** 2  # in MB
    final_vram_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
    final_vram_reserved = torch.cuda.memory_reserved() / 1024 ** 2    # in MB

    memory_usage = final_memory - initial_memory
    vram_allocated = final_vram_allocated - initial_vram_allocated
    vram_reserved = final_vram_reserved - initial_vram_reserved

    results.append({
        'n_cells': n_cells,
        'time_cost': elapsed_time,
        'memory_usage_mb': memory_usage,
        'vram_allocated_mb': vram_allocated,
        'vram_reserved_mb': vram_reserved,
    })

    print(f"Number of cells: {n_cells}")
    print(f"Time cost: {elapsed_time} seconds")
    print(f"Memory usage: {memory_usage} MB")
    print(f"Allocated VRAM: {vram_allocated} MB")
    print(f"Reserved VRAM: {vram_reserved} MB")

# 将结果保存到 CSV 文件
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(file_path, "cluster_metric", "SpaceFlow_resource_usage.csv"), index=False)
