# -*- coding:utf-8 -*-
# environment: python3.9
import stlearn as st
import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import warnings
import psutil
import gc
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# read files
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "ForTimeRecord")

# read h5ad
# counts = pd.read_csv(os.path.join(file_path, "raw_data", "RNA_counts.tsv"), sep="\t", index_col=0)
# position = pd.read_csv(os.path.join(file_path, "raw_data", "position.tsv"), sep="\t")
# counts.columns = ['Spot_'+str(x) for x in counts.columns]
# position.index = position['label'].map(lambda x: 'Spot_'+str(x))
# position = position.loc[:, ['x', 'y']]
# adata = sc.AnnData(counts.T)
# adata.var_names_make_unique()
# position = position.loc[adata.obs_names, ["y", "x"]]
# adata.obsm["spatial"] = position.to_numpy()
# used_barcodes = pd.read_csv(os.path.join(file_path, "raw_data", "used_barcodes.txt"), sep="\t", header=None)
# adata = adata[used_barcodes[0], :]
# adata.X = torch.from_numpy(adata.X)

# # 原始数据集的过采样
# original_adata = adata.copy()
# oversampling_factor = 2
# new_n_obs = original_adata.n_obs * oversampling_factor
# indices = np.random.choice(original_adata.n_obs, new_n_obs, replace=True)
# oversampled_adata = original_adata[indices].copy()

results = []
oversampled_adata = sc.read_h5ad(os.path.join(file_path, "FortimeRecord.h5ad"))
# 测试不同数量的细胞
for n_cells in tqdm(range(19000, 20001, 2000), desc="Processing", position=0, leave=True):
    if n_cells > oversampled_adata.shape[0]:
        break

    subset_adata = oversampled_adata[:n_cells, :].copy()

    # 强制进行垃圾回收，确保测量的准确性
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.1)  # 确保内存释放完成

    # 初始化 RAM 和 VRAM 测量
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 ** 2  # in MB
    initial_vram_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
    initial_vram_reserved = torch.cuda.memory_reserved() / 1024 ** 2    # in MB

    # 记录开始时间
    begin = time.time()

    # 处理数据
    st.pp.filter_genes(subset_adata, min_cells=1)
    st.pp.normalize_total(subset_adata)
    st.pp.log1p(subset_adata)
    st.em.run_pca(subset_adata, n_comps=50)

    st.pp.neighbors(subset_adata, n_neighbors=25, use_rep='X_pca', random_state=0)
    st.tl.clustering.louvain(subset_adata, random_state=0, resolution=1.2)

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
results_df = pd.DataFrame(results)
# results_df.to_csv(os.path.join(file_path, "cluster_metric", "stlearn_resource_usage.csv"), index=False)
#
