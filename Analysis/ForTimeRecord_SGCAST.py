# -*- coding:utf-8 -*-
# environment: sgcast_env[wsl]

# load packages
import sys
# load SGCAST and refine function
sys.path.append('/mnt/d/Work/code_learning/SGCAST-main/SGCAST-main')
from utils.utils import refine
# load packages
import os
import scanpy as sc
import torch
import copy
import numpy as np
import random
import pandas as pd
import gc
import matplotlib.pyplot as plt
import matplotlib
import psutil
import time
from tqdm import tqdm
from datetime import datetime
from train import Training 
from sklearn.metrics.cluster import adjusted_rand_score

matplotlib.use('Agg')

# Suppressing runtime warnings
# warnings.filterwarnings("ignore")

"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""

BASE_DIR = "/mnt/d/Work/MCGAE project/MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "ForTimeRecord")
dir_path = os.path.join(file_path, "raw_data")


class Config(object): # we create a config class to include all paths and parameters 
    def __init__(self):
        self.use_cuda = True
        self.threads = 1
        self.device = torch.device('cuda:0')
        # self.spot_paths = dir_path
        self.spot_paths = [os.path.join(dir_path, "filtered_feature_bc_matrix.h5ad")] # in spot_paths, there can be multiple paths and SGCAST will run on the data one by one
        self.nfeat = 50 # Training config
        self.nhid = 50
        self.nemb = 50
        self.batch_size = 2000  
        self.lr_start = 0.2 
        self.lr_times = 2
        self.lr_decay_epoch = 80 
        self.epochs_stage =100 
        self.seed = 2022
        self.checkpoint = ''
        self.train_conexp_ratio = 0.07 
        self.train_conspa_ratio = 0.07
        self.test_conexp_ratio = 0.07 
        self.test_conspa_ratio = 0.07 

def search_res(adata, n_clusters, method="leiden", start=0.1, end=2.0, increment=0.05, rep=None):
    """
    Searching corresponding resolution according to given cluster number
    """
    print("Searching resolution...")
    label = 0
    sc.pp.neighbors(adata, n_neighbors=10, use_rep=rep)
    res = 0.4
    count_unique = None
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["leiden"]).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["louvain"]).louvain.unique())
        if count_unique == n_clusters:
            label = 1
            break

    if label != 1:
        res = 1.8
        print("********************************************manual set")

    return res

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
# used_barcode = pd.read_csv(os.path.join(file_path, "raw_data", "used_barcodes.txt"), sep="\t", header=None)
# adata = adata[used_barcode[0], :]
# adata.X = torch.from_numpy(adata.X)
# n_clusters = 7
# adata.var_names_make_unique()
# sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.scale(adata, zero_center=False, max_value=10)
#
#
# save_obj = pd.DataFrame()
#
# # 原始数据集的过采样
# original_adata = adata.copy()
# oversampling_factor = 2
# new_n_obs = original_adata.n_obs * oversampling_factor
# indices = np.random.choice(original_adata.n_obs, new_n_obs, replace=True)
# oversampled_adata = original_adata[indices].copy()
# results = []

# oversampled_adata.write(os.path.join(dir_path, "adataForCalRAM.adata"),)
# 测试不同数量的细胞
results = []
oversampled_adata = sc.read_h5ad(os.path.join(file_path, "FortimeRecord.h5ad"))
for n_cells in tqdm(range(15000, 20001, 2000), desc="Processing", position=0, leave=True):
    if n_cells > oversampled_adata.shape[0]:
        break

    subset_adata = oversampled_adata[:n_cells, :].copy()
    subset_adata.write(os.path.join(dir_path, "filtered_feature_bc_matrix.h5ad"))
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

    # Prepare config and run training
    config = Config()
    config_used = copy.copy(config)
    config_used.spot_paths = config.spot_paths[0]

    torch.manual_seed(config_used.seed)
    random.seed(config_used.seed)
    np.random.seed(config_used.seed)

    print('Training start')
    model_train = Training(config_used)
    for epoch in range(config_used.epochs_stage):
        print('Epoch:', epoch)
        model_train.train(epoch)

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
results_df.to_csv(os.path.join(file_path, "cluster_metric", "SGCAST_resource_usage.csv"), index=False)
