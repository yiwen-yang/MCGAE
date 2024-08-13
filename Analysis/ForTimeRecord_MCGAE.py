# -*- coding:utf-8 -*-
# environment: python3.9

import os
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import warnings
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
import itertools
import psutil
import copy
import gc
from tqdm import tqdm  # 导入 tqdm 库
# Importing custom modules
from MCGAE.model import MCGAE
from MCGAE.utils import load_dataset, norm_and_filter, compute_adata_components, mclust_R, search_res, refine_label, set_seed

# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ['R_HOME'] = r'D:\Software\R-4.3.1'
# the location of R (used for the mclust clustering)
os.environ['R_USER'] = r'D:\Software\Anaconda\envs\py3.9\Lib\site-packages\rpy2'

"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "DLPFC", "slide151669")
dir_path = os.path.join(file_path, "raw_data", "151669")

adata = load_dataset(dir_path, use_image=False)
adata = norm_and_filter(adata)


set_seed(1234)
original_adata = adata.copy()
# set oversample factor
oversampling_factor = 6
new_n_obs = original_adata.n_obs * oversampling_factor
# random sampling
indices = np.random.choice(original_adata.n_obs, new_n_obs, replace=True)
# create AnnData object
oversampled_adata = original_adata[indices].copy()
n_clusters = 5
save_obj_z = pd.DataFrame()
print(oversampled_adata.shape[0])
for j in range(10):
    print(j)
    results = []
    for n_cells in tqdm(range(19000, 20001, 2000), desc="Processing", position=0, leave=True):
        if n_cells > oversampled_adata.shape[0]:
            break

        subset_adata = oversampled_adata[:n_cells, :]
        compute_adata_components(subset_adata, n_components=100)
        
        gc.collect()
        torch.cuda.empty_cache()
        process = psutil.Process(os.getpid())

        def measure_memory():
            gc.collect()
            torch.cuda.empty_cache()
            return process.memory_info().rss / 1024 ** 2  # in MB

        def measure_vram():
            gc.collect()
            torch.cuda.empty_cache()
            return torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2  # in MB

        initial_memory = measure_memory()
        initial_vram_allocated, initial_vram_reserved = measure_vram()

        begin = time.time()

        model = MCGAE(
            subset_adata,
            n_latent=50,
            n_components=100,
            use_pca=True,
            # fusion_mode="holistic",
            fusion_mode="fractional",
            # fusion_mode="vanilla",
            use_emb_x_rec=True,
            use_emb_g_rec=True,
            dropout=0.01, 
            random_seed = j,
            w_morph=0,
        )

        model.train(
            max_epochs=600,  
            weight_decay=5e-4,
            w_recon_x=0.01,
            w_recon_g=0.01,
            w_contrast=0.01,
            w_cluster=1,
            n_clusters=n_clusters,
            cl_start_epoch=100,
            cluster_method="kmeans",
        )

        # time end
        end = time.time()
        elapsed_time = end - begin

        # 
        gc.collect()
        torch.cuda.empty_cache()
        # time.sleep(0.1)  # Ensure that memory release is complete


        final_memory = measure_memory()
        final_vram_allocated, final_vram_reserved = measure_vram()

        memory_usage =  final_memory - initial_memory  
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
        time.sleep(1)

    # 将结果保存到 CSV 文件
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(file_path, "cluster_metric", "MCGAE", "MCGAE_resource_usage20240807" + str(j) + ".csv"), index=False)
