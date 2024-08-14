# -*- coding:utf-8 -*-
# environment: python3.9

import os
import sys
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
import itertools

import STAGATE_pyG
# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
file_name = "ST-colon1"
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "Colorectal Cancer", f"{file_name}")
adata = sc.read_visium(path=os.path.join(file_path, "raw_data"),
                       count_file="filtered_feature_bc_matrix.h5",
                       library_id="none",
                       source_image_path=os.path.join(file_path, "raw_data", "spatial"))
adata.var_names_make_unique()

n_clusters = 3
print(n_clusters)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

save_obj = pd.DataFrame()
for i in range(1, 2):
    print("Now the cycle is:", i)
    STAGATE_pyG.utils.Cal_Spatial_Net(adata, k_cutoff=3,model="KNN")
    adata = STAGATE_pyG.train_STAGATE(adata, n_epochs=800)
    from GraphST.utils import search_res
    res = search_res(adata, use_rep="STAGATE", n_clusters=n_clusters)
    sc.pp.neighbors(adata, use_rep="STAGATE", n_neighbors=10)
    sc.tl.leiden(adata, resolution=res)
    sc.pl.spatial(adata, img_key="hires", color='leiden',
                  show=False, title="STAGATE")
    plt.tight_layout()
