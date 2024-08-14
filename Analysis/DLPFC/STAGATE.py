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

from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
import itertools
import STAGATE_pyG
# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

slice = sys.argv[1]
"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "DLPFC", f"slide{slice}")
adata = sc.read_visium(path=os.path.join(file_path, "raw_data", f"{slice}"),
                       count_file="filtered_feature_bc_matrix.h5",
                       library_id=f"{slice}",
                       source_image_path=os.path.join(file_path, "raw_data", f"{slice}", "spatial"))
truth = pd.read_table(os.path.join(file_path, "raw_data", f"{slice}", "metadata.tsv"), index_col=0)
truth.drop(truth[truth["layer_guess_reordered"].isna()].index, inplace=True)
n_clusters = len(truth["layer_guess_reordered"].unique())
print(n_clusters)
adata = adata[truth.index, :]
adata.obs['Ground Truth'] = truth["layer_guess_reordered"]
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

save_obj = pd.DataFrame()
for i in range(1, 11):
    print("Now the cycle is:", i)
    STAGATE_pyG.utils.Cal_Spatial_Net(adata, k_cutoff=3,model="KNN")
    adata = STAGATE_pyG.train_STAGATE(adata, n_epochs=700)
    from GraphST.utils import search_res
    res = search_res(adata, use_rep="STAGATE", n_clusters=n_clusters, start=0.1, end=1, increment=0.01)
    sc.pp.neighbors(adata, use_rep="STAGATE", n_neighbors=10)
    sc.tl.leiden(adata, resolution=res)
    save_obj.index = adata.obs.index
    save_obj.index.name = "ID"
    save_obj = pd.concat([save_obj, adata.obs["leiden"]], axis=1)

save_obj.to_csv(os.path.join(file_path, "cluster_output",  f"DLPFC_Slide{slice}_STAGATE_ClusterOutput.csv"))

