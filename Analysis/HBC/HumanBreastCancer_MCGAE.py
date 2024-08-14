# -*- coding:utf-8 -*-
# environment: python3.9

import os
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
import torch.nn as nn
import warnings

from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
import itertools
# Importing custom modules
from MCGAE.model import MCGAE
from MCGAE.utils import load_dataset, norm_and_filter, compute_adata_components, search_res, refine_label, set_seed

# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "Human_Breast_Cancer")
dir_path = os.path.join(file_path, "raw_data")
truth = pd.read_table(os.path.join(dir_path, "metadata.tsv"), index_col=0)
set_seed(2023)
adata = sc.read(os.path.join(dir_path, "adata.h5ad"))
adata = norm_and_filter(adata)
adata = adata[truth.index, :]
adata.obs["label"] = truth["ground_truth"].astype("category")
n_clusters = len(truth["ground_truth"].unique())
print(n_clusters)

compute_adata_components(adata, n_components=100)
save_obj_z = pd.DataFrame()
# save_obj_q = pd.DataFrame()
for i in range(1,11):
    print("Now the cycle is:", i)
    model = MCGAE(
        adata,
        n_latent=50,
        n_components=100,
        use_pca=True,
        fusion_mode="fractional",
        use_emb_x_rec=True,
        use_emb_g_rec=True,
        dropout=0.01,
        random_seed=i,
        w_morph=1.9,
    )
    
    
    model.train(
        weight_decay=5e-4,
        w_recon_x=0.5,
        w_recon_g=0.1,
        w_contrast=0.1,
        w_cluster=1,
        n_clusters=n_clusters,
        cl_start_epoch=50,
        compute_g_loss="cross_entropy",
    )
    temp = model.get_model_output()
    emb, y_pred = temp["emb"], temp["y_pred"]
    adata.obsm["z"] = emb
    adata.obs["pred"] = y_pred
    res = search_res(adata, n_clusters, rep="z")
    sc.pp.neighbors(adata, use_rep="z", n_neighbors=10, random_state=2023)
    sc.tl.leiden(adata, key_added="leiden", resolution=res, random_state=2023)
    adata.obs["leiden"] = refine_label(adata, key='leiden', radius=100)