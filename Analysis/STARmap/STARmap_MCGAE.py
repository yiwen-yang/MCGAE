# -*- coding:utf-8 -*-
# environment: python3.9

import os
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
import itertools
from tqdm import tqdm
# Importing custom modules
from MCGAE.model import MCGAE
from MCGAE.utils import load_dataset, norm_and_filter, compute_adata_components, search_res, refine_label, set_seed, \
    mclust_R

# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: File path
"""
os.environ['R_HOME'] = r'D:\Software\R-4.3.1'
# the location of R (used for the mclust clustering)
os.environ['R_USER'] = r'D:\Software\Anaconda\envs\py3.9\Lib\site-packages\rpy2'

BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "STARmap")
dir_path = os.path.join(file_path, "raw_data")
csv_file_path = os.path.join(file_path, "cluster_metric", "STARmapSum.csv")

set_seed(2024)
adata = sc.read(os.path.join(dir_path, "STARmap_20180505_BY3_1k.h5ad"))
n_clusters = len(np.unique(adata.obs["label"]))

sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

compute_adata_components(adata, n_components=100)
save_obj_z = pd.DataFrame()

# Lists to store ARI scores
leiden_ari_list = []
mclust_before_refine_ari_list = []
mclust_refine_ari_list = []

# Lists to store i and ARI values when ARI > 0.51
selected_ari = []

for i in range(1, 11):
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
        w_morph=0,
    )

    model.train(
        weight_decay=5e-4,
        max_epochs=600,
        w_recon_x=0.5,
        w_recon_g=0.1,
        w_contrast=0.1,
        w_cluster=1,
        n_clusters=n_clusters,
        cl_start_epoch=100,
        compute_g_loss="cross_entropy",
    )
    temp = model.get_model_output()
    emb, y_pred = temp["emb"], temp["y_pred"]
    adata.obsm["z"] = emb
    adata.obs["pred"] = y_pred

    adata = mclust_R(adata, used_obsm="z", num_cluster=n_clusters)

    new_type = refine_label(adata, key='mclust')
    adata.obs['mclust'] = new_type
    # save_obj_q = pd.concat([save_obj_q, adata.obs["pred"]], axis=1)
    ari = ari_score(adata.obs["label"], adata.obs["mclust"])
    print("mclust refine ari is :", ari)

    # Append ARI to CSV
    result_df = pd.DataFrame([[i, "MCGAE", ari]], columns=["Cycle", "method", "Leiden ARI"])
    if not os.path.isfile(csv_file_path):
        result_df.to_csv(csv_file_path, index=False)
    else:
        result_df.to_csv(csv_file_path, mode='a', header=False, index=False)