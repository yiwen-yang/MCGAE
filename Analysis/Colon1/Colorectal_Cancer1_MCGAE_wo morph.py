# -*- coding:utf-8 -*-
# environment: python3.9

import os
import pandas as pd
import scanpy as sc
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
for j in range(1,2):
    j = 3
    file_name = "ST-colon" + str(j)
    BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
    file_path = os.path.join(BASE_DIR, "benchmark", "Colorectal Cancer", f"{file_name}")
    dir_path = os.path.join(file_path, "raw_data")
    set_seed(1234)
    adata = load_dataset(dir_path, use_image=False)
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var["highly_variable"]]
    n_clusters = 20
    print(n_clusters)
    compute_adata_components(adata, n_components=100)
    for i in range(1, 2):
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
            random_seed=24,
            w_morph=0,
        )
       
        
        model.train(
            weight_decay=5e-4,
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
        res = search_res(adata, n_clusters, rep="z", start=0.3, end=3, increment=0.02)
        sc.pp.neighbors(adata, use_rep="z", n_neighbors=10, random_state=1234)
        sc.tl.leiden(adata, key_added="leiden", resolution=res, random_state=1234)
        new_type = refine_label(adata, key='leiden', radius=30)
        adata.obs['leiden'] = new_type
        sc.pl.spatial(adata, img_key="hires", color='leiden',
                    show=False, title="MCGAE")
        plt.tight_layout()

