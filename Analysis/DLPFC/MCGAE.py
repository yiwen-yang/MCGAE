# -*- coding:utf-8 -*-
# environment: python3.9

import os
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
import sys
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import itertools
# Importing custom modules
sys.path.append(r"D:\Work\MCGAE project\MCGAE-master\benchmark\DLPFC\slide151507\cluster_code")
from MCGAE.model import MCGAE
from MCGAE.utils import load_dataset, norm_and_filter, compute_adata_components, mclust_R, search_res, refine_label, set_seed

# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
slice = sys.argv[1]

os.environ['R_HOME'] = r'D:\Software\R-4.3.1'
# the location of R (used for the mclust clustering)
os.environ['R_USER'] = r'D:\Software\Anaconda\envs\py3.9\Lib\site-packages\rpy2'

BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "DLPFC", f"slide{slice}")
dir_path = os.path.join(file_path, "raw_data", f"{slice}")
truth = pd.read_table(os.path.join(dir_path, "metadata.tsv"), index_col=0)
truth.drop(truth[truth["layer_guess_reordered"].isna()].index, inplace=True)
n_clusters = len(truth["layer_guess_reordered"].unique())
set_seed(2023)
adata = load_dataset(dir_path, use_image=False)
adata = norm_and_filter(adata)
adata = adata[truth.index, :]
adata.var_names_make_unique()
adata.obs["label"] = truth["layer_guess_reordered"].astype("category")
print(n_clusters)
compute_adata_components(adata, n_components=100)
save_obj_z = pd.DataFrame()
save_obj_q = pd.DataFrame()
save_obj_mcluster = pd.DataFrame()
for i in range(1, 11):
    print("Now the cycle is:", i)
    model = MCGAE(adata,
                  n_latent=50,
                  n_components=100,
                  use_pca=True,
                  fusion_mode="fractional",
                  random_seed=i,
                  dropout=0.01, 
                  w_morph= 0)
    model.train(weight_decay=5e-4,
                max_epochs=600,
                w_recon_x=0.1,
                w_recon_g=0.1,
                w_contrast=0.5,
                w_cluster=1,
                cl_start_epoch=100,
                n_clusters=n_clusters,
                compute_g_loss="cross_entropy",
                adj_diag=0.9,
                 )

    temp = model.get_model_output()
    emb, y_pred = temp["emb"], temp["y_pred"]
    adata.obsm["z"] = emb
    adata.obs["pred"] = y_pred
    adata = mclust_R(adata, used_obsm="z", num_cluster=15)
    new_type = refine_label(adata, key='mclust')
    adata.obs['mclust'] = new_type 
    save_obj_mcluster = pd.concat([save_obj_mcluster, adata.obs["mclust"]], axis=1)

save_obj_mcluster.to_csv(os.path.join(file_path, "cluster_output", f"DLPFC_Slide{slice}_MCGAE_ClusterOutput.csv"))
