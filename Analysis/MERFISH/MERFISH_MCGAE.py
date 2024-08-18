# -*- coding:utf-8 -*-
# environment: python3.9

import os

import matplotlib.pyplot as plt
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
import stlearn as st
import squidpy as sq
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
file_path = os.path.join(BASE_DIR, "benchmark", "MERFISH")
dir_path = os.path.join(file_path, "raw_data")
csv_file_path = os.path.join(file_path, "cluster_metric", "MERFISHSum.csv")


adata = sc.read(os.path.join(dir_path, "subMERFISH.h5ad"))
n_clusters = len(np.unique(adata.obs["Cell_class"]))

# sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
st.pp.filter_genes(adata, min_cells=1)
st.pp.normalize_total(adata)
st.pp.log1p(adata)
# adata.X = data.X
save_obj_z = pd.DataFrame()

# Lists to store ARI scores
leiden_ari_list = []
mclust_before_refine_ari_list = []
mclust_refine_ari_list = []

# Lists to store i and ARI values when ARI > 0.51
selected_ari = []

for i in range(1,11):
    print("Now the cycle is:", i)
    set_seed(i)
    st.em.run_pca(adata, n_comps=50,random_state=i)
    compute_adata_components(adata, n_components=50)
    adata.obsm["feat_pca_vae"] = adata.obsm["X_pca"]
    # adata.obsm["feat_pca_orig"] = adata.X.A
    # adata.obsm['feat_pca_corr'] = adata.X.A


    model = MCGAE(
        adata,
        n_latent=50,
        n_components=50,
        use_pca=True,
        fusion_mode="vanilla",
        use_emb_x_rec=True,
        use_emb_g_rec=True,
        dropout=0.01,
        random_seed=i,
        w_morph=0,
    )

    model.train(
        weight_decay=5e-4,
        max_epochs=10,
        w_recon_x=0.1,
        w_recon_g=0.1,
        w_contrast=0.5,
        w_cluster=1,
        n_clusters=n_clusters,
        cl_start_epoch=0,
        compute_g_loss="cross_entropy",
        cluster_method="kmeans",
        adj_diag=1
    )
    temp = model.get_model_output()
    emb, y_pred = temp["emb"], temp["y_pred"]
    adata.obsm["z"] = emb
    adata.obs["pred"] = y_pred
    # model.plot_train_loss(pause_time=20)
    from sklearn.cluster import KMeans

    # 使用 `KMeans` 进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=i)
    adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm["z"])
    adata.obs['kmeans'] = pd.Categorical(adata.obs['kmeans'])
    import matplotlib.pyplot as plt
    sc.pl.embedding(adata, basis="spatial3d", projection="3d", color="kmeans")
    plt.savefig(os.path.join(file_path, "plot", "MCGAE.pdf"), format="pdf", bbox_inches='tight')

    # 计算调整兰德指数（ARI）
    ari = ari_score(adata.obs['Cell_class'], adata.obs['kmeans'])

    # 输出聚类后的类数
    num_clusters = len(np.unique(adata.obs['kmeans']))
    print(f"Number of clusters: {num_clusters}")
    print(f"ARI score: {ari}")

    sc.pl.embedding(adata, basis="spatial3d", projection="3d", color="kmeans")
    # # # Append ARI to CSV
    result_df = pd.DataFrame([[i, "MCGAE", ari]], columns=["Cycle", "method", "ARI"])
    if not os.path.isfile(csv_file_path):
        result_df.to_csv(csv_file_path, index=False)
    else:
        result_df.to_csv(csv_file_path, mode='a', header=False, index=False)