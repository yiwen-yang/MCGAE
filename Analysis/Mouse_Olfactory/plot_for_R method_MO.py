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
# Importing custom modules
from MCGAE.model import MCGAE
from MCGAE.utils import load_dataset, norm_and_filter, compute_adata_components, mclust_R, search_res, refine_label, \
    set_seed


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
file_path = os.path.join(BASE_DIR, "benchmark", "Mouse_Olfactory")

# read h5ad
counts = pd.read_csv(os.path.join(file_path, "raw_data", "RNA_counts.tsv"), sep="\t", index_col=0)
position = pd.read_csv(os.path.join(file_path, "raw_data", "position.tsv"), sep="\t")
counts.columns = ['Spot_' + str(x) for x in counts.columns]
position.index = position['label'].map(lambda x: 'Spot_' + str(x))
position = position.loc[:, ['x', 'y']]
adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
position = position.loc[adata.obs_names, ["y", "x"]]
adata.obsm["spatial"] = position.to_numpy()
used_barcode = pd.read_csv(os.path.join(file_path, "raw_data", "used_barcodes.txt"), sep="\t", header=None)
adata = adata[used_barcode[0], :]
adata.X = torch.from_numpy(adata.X)
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
set_seed(1234)
compute_adata_components(adata, n_components=100)
n_clusters = 7
save_obj_z = pd.DataFrame()
# save_obj_q = pd.DataFrame()

begin = time.time()
model = MCGAE(
    adata,
    n_latent=50,
    n_components=100,
    use_pca=True,
    # fusion_mode="holistic",
    fusion_mode="fractional",
    # fusion_mode="vanilla",
    use_emb_x_rec=True,
    use_emb_g_rec=True,
    dropout=0.01,  # 0.01
    random_seed=20,
    w_morph=0,
)

model.train(
    max_epochs=100,  # 600
    weight_decay=5e-4,
    # weight_decay=0.0,
    w_recon_x=0.01,
    w_recon_g=0.01,
    w_contrast=0.01,
    w_cluster=3,
    n_clusters=n_clusters,
    cl_start_epoch=50,
    compute_g_loss="cross_entropy",
    # cluster_method="kmeans",
)
end = time.time()
print("Time cost: ", end - begin)
output_mod = model.get_model_output()
emb, y_pred = output_mod["emb"], output_mod["y_pred"]
adata.obsm["z"] = emb
res = search_res(adata, n_clusters, rep="z", start=0.05, end=1)
sc.pp.neighbors(adata, use_rep='z', random_state=1234)
sc.tl.umap(adata)
sc.tl.louvain(adata, resolution=res, random_state=1234)
sc.tl.leiden(adata, resolution=res, random_state=1234)
name_tmp = 'louvain_' + "explore"
plt.rcParams["figure.figsize"] = (5, 5)
# sc.pl.embedding(adata, basis="spatial", color="pred", s=20, show=False, title='MCGAE')

# Mouse Olfactory
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "Mouse_Olfactory", "result_new")
methods = ["Giotto", "Bayes", "Seurat"]
for met in methods:
    output = pd.read_csv(os.path.join(file_path, f"{met}"+"_ClusterOutput.csv"))
    adata.uns["pred"] = output
    # 确保 ID 列与 adata.obs 的索引匹配
    output.set_index('ID', inplace=True)
    # 检查是否所有 ID 都在 adata.obs 的索引中
    if not all(output.index.isin(adata.obs.index)):
        missing_ids = set(output.ID) - set(adata.obs.index)
        print(f"Missing IDs in adata.obs: {missing_ids}")

    # 添加 cluster 列到 adata.obs 中
    adata.obs['pred'] = output.loc[adata.obs.index, 'cluster']
    # 添加 cluster 列到 adata.obs 中，并将其转换为分类变量
    adata.obs['pred'] = output.loc[adata.obs.index, 'cluster'].astype('category')
    sc.pl.embedding(adata, basis="spatial", color="pred", s=20, show=False, title=f"{met}")
    plt.savefig(os.path.join(file_path, f"{met}"+"_MO.pdf"), format="pdf")