import os
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
from sklearn.metrics import adjusted_rand_score as ari_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import STAGATE_pyG
from GraphST.utils import search_res

# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "STARmap")
dir_path = os.path.join(file_path, "raw_data")

adata = sc.read(os.path.join(dir_path, "STARmap_20180505_BY3_1k.h5ad"))
adata.var_names_make_unique()
n_clusters = len(np.unique(adata.obs["label"]))
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

csv_file_path = os.path.join(file_path, "cluster_metric", "STARmapSum.csv")

for i in range(1, 11):
    print("Now the cycle is:", i)
    STAGATE_pyG.utils.Cal_Spatial_Net(adata, k_cutoff=3, model="KNN")
    adata = STAGATE_pyG.train_STAGATE(adata, n_epochs=800, random_seed=i)

    res = search_res(adata, use_rep="STAGATE", n_clusters=n_clusters, start=0.1, end=1.0, increment=0.01)
    sc.pp.neighbors(adata, use_rep="STAGATE", n_neighbors=10)
    sc.tl.leiden(adata, resolution=res)
    sc.pl.embedding(adata, basis="spatial", color='leiden', s=100, show=False)
    plt.savefig(os.path.join(file_path, "plot", "STAGATE.pdf"), format="pdf")
    ari = ari_score(adata.obs['label'], adata.obs['leiden'])
    print("leiden ari is :", ari)

    # Append ARI to CSV
    result_df = pd.DataFrame([[i, "STAGATE", ari]], columns=["Cycle", "method", "Leiden ARI"])
    if not os.path.isfile(csv_file_path):
        result_df.to_csv(csv_file_path, index=False)
    else:
        result_df.to_csv(csv_file_path, mode='a', header=False, index=False)
