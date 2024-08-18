# -*- coding:utf-8 -*-
# environment: python3.9
import stlearn as st
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
from pathlib import Path
import numpy as np


"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
# read dataset
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "MERFISH")
dir_path = os.path.join(file_path, "raw_data")
csv_file_path = os.path.join(file_path, "cluster_metric", "MERFISHSum.csv")
data = sc.read(os.path.join(dir_path, "subMERFISH.h5ad"))
data.var_names_make_unique()
n_clusters = len(np.unique(data.obs["Cell_class"]))
data.var_names_make_unique()

# pre-processing for gene count table
st.pp.filter_genes(data, min_cells=1)
st.pp.normalize_total(data)
st.pp.log1p(data)
st.em.run_pca(data, n_comps=100)
data_SME = data.copy()
# apply stSME to normalise log transformed data
for i in range(1, 11):
    print("Now the cycle is:", i)

    st.em.run_pca(data_SME, n_comps=100, random_state=i)
    # K-means clustering on stSME normalised PCA
    st.tl.clustering.kmeans(data_SME,n_clusters=n_clusters, use_data="X_pca", key_added="X_pca_kmeans", random_state=i)
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    sc.pl.embedding(data_SME, basis="spatial3d", projection="3d", color="X_pca_kmeans")
    plt.savefig(os.path.join(file_path, "plot", "Seurat.pdf"), format="pdf", bbox_inches='tight')
    from sklearn.metrics import adjusted_rand_score as ari_score
    ari = ari_score(data_SME.obs['Cell_class'], data_SME.obs['X_pca_kmeans'])
    print("leiden ari is :", ari)
    # sc.pl.embedding(data_SME, basis="spatial", color='X_pca_kmeans', s=100, show=False)
    # plt.savefig(os.path.join(file_path, "plot", "stLearn.pdf"), format="pdf")
    # Append ARI to CSV
    result_df = pd.DataFrame([[i, "stLearn", ari]], columns=["Cycle", "method", "ARI"])
    # if not os.path.isfile(csv_file_path):
    #     result_df.to_csv(csv_file_path, index=False)
    # else:
    #     result_df.to_csv(csv_file_path, mode='a', header=False, index=False)
    #
