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
import stlearn as st
from pathlib import Path

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

data = st.Read10X(dir_path)
truth = pd.read_table(os.path.join(dir_path, "metadata.tsv"), index_col=0)
data = data[truth.index, :]
n_clusters = len(truth["ground_truth"].unique())
print(n_clusters)
data.var_names_make_unique()
data.obs["label"] = truth["ground_truth"].astype("category")
TILE_PATH = Path(os.path.join(dir_path, "image_segmentation"))
TILE_PATH.mkdir(parents=True, exist_ok=True)
# pre-processing for gene count table

st.pp.filter_genes(data, min_cells=1)
st.pp.normalize_total(data)
st.pp.log1p(data)
save_obj = pd.DataFrame()
# pre-processing for spot image
st.pp.tiling(data, TILE_PATH)

    # this step uses deep learning model to extract high-level features from tile images
    # may need few minutes to be completed
st.pp.extract_feature(data)
for i in range(1, 11):
    # run PCA for gene expression data
    st.em.run_pca(data, n_comps=15)
    # stSME
    st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
    data_ = data.copy()
    data_.X = data_.obsm['raw_SME_normalized']

    st.pp.scale(data_)
    st.em.run_pca(data_, n_comps=15, random_state=i)

    st.tl.clustering.kmeans(data_, n_clusters=n_clusters, use_data="X_pca", key_added="X_pca_kmeans", random_state=i )

    st.pl.cluster_plot(data_, use_label="X_pca_kmeans")
    save_obj.index = data.obs.index
    save_obj.index.name = "ID"
    save_obj = pd.concat([save_obj, data_.obs["X_pca_kmeans"]], axis=1)
    from sklearn.metrics import adjusted_rand_score as ari_score
    ari = ari_score(data_.obs["label"], data_.obs['X_pca_kmeans'])
    print("leiden ari is :", ari)

    # Append ARI to CSV
    result_df = pd.DataFrame([[i, "stLearn", ari]], columns=["Cycle", "method", "Leiden ARI"])
    csv_file_path = os.path.join(file_path, "cluster_metric", "SGCASTandStLearn_ari.csv")
    if not os.path.isfile(csv_file_path):
        result_df.to_csv(csv_file_path, index=False)
    else:
        result_df.to_csv(csv_file_path, mode='a', header=False, index=False)
