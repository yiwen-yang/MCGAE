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

slices = ["151507", "151508", "151509",
         "151510", "151669", "151670",
         "151671", "151672", "151673",
         "151674", "151675", "151676"]

for slice in slices:
    print("Now slice is :", slice)
    BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
    file_path = os.path.join(BASE_DIR, "benchmark", "DLPFC", f"slide{slice}")
    dir_path = os.path.join(file_path, "raw_data", f"{slice}")
    TILE_PATH = Path(os.path.join(dir_path, "image_segmentation"))
    TILE_PATH.mkdir(parents=True, exist_ok=True)
    truth = pd.read_table(os.path.join(dir_path, "metadata.tsv"), index_col=0)
    truth.drop(truth[truth["layer_guess_reordered"].isna()].index, inplace=True)
    n_clusters = len(truth["layer_guess_reordered"].unique())
    # load data
    data = st.Read10X(dir_path)
    data = data[truth.index, :]
    data.obs["label"] = truth["layer_guess_reordered"].astype("category")

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

        save_obj.index = data.obs.index
        save_obj.index.name = "ID"
        save_obj = pd.concat([save_obj, data_.obs["X_pca_kmeans"]], axis=1)
        # print(save_obj)
    save_obj.to_csv(os.path.join(file_path, "cluster_output", f"DLPFC_Slide{slice}_stLearn_ClusterOutput.csv"))
