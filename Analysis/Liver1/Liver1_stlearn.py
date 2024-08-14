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


"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
# read dataset
file_name = "ST-liver1"
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "Liver", f"{file_name}")
data_path = os.path.join(file_path, "raw_data")
data = st.Read10X(data_path, count_file="filtered_feature_bc_matrix.h5")
TILE_PATH = Path(os.path.join(data_path, "image_segmentation"))
TILE_PATH.mkdir(parents=True, exist_ok=True)
# output path
OUT_PATH = Path(os.path.join(data_path, "image_tile_result"))
OUT_PATH.mkdir(parents=True, exist_ok=True)
st.pp.tiling(data, TILE_PATH)
st.pp.extract_feature(data, n_components=100)

# pre-processing for gene count table
st.pp.filter_genes(data, min_cells=1)
st.pp.normalize_total(data)
st.pp.log1p(data)
st.em.run_pca(data, n_comps=50)
data_SME = data.copy()
#apply stSME to normalise log transformed data
st.spatial.SME.SME_normalize(data_SME, use_data="raw")
data_SME.X = data_SME.obsm['raw_SME_normalized']
st.pp.scale(data_SME)
st.em.run_pca(data_SME, n_comps=50)
# K-means clustering on stSME normalised PCA
st.tl.clustering.kmeans(data_SME,n_clusters=20, use_data="X_pca", key_added="X_pca_kmeans")
# st.pl.cluster_plot(data_SME, use_label="X_pca_kmeans", size=20)
sc.pl.spatial(data_SME, img_key="hires", color='X_pca_kmeans',
              show=False, title="stlearn")


plt.savefig(os.path.join(file_path, "result_new", "stLearn_liver1.pdf"), format="pdf")