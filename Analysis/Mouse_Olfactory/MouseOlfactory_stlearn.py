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
# read files
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "Mouse_Olfactory")


# read h5ad
counts = pd.read_csv(os.path.join(file_path, "raw_data", "RNA_counts.tsv"), sep="\t", index_col=0)
position = pd.read_csv(os.path.join(file_path, "raw_data", "position.tsv"), sep="\t")
counts.columns = ['Spot_'+str(x) for x in counts.columns]
position.index = position['label'].map(lambda x: 'Spot_'+str(x))
position = position.loc[:, ['x', 'y']]
adata = sc.AnnData(counts.T)
adata.var_names_make_unique()
position = position.loc[adata.obs_names, ["y", "x"]]
adata.obsm["spatial"] = position.to_numpy()
used_barcode = pd.read_csv(os.path.join(file_path, "raw_data", "used_barcodes.txt"), sep="\t", header=None)
adata = adata[used_barcode[0], :]
adata.X = torch.from_numpy(adata.X)
data = adata.copy()

# pre-processing for gene count table
st.pp.filter_genes(data, min_cells=1)
st.pp.normalize_total(data)
st.pp.log1p(data)
st.em.run_pca(data, n_comps=50)

st.pp.neighbors(data, n_neighbors=25, use_rep='X_pca', random_state=0)
st.tl.clustering.louvain(data, random_state=0, resolution=1.2)
sc.pl.embedding(data, basis="spatial", color="louvain", s=20, show=False, title='stlearn')
plt.savefig(os.path.join(file_path, "result_new", "stLearn_MO.pdf"), format="pdf")