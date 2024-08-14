# use py3.9 environment
import os, csv, re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import sys
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings

warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import sklearn

# read files
file_path = r"D:\\Work\\MCGAE project\\MCGAE-master\\benchmark\\Human_Breast_Cancer"
adata = sc.read_visium(path=os.path.join(file_path, "raw_data"),
                       count_file="filtered_feature_bc_matrix.h5",
                       library_id="none",
                       source_image_path=os.path.join(file_path, "raw_data", "spatial"))
truth = pd.read_table(os.path.join(file_path, "raw_data", "metadata.tsv"), index_col=0)
n_clusters = len(truth["ground_truth"].unique())
adata = adata[truth.index, :]
adata.var_names_make_unique()
x_pixel = adata.obsm["spatial"][:, 0].tolist()
y_pixel = adata.obsm["spatial"][:, 1].tolist()
# set histology is False
adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
adata.var_names_make_unique()
# Normalize and take log for UMI
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
save_obj = pd.DataFrame()
# p: percentage of total expression contributed by neighborhoods.
p = 0.5
# l: parameter to control p.
# Search l from 100 to 500
for i in range(1, 11):
    l = spg.search_l(p=p, adj=adj, start=0.01, end=1000, tol=0.01, max_run=100)
    # Set seed
    save_obj.index = adata.obs.index
    save_obj.index.name = "ID"
    r_seed = t_seed = n_seed = i
    # Search for suitable resolution
    res = spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20,
                         r_seed=r_seed,
                         t_seed=t_seed, n_seed=n_seed)
    clf = spg.SpaGCN()
    clf.set_l(l)
    # Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    # Run
    clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred
    adata.obs["pred"] = adata.obs["pred"].astype('category')
    save_obj = pd.concat([save_obj, adata.obs["pred"]], axis=1)

save_obj.to_csv(os.path.join(file_path, "cluster_output", "HumanBreastCancer_SpaGCN_ClusterOutput.csv"))