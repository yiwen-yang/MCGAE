# use py3.9 environment
import os, csv, re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import sys

sys.path.append("D:\Work\scMSI_project\code_learning\SpaGCN-master\SpaGCN_package")
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings

warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import sklearn

# read files
file_name = "ST-colon1"
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "Colorectal Cancer", f"{file_name}")
adata = sc.read_visium(path=os.path.join(file_path, "raw_data"),
                       count_file="filtered_feature_bc_matrix.h5",
                       library_id="none",
                       source_image_path=os.path.join(file_path, "raw_data", "spatial"))

n_clusters = 3
adata.var_names_make_unique()
x_pixel = adata.obsm["spatial"][:, 0].tolist()
y_pixel = adata.obsm["spatial"][:, 1].tolist()

adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
adata.var_names_make_unique()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
save_obj = pd.DataFrame()
# p: percentage of total expression contributed by neighborhoods.
p = 0.5
# l: parameter to control p.
# Search l from 100 to 500
for i in range(1, 2):
    l = spg.search_l(p=p, adj=adj, start=0.01, end=1000, tol=0.01, max_run=100)
    # Set seed
    save_obj.index = adata.obs.index
    save_obj.index.name = "ID"
    r_seed = t_seed = n_seed = i
    # Search for suitable resolution
    res = spg.search_res(adata, adj, l, n_clusters, start=0.01, step=0.1, tol=5e-3, lr=0.05, max_epochs=20,
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
    sc.pl.spatial(adata, img_key="hires", color='pred',
                  show=False, title="SpaGCN")
    plt.tight_layout()
