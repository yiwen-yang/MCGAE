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
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "MERFISH")
dir_path = os.path.join(file_path, "raw_data")
csv_file_path = os.path.join(file_path, "cluster_metric", "MERFISHSum.csv")
adata = sc.read(os.path.join(dir_path, "subMERFISH.h5ad"))
adata.var_names_make_unique()
n_clusters = len(np.unique(adata.obs["Cell_class"]))
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
for i in range(1,11):
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
    clf.train(adata, adj, init_spa=True, init="kmeans", n_clusters= n_clusters, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred
    adata.obs["pred"] = adata.obs["pred"].astype('category')
    # save_obj = pd.concat([save_obj, adata.obs["pred"]], axis=1)
    from sklearn.metrics import adjusted_rand_score as ari_score
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    # sc.pl.embedding(adata, basis="spatial3d", projection="3d", color="pred")
    # plt.savefig(os.path.join(file_path, "plot", "spagcn.pdf"), format="pdf", bbox_inches='tight') 
    
    ari = ari_score(adata.obs['Cell_class'], adata.obs['pred'])
    print("leiden ari is :", ari)
    # sc.pl.embedding(adata, basis="spatial", color='pred', s=100, show=False)
    # plt.savefig(os.path.join(file_path, "plot", "SpaGCN.pdf"), format="pdf")
    # Append ARI to CSV
    result_df = pd.DataFrame([[i, "SpaGCN", ari]], columns=["Cycle", "method", "ARI"])
    if not os.path.isfile(csv_file_path):
        result_df.to_csv(csv_file_path, index=False)
    else:
        result_df.to_csv(csv_file_path, mode='a', header=False, index=False)
