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
x_pixel = adata.obsm["spatial"][:, 0].tolist()
y_pixel = adata.obsm["spatial"][:, 1].tolist()
adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
sc.pp.filter_genes(adata, min_cells=50)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
n_clusters = 7
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
    sc.pl.embedding(adata, basis="spatial", color="pred", s=20, show=False, title='spagcn')
    plt.tight_layout()
#     tmp_path = os.path.join(file_path, "cluster_metric", "other_fig")
#     os.makedirs(tmp_path, exist_ok=True)
