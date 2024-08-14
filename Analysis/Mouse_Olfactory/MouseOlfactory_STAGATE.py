# use py3.9 environment

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import STAGATE_pyG

import warnings
warnings.filterwarnings("ignore")

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
sc.pp.filter_genes(adata, min_cells=50)
print('After flitering: ', adata.shape)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=50)
STAGATE_pyG.Stats_Spatial_Net(adata)
adata = STAGATE_pyG.train_STAGATE(adata, n_epochs=100)
sc.pp.neighbors(adata, use_rep='STAGATE')
sc.tl.umap(adata)
sc.tl.louvain(adata, resolution=0.8)
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.embedding(adata, basis="spatial", color="louvain",s=20, show=False, title='STAGATE')
plt.tight_layout()
plt.savefig(os.path.join(file_path, "cluster_metric", "STAGATE.pdf"), format="pdf")
plt.axis('off')
sc.pl.umap(adata, color='louvain', title='STAGATE')

