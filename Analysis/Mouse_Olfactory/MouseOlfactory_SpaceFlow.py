# use spaceflow_env enviroment
import os, csv, re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import squidpy as sq
from SpaceFlow import SpaceFlow
from scipy.sparse import issparse
import random, torch
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg') 
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import sklearn


"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""
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
sc.pp.filter_genes(adata, min_cells=50)
n_clusters = 7
# SpaceFlow Object
save_obj = pd.DataFrame()

sfobj = SpaceFlow.SpaceFlow(adata,
                            spatial_locs=adata.obsm["spatial"])
sfobj.preprocessing_data(n_top_genes=2000)
sfobj.train(spatial_regularization_strength=0.1,
            embedding_save_filepath=os.path.join(file_path, "data_temp", "SpaceFlow_EmbeddingFile.tsv"),
            z_dim=50,
            lr=1e-3,
            epochs=500,
            max_patience=50,
            min_stop=100,
            random_seed=1234,
            gpu=0,
            regularization_acceleration=True,
            edge_subset_sz=1000000)

sfobj.segmentation(domain_label_save_filepath=os.path.join(file_path, "data_temp", "SpaceFlow_EmbeddingFile.tsv"),
                        n_neighbors=50,
                        resolution=0.5)
pred_clusters = np.array(sfobj.domains).astype(int)
print("Cluster number is: ", len(np.unique(pred_clusters)))
# if len(np.unique(pred_clusters)) != 7:
#     sfobj.segmentation(domain_label_save_filepath=os.path.join(file_path, "data_temp", "SpaceFlow_EmbeddingFile.tsv"),
#                         n_neighbors=50,
#                         resolution=0.55)
adata.obs["pred"] = pd.Categorical(pred_clusters)
sc.pl.embedding(adata, basis="spatial", color="pred", s=20, show=False, title='SpaceFlow')
plt.savefig(os.path.join(file_path, "result_new", "SpaceFlow_MO.pdf"), format="pdf")


