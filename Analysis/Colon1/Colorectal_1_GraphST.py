# use py3.9 environment
import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
import sys
import matplotlib.pyplot as plt
sys.path.append("D:\Work\code_learning\GraphST")
from GraphST import GraphST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation
# path
os.environ['R_HOME'] = r'D:\Software\Anaconda\envs\RStudio\lib\R\bin'
# read dataset
file_name = "ST-colon1"
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "Colorectal Cancer", f"{file_name}")

adata = sc.read_visium(path=os.path.join(file_path, "raw_data"),
                       count_file="filtered_feature_bc_matrix.h5",
                       library_id="none",
                       source_image_path=os.path.join(file_path, "raw_data", "spatial"))

n_clusters = 20
adata.var_names_make_unique()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, zero_center=False, max_value=10)
# define model
save_obj = pd.DataFrame()
for i in range(1, 3):
    print("Now the cycle is:", i)
    model = GraphST.GraphST(adata, device=device, random_seed=i)

    # train model
    adata = model.train()
    # Only for DLPFC
    # set radius to specify the number of neighbors considered during refinement
    radius = 50
    tool = 'leiden'  # mclust, leiden, and louvain
    # Note: rpy2 is required for mclust, and rpy2 in not supported by Windows.
    # clustering
    from GraphST.utils import clustering

    if tool == 'mclust':
        clustering(adata, n_clusters, radius=radius, method=tool,
                   refinement=True)  # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, radius=radius, method=tool, start=0.01, end=1, increment=0.01, refinement=False)
    sc.pl.spatial(adata, img_key="hires", color='leiden',
                  show=False, title="GraphST")
    plt.tight_layout()
