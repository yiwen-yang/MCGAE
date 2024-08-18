# use py3.9 environment
import os
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import squidpy as sq
from sklearn import metrics
import multiprocessing as mp
import sys
import matplotlib.pyplot as plt
sys.path.append("D:\Work\code_learning\GraphST")
from GraphST import GraphST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation
# path
os.environ['R_HOME'] = r'D:\Software\R-4.3.1'
# the location of R (used for the mclust clustering)
os.environ['R_USER'] = r'D:\Software\Anaconda\envs\py3.9\Lib\site-packages\rpy2'

# sc.logging.print_header()
# print(f"squidpy=={sq.__version__}")

# load the pre-processed dataset
# adata = sq.datasets.merfish()
# adata.write(os.path.join(dir_path, "MERFISH.h5ad"))
# read dataset
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "MERFISH")
dir_path = os.path.join(file_path, "raw_data")
csv_file_path = os.path.join(file_path, "cluster_metric", "MERFISHSum.csv")
adata = sc.read(os.path.join(dir_path, "subMERFISH.h5ad"))
adata.var_names_make_unique()
n_clusters = len(np.unique(adata.obs["Cell_class"]))
print(n_clusters)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, zero_center=False, max_value=10)
# define model
save_obj = pd.DataFrame()
for i in range(1, 11):
    print("Now the cycle is:", i)
    model = GraphST.GraphST(adata, device=device, random_seed=i)

    # train model
    adata = model.train()
    # Only for DLPFC
    # set radius to specify the number of neighbors considered during refinement
    radius = 50
    tool = 'mclust'  # mclust, leiden, and louvain
    # Note: rpy2 is required for mclust, and rpy2 in not supported by Windows.
    # clustering
    from GraphST.utils import clustering

    if tool == 'mclust':
        clustering(adata, n_clusters, radius=radius, method=tool,
                   refinement=True)  # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, radius=radius, method=tool, start=0.3, end=3, increment=0.05, refinement=False)
    # sc.pl.embedding(adata, basis="spatial", color='leiden', s=100, show=False)
    # plt.savefig(os.path.join(file_path, "plot", "GraphST.pdf"), format="pdf")
    from sklearn.metrics import adjusted_rand_score as ari_score
    ari = ari_score(adata.obs['Cell_class'], adata.obs['mclust'])
    print("leiden ari is :", ari)
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    sc.pl.embedding(adata, basis="spatial3d", projection="3d", color="mclust", show=False)
    plt.savefig(os.path.join(file_path, "plot", "GraphST.pdf"), format="pdf", bbox_inches='tight')
    # # Append ARI to CSV
    result_df = pd.DataFrame([[i, "GraphST", ari]], columns=["Cycle", "method", "ARI"])
    # if not os.path.isfile(csv_file_path):
    #     result_df.to_csv(csv_file_path, index=False)
    # else:
    #     result_df.to_csv(csv_file_path, mode='a', header=False, index=False)
