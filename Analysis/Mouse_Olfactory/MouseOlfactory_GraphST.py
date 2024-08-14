# use py3.9 environment
import os
import torch
import pandas as pd
import scanpy as sc
import time
from sklearn import metrics
import multiprocessing as mp
from GraphST import GraphST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation
# path
os.environ['R_HOME'] = r'D:\Software\R-4.3.1'
# the location of R (used for the mclust clustering)
os.environ['R_USER'] = r'D:\Software\Anaconda\envs\py3.9\Lib\site-packages\rpy2'

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
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
n_clusters = 7
# define model
save_obj = pd.DataFrame()
begin = time.time()
model = GraphST.GraphST(adata, device=device)

# train model
adata = model.train()
end = time.time()
print("Time cost: ", end - begin)
# Only for DLPFC
# set radius to specify the number of neighbors considered during refinement
radius = 50
tool = 'leiden'  # mclust, leiden, and louvain
# Note: rpy2 is required for mclust, and rpy2 in not supported by Windows.
# clustering
from GraphST.utils import clustering
tool = 'mclust' 
clustering(adata, n_clusters, radius=radius, method=tool,
                refinement=True)  # For DLPFC dataset, we use optional refinement step.
adata.obs['mclust'] = adata.obs['domain'] 
tool = 'leiden' 
clustering(adata, n_clusters, radius=radius, method=tool, start=0.3, end=3, increment=0.02, refinement=False)
adata.obs['leiden'] = adata.obs['domain'] 

