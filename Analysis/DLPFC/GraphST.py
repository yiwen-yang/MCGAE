# use py3.9 environment
import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
from GraphST import GraphST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation
# path
os.environ['R_HOME'] = r'D:\Software\R-4.3.1'
# the location of R (used for the mclust clustering)
os.environ['R_USER'] = r'D:\Software\Anaconda\envs\py3.9\Lib\site-packages\rpy2'
# read dataset
file_path = r"D:\Work\MCGAE project\MCGAE-master\benchmark\DLPFC\slide151507"
adata = sc.read_visium(path=os.path.join(file_path, "raw_data", "151507"),
                       count_file="filtered_feature_bc_matrix.h5",
                       library_id="151507",
                       source_image_path=os.path.join(file_path, "raw_data", "151507", "spatial"))
truth = pd.read_table(os.path.join(file_path, "raw_data", "151507", "metadata.tsv"), index_col=0)
# Need to remove NAN
truth.drop(truth[truth["layer_guess_reordered"].isna()].index, inplace=True)
n_clusters = len(truth["layer_guess_reordered"].unique())
adata = adata[truth.index, :]
adata.var_names_make_unique()
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
    tool = 'leiden'  # mclust, leiden, and louvain
    # Note: rpy2 is required for mclust, and rpy2 in not supported by Windows.
    # clustering
    from GraphST.utils import clustering

    if tool == 'mclust':
        clustering(adata, n_clusters, radius=radius, method=tool,
                   refinement=True)  # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, radius=radius, method=tool, start=0.3, end=3, increment=0.02, refinement=False)
    adata.obs["leiden"]
    save_obj.index = adata.obs.index
    save_obj.index.name = "ID"
    save_obj = pd.concat([save_obj, adata.obs["leiden"]], axis=1)

save_obj.to_csv(os.path.join(file_path, "cluster_output", "DLPFC_Slide151507_GraphST_ClusterOutput.csv"))

