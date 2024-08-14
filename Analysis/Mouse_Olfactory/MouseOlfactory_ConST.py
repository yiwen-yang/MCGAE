# use ConST_env environment
import torch
import argparse
import random
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use("Agg")
sys.path.append(r"D:\Work\code_learning\conST-main")
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_ST_file, res_search_fixed_clus, plot_clustering
from src.training import conST_training

import anndata
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc
import os
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=300, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--use_img', type=bool, default=False, help='Use histology images.')
parser.add_argument('--img_w', type=float, default=0.1, help='Weight of image features.')
parser.add_argument('--use_pretrained', type=bool, default=False, help='Use pretrained weights.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--beta', type=float, default=100, help='beta value for l2c')
parser.add_argument('--cont_l2l', type=float, default=0.3, help='Weight of local contrastive learning loss.')
parser.add_argument('--cont_l2c', type=float, default=0.1, help='Weight of context contrastive learning loss.')
parser.add_argument('--cont_l2g', type=float, default=0.1, help='Weight of global contrastive learning loss.')

parser.add_argument('--edge_drop_p1', type=float, default=0.1, help='drop rate of adjacent matrix of the first view')
parser.add_argument('--edge_drop_p2', type=float, default=0.1, help='drop rate of adjacent matrix of the second view')
parser.add_argument('--node_drop_p1', type=float, default=0.2, help='drop rate of node features of the first view')
parser.add_argument('--node_drop_p2', type=float, default=0.3, help='drop rate of node features of the second view')

# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')

params = parser.parse_args(args=['--k', '20', '--knn_distanceType', 'euclidean', '--epochs', '200'])

np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using device: ' + device)
params.device = device


# set seed before every run
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# read related files
# read dataset
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
# adata = adata[truth.index, :]
adata.var_names_make_unique()
n_clusters = 7
adata_h5 = adata.copy()
# sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
adata_X = sc.pp.scale(adata_X)
adata_X = sc.pp.pca(adata_X, n_comps=300)
graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0], params)
params.cell_num = adata.shape[0]
save_obj = pd.DataFrame()
for i in range(1, 2):
    params.seed = i
    seed_torch(params.seed)
    if params.use_img:
        img_transformed = np.load('./MAE-pytorch/extracted_feature.npy')
        img_transformed = (img_transformed - img_transformed.mean()) / img_transformed.std() * adata_X.std() + adata_X.mean()
        conST_net = conST_training(adata_X, graph_dict, params, n_clusters, img_transformed)
    else:
        conST_net = conST_training(adata_X, graph_dict, params, n_clusters)
    if params.use_pretrained:
        conST_net.load_model('conST_151673.pth')
    else:
        conST_net.pretraining()
        conST_net.major_training()

    conST_embedding = conST_net.get_embedding()

    # np.save(f'{params.save_path}/conST_result.npy', conST_embedding)
    # clustering
    adata_conST = anndata.AnnData(conST_embedding)
    adata_conST.obs_names = adata.obs_names
    # adata_conST.uns['spatial'] = adata_h5.uns['spatial']
    adata_conST.obsm['spatial'] = adata_h5.obsm['spatial']

    sc.pp.neighbors(adata_conST, n_neighbors=params.eval_graph_n)

    eval_resolution = res_search_fixed_clus(adata_conST, n_clusters)
    print(eval_resolution)
    sc.tl.leiden(adata_conST, resolution=eval_resolution)
    sc.pl.embedding(adata_conST, basis="spatial", color="leiden", s=20, show=False, title='conST')
    plt.tight_layout()

    plt.savefig(os.path.join(file_path, "result_new", "conST_MO.pdf"), format="pdf")
    