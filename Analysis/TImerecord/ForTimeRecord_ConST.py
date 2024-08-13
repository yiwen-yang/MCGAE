# -*- coding:utf-8 -*-
# use ConST_env environment
import torch
import argparse
import random
import numpy as np
import pandas as pd
import sys
import psutil
import gc
import time
from tqdm import tqdm

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
file_path = os.path.join(BASE_DIR, "benchmark", "ForTimeRecord")

# # read h5ad
# counts = pd.read_csv(os.path.join(file_path, "raw_data", "RNA_counts.tsv"), sep="\t", index_col=0)
# position = pd.read_csv(os.path.join(file_path, "raw_data", "position.tsv"), sep="\t")
# counts.columns = ['Spot_'+str(x) for x in counts.columns]
# position.index = position['label'].map(lambda x: 'Spot_'+str(x))
# position = position.loc[:, ['x', 'y']]
# adata = sc.AnnData(counts.T)
# adata.var_names_make_unique()
# position = position.loc[adata.obs_names, ["y", "x"]]
# adata.obsm["spatial"] = position.to_numpy()
# used_barcode = pd.read_csv(os.path.join(file_path, "raw_data", "used_barcodes.txt"), sep="\t", header=None)
# adata = adata[used_barcode[0], :]
# adata.write(os.path.join(file_path, "adataForCalRAM_conST1.h5ad"))
adata = sc.read_h5ad(os.path.join(file_path, "adataForCalRAM_conST1.h5ad"))
adata.X = torch.from_numpy(adata.X)
adata.var_names_make_unique()
n_clusters = 7
adata_h5 = adata.copy()
adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
adata_X = sc.pp.scale(adata_X)
adata_X = sc.pp.pca(adata_X, n_comps=300)

save_obj = pd.DataFrame()

# 添加统计代码
results = []
original_adata = adata_X.copy()
original_adata2 = adata.copy()
# 设置过采样因子，例如将样本数量增加到原始数据集的2倍
oversampling_factor = 2
new_n_obs = original_adata.shape[0] * oversampling_factor
# 随机选择样本进行过采样
indices = np.random.choice(original_adata.shape[0], new_n_obs, replace=True)
# 创建新的 AnnData 对象
oversampled_adata = original_adata[indices].copy()
oversampled_adata2 = original_adata2[indices].copy()
# oversampled_adata.write(os.path.join(file_path, "adataForCalRAM_conST1.h5ad"))
# oversampled_adata2.write(os.path.join(file_path, "adataForCalRAM_conST2.h5ad"))
# oversampled_adata = sc.read_h5ad(os.path.join(file_path, "adataForCalRAM_conST1.h5ad"))
# oversampled_adata2 = sc.read_h5ad(os.path.join(file_path, "adataForCalRAM_conST2.h5ad"))
for n_cells in tqdm(range(1000, 20001, 2000), desc="Processing", position=0, leave=True):
    if n_cells > oversampled_adata.shape[0]:
        break

    subset_adata = oversampled_adata[:n_cells, :]
    subset_adata2 = oversampled_adata2[:n_cells, :]    
    graph_dict = graph_construction(subset_adata2.obsm['spatial'], subset_adata2.shape[0], params)
    params.cell_num = subset_adata.shape[0]
    params.seed = random.randint(0, 10000)
    seed_torch(params.seed)
    # 强制进行垃圾回收，确保测量的准确性
    gc.collect()
    torch.cuda.empty_cache()

    # 初始化 RAM 和 VRAM 测量
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 ** 2  # in MB
    initial_vram_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
    initial_vram_reserved = torch.cuda.memory_reserved() / 1024 ** 2    # in MB

    # 记录开始时间
    begin = time.time()
    
    if params.use_img:
        img_transformed = np.load('./MAE-pytorch/extracted_feature.npy')
        img_transformed = (img_transformed - img_transformed.mean()) / img_transformed.std() * adata_X.std() + adata_X.mean()
        conST_net = conST_training(subset_adata, graph_dict, params, n_clusters, img_transformed)
    else:
        conST_net = conST_training(subset_adata, graph_dict, params, n_clusters)
    if params.use_pretrained:
        conST_net.load_model('conST_151673.pth')
    else:
        conST_net.pretraining()
        conST_net.major_training()

    conST_embedding = conST_net.get_embedding()
    # 记录结束时间
    end = time.time()
    elapsed_time = end - begin

    # 强制进行垃圾回收，确保测量的准确性
    gc.collect()
    torch.cuda.empty_cache()
    # 测量结束时的 RAM 和 VRAM
    final_memory = process.memory_info().rss / 1024 ** 2  # in MB
    final_vram_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # in MB
    final_vram_reserved = torch.cuda.memory_reserved() / 1024 ** 2    # in MB

    memory_usage = final_memory - initial_memory
    vram_allocated = final_vram_allocated - initial_vram_allocated
    vram_reserved = final_vram_reserved - initial_vram_reserved

    results.append({
        'n_cells': n_cells,
        'time_cost': elapsed_time,
        'memory_usage_mb': memory_usage,
        'vram_allocated_mb': vram_allocated,
        'vram_reserved_mb': vram_reserved,
    })

    print(f"Number of cells: {n_cells}")
    print(f"Time cost: {elapsed_time} seconds")
    print(f"Memory usage: {memory_usage} MB")
    print(f"Allocated VRAM: {vram_allocated} MB")
    print(f"Reserved VRAM: {vram_reserved} MB")

# 将结果保存到 CSV 文件
results_df = pd.DataFrame(results)
# results_df.to_csv(os.path.join(file_path, "cluster_metric", "ConST_resource_usage.csv"), index=False)
