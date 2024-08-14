# -*- coding:utf-8 -*-
# environment: sgcast_env[wsl]

# load packages
import sys
# load SGCAST and refine function
sys.path.append('/mnt/d/Work/code_learning/SGCAST-main/SGCAST-main')
from utils.utils import refine
# load packages
import os
import scanpy as sc
import torch
import copy
import numpy as np
import random
import pandas as pd
import gc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from train import Training
from sklearn.metrics.cluster import adjusted_rand_score
# Suppressing runtime warnings




class Config(object): # we create a config class to include all paths and parameters 
    def __init__(self):
        self.use_cuda = True
        self.threads = 1
        self.device = torch.device('cuda:0')
        # self.spot_paths = dir_path
        self.spot_paths = [os.path.join(dir_path, "filtered_feature_bc_matrix.h5ad")] # in spot_paths, there can be multiple paths and SGCAST will run on the data one by one
        self.nfeat = 50 # Training config
        self.nhid = 50
        self.nemb = 50
        self.batch_size = 2000  
        self.lr_start = 0.2 
        self.lr_times = 2
        self.lr_decay_epoch = 80 
        self.epochs_stage =100 
        self.seed = 2024
        self.checkpoint = ''
        self.train_conexp_ratio = 0.07 
        self.train_conspa_ratio = 0.07
        self.test_conexp_ratio = 0.07 
        self.test_conspa_ratio = 0.07 

def search_res(adata, n_clusters, method="leiden", start=0.1, end=2.0, increment=0.05, rep=None):
    """
    Searching corresponding resolution according to given cluster number

    """
    print("Searching resolution...")
    label = 0
    sc.pp.neighbors(adata, n_neighbors=10, use_rep=rep)
    res = 0.4
    count_unique = None
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["leiden"]).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs["louvain"]).louvain.unique())
            # print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    if label != 1:
        res = 1.8
        print("********************************************manual set")

    return res



"""
BASE_DIR: Project directory
data_dir: Data directory
result_dir: Result directory
file_path: File path
"""

BASE_DIR = "/mnt/d/Work/MCGAE project/MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "STARmap")
dir_path = os.path.join(file_path, "raw_data")
csv_file_path = os.path.join(file_path, "cluster_metric", "STARmapSum.csv")
adata = sc.read(os.path.join(dir_path, "STARmap_20180505_BY3_1k.h5ad"))
n_clusters = len(np.unique(adata.obs["label"]))

print(n_clusters)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, zero_center=False, max_value=10)


adata.write(os.path.join(dir_path, "filtered_feature_bc_matrix.h5ad"))
save_obj = pd.DataFrame()

for i in range(1, 11):
    print("Now cycle is :", i)
##### Prepare config and run training
    config = Config()
    config_used = copy.copy(config)
    config_used.spot_paths = config.spot_paths[0]
    new_seed = config.seed + i
    config_used.seed = new_seed
    # set random seed for each package
    torch.manual_seed(config_used.seed)
    random.seed(config_used.seed)
    np.random.seed(config_used.seed)
    # record starting time
    a = datetime.now()
    print('Start time: ', a.strftime('%H:%M:%S'))
    # reset GPU memory
    torch.cuda.reset_peak_memory_stats()
    # start training
    print('Training start ')
    model_train = Training(config_used)
    # for each epoch, there will be a reminder in a new line
    for epoch in range(config_used.epochs_stage):
        print('Epoch:', epoch)
        model_train.train(epoch)

    # finish training
    b = datetime.now()
    print('End time: ', b.strftime('%H:%M:%S'))
    c = b - a
    minutes = divmod(c.seconds, 60)
    # calculate time used in training
    print('Time used: ', minutes[0], 'minutes', minutes[1], 'seconds')
    print('Write embeddings')
    model_train.write_embeddings()
    # write result to output path
    print('Training finished: ', datetime.now().strftime('%H:%M:%S'))
    print("torch.cuda.max_memory_allocated: %fGB" % (torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024))
    spots_embeddings = np.loadtxt(os.path.join(dir_path,  "filtered_feature_bc_matrix_embeddings.txt"))#
    adata.obsm['embedding'] = np.float32(spots_embeddings)


    res = search_res(adata, n_clusters, rep="embedding")
    sc.pp.neighbors(adata, use_rep="embedding", n_neighbors=10)
    sc.tl.leiden(adata, key_added="leiden", resolution=res)
    sc.pl.embedding(adata, basis="spatial", color='leiden', s=100, show=False)
    plt.savefig(os.path.join(file_path, "plot", "SGCAST.pdf"), format="pdf")

    from sklearn.metrics import adjusted_rand_score as ari_score
    ari = ari_score(adata.obs['label'], adata.obs['leiden'])
    print("leiden ari is :", ari)

    # Append ARI to CSV
    # result_df = pd.DataFrame([[i, "SGCAST", ari]], columns=["Cycle", "method", "Leiden ARI"])
    # if not os.path.isfile(csv_file_path):
    #     result_df.to_csv(csv_file_path, index=False)
    # else:
    #     result_df.to_csv(csv_file_path, mode='a', header=False, index=False)