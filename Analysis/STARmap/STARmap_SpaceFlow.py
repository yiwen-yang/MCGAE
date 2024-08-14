# use spaceflow_env enviroment_Interactive
import os, csv, re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import squidpy as sq
import sys

sys.path.append("D:\Work\code_learning\SpaceFlow-master")
from SpaceFlow import SpaceFlow
from scipy.sparse import issparse
import random, torch
import warnings

warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import sklearn


# read dataset
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "STARmap")
dir_path = os.path.join(file_path, "raw_data")
csv_file_path = os.path.join(file_path, "cluster_metric", "STARmapSum.csv")
adata = sc.read(os.path.join(dir_path, "STARmap_20180505_BY3_1k.h5ad"))
adata.var_names_make_unique()
n_clusters = len(np.unique(adata.obs["label"]))
adata.var_names_make_unique()
# SpaceFlow Object
save_obj = pd.DataFrame()
for i in range(1, 11):
    print("Now the cycle is:", i)
    sfobj = SpaceFlow.SpaceFlow(adata,
                                spatial_locs=adata.obsm["spatial"])
    sfobj.preprocessing_data(n_top_genes=3000)
    sfobj.train(spatial_regularization_strength=0.1,
                embedding_save_filepath=os.path.join(file_path, "data_temp", "SpaceFlow_EmbeddingFile.tsv"),
                z_dim=50,
                lr=1e-3,
                epochs=500,
                max_patience=50,
                min_stop=100,
                random_seed=i,
                gpu=0,
                regularization_acceleration=True,
                edge_subset_sz=1000000)
    for res in np.arange(0.2, 2, 0.02):
        res = round(res, 1)
        print(res)
        sfobj.segmentation(domain_label_save_filepath=os.path.join(file_path, "data_temp", "SpaceFlow_EmbeddingFile.tsv"),
                           n_neighbors=50,
                           resolution=res)
        pred_clusters = np.array(sfobj.domains).astype(int)
        print("Cluster number is: ", len(np.unique(pred_clusters)))
        if len(np.unique(pred_clusters)) == n_clusters:
            print("Set resolustion is: ", res)
            break
        else:
            continue

    if len(np.unique(pred_clusters)) != n_clusters:
        res = 0.1
        print("Manually set resolustion is: ", res)
        sfobj.segmentation(domain_label_save_filepath=os.path.join(file_path, "data_temp", "SpaceFlow_EmbeddingFile.tsv"),
                           n_neighbors=50,
                           resolution=res)
    save_obj.index = adata.obs.index
    save_obj.index.name = "ID"
    pred_clusters = np.array(sfobj.domains).astype(int)
    # print("Final cluster number is: ", len(np.unique(pred_clusters)))
    # pred_clusters = pd.DataFrame(pred_clusters, index=adata.obs.index, columns=["cluster"])
    # save_obj = pd.concat([save_obj, pred_clusters], axis=1)
    from sklearn.metrics import adjusted_rand_score as ari_score
    ari = ari_score(adata.obs['label'], pred_clusters)
    print("leiden ari is :", ari)

    # Append ARI to CSV
    result_df = pd.DataFrame([[i, "SpaceFlow", ari]], columns=["Cycle", "method", "Leiden ARI"])
    if not os.path.isfile(csv_file_path):
        result_df.to_csv(csv_file_path, index=False)
    else:
        result_df.to_csv(csv_file_path, mode='a', header=False, index=False)
