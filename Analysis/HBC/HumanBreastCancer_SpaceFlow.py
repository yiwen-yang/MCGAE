# use spaceflow_env enviroment
import os, csv, re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import squidpy as sq
import sys

from SpaceFlow import SpaceFlow
from scipy.sparse import issparse
import random, torch
import warnings

warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import sklearn


# read dataset
file_path = r"D:\\Work\\MCGAE project\\MCGAE-master\\benchmark\\Human_Breast_Cancer"
adata = sc.read_visium(path=os.path.join(file_path, "raw_data"),
                       count_file="filtered_feature_bc_matrix.h5",
                       library_id="none",
                       source_image_path=os.path.join(file_path, "raw_data", "spatial"))
truth = pd.read_table(os.path.join(file_path, "raw_data", "metadata.tsv"), index_col=0)
n_clusters = len(truth["ground_truth"].unique())
adata = adata[truth.index, :]
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
    print("Final cluster number is: ", len(np.unique(pred_clusters)))
    pred_clusters = pd.DataFrame(pred_clusters, index=adata.obs.index, columns=["cluster"])
    save_obj = pd.concat([save_obj, pred_clusters], axis=1)

save_obj.to_csv(os.path.join(file_path, "cluster_output", "HumanBreastCancer_SpaceFlow_ClusterOutput.csv"))


