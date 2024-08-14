# -*- coding:utf-8 -*-
# use py3.9 environment
import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from STMGCN.loss import target_distribution, kl_loss
import torch.optim as optim
from torch.nn.parameter import Parameter
from anndata import AnnData
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from STMGCN.util import *
import torch.nn as nn
import argparse
from sklearn.decomposition import PCA
from STMGCN.models import *

BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "STARmap")

def train_MSpaGCN(opts):
    save_obj = pd.DataFrame()
    save_obj.index.name = "ID"
    for i in range(1, 11):
        opts.seed = i
        if opts.dataset == 'DLPFC':
            features_adata, features, labels = load_data(opts.dataset, opts.sicle, opts.npca)
        else:
            features_adata, features, labels = load_data(opts.dataset, 50, 50)
        adj1, adj2 = load_graph(opts.dataset, opts.sicle, opts.l)

        model = STMGCN(nfeat=features.shape[1],
                       nhid1=opts.nhid1,
                       nclass=opts.n_cluster
                       )
        if opts.cuda:
            model.cuda()
            features = features.cuda()
            adj1 = adj1.cuda()
            adj2 = adj2.cuda()

        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        emb = model.mgcn(features, adj1, adj2)

        if opts.initcluster == "kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            n_clusters = opts.n_cluster
            kmeans = KMeans(n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(emb.detach().cpu().numpy())

        elif opts.initcluster == "louvain":
            print("Initializing cluster centers with louvain,resolution=", opts.res)
            adata = sc.AnnData(emb.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=opts.n_neighbors)
            sc.tl.louvain(adata, resolution=opts.res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            n = len(np.unique(y_pred))

        emb = pd.DataFrame(emb.detach().cpu().numpy(), index=np.arange(0, emb.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, emb.shape[0]), name="Group")
        Mergefeature = pd.concat([emb, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        y_pred_last = y_pred
        with torch.no_grad():
            model.cluster_layer.copy_(torch.tensor(cluster_centers))

        # 模型训练
        model.train()
        for epoch in range(opts.max_epochs):
            if epoch % opts.update_interval == 0:
                _, tem_q = model(features, adj1, adj2)
                tem_q = tem_q.detach()
                p = target_distribution(tem_q)

                y_pred = torch.argmax(tem_q, dim=1).cpu().numpy()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                y = labels

                nmi = nmi_score(y, y_pred)
                ari = ari_score(y, y_pred)
                print('Iter {}'.format(epoch), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

                if epoch > 0 and delta_label < opts.tol:
                    print('delta_label ', delta_label, '< tol ', opts.tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

            optimizer.zero_grad()
            x, q = model(features, adj1, adj2)
            loss = kl_loss(q.log(), p)
            loss.backward()
            optimizer.step()

        # save embeddings
        key_added = "STMGCN"
        embeddings = pd.DataFrame(x.detach().cpu().numpy())
        embeddings.index = features_adata.obs_names
        features_adata.obsm[key_added] = embeddings.loc[features_adata.obs_names,].values
        features_adata.obs["pred"] = y_pred
        features_adata.obs["pred"] = features_adata.obs["pred"].astype('category')
        save_obj.index = features_adata.obs.index
        save_obj.index.name = "ID"
        save_obj = pd.concat([save_obj, features_adata.obs["pred"]], axis=1)
        # sc.pl.embedding(features_adata, basis="spatial", color='pred', s=100, show=False)
        # plt.savefig(os.path.join(file_path, "plot", "STMGCN.pdf"), format="pdf")
        ARI = ari_score(y, y_pred)
        nmi = nmi_score(y, y_pred)
        print('nmi {:.4f}'.format(nmi), ',ari {:.4f}'.format(ARI))
        # plot spatial
        plt.rcParams["figure.figsize"] = (6, 3)
        csv_file_path = os.path.join(file_path, "cluster_metric", "STARmapSum.csv")

        # Append ARI to CSV
        result_df = pd.DataFrame([[i, "STMGCN", ARI]], columns=["Cycle", "method", "Leiden ARI"])
        if not os.path.isfile(csv_file_path):
            result_df.to_csv(csv_file_path, index=False)
        else:
            result_df.to_csv(csv_file_path, mode='a', header=False, index=False)



def parser_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--nhid1', type=int, default=16)
    parser.add_argument('--n_cluster', default=7, type=int)
    parser.add_argument('--max_epochs', default=2000, type=int)
    parser.add_argument('--update_interval', default=3, type=int)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--dataset', type=str, default='other')  # DLPFC
    parser.add_argument('--sicle', default="151673", type=str)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--l', default=1, type=float)
    parser.add_argument('--npca', default=50, type=int)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--initcluster', default="kmeans", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    return args


if __name__ == "__main__":
    opts = parser_set()
    print(opts)
    train_MSpaGCN(opts)
