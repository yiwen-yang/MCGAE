import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import stlearn as st
from pathlib import Path
import scipy
import os, ot, csv, re
import math, numba
import sklearn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from anndata import AnnData, read_csv, read_text, read_mtx
from scanpy import read_10x_h5
from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import random
import torch
from torch.backends import cudnn
import torch.nn as nn
from .layers import VAE


def load_dataset(dataset_path, use_image=True):
    adata = sc.read_visium(
        path=dataset_path, count_file="filtered_feature_bc_matrix.h5", load_images=True
    )
    adata.obs["x_pixel"] = adata.obsm["spatial"][:, 0].tolist()
    adata.obs["y_pixel"] = adata.obsm["spatial"][:, 1].tolist()
    adata.var_names_make_unique()
    if use_image:
        st.settings.set_figure_params(dpi=300)
        # spot tile is the intermediate result of image pre-processing
        TILE_PATH = Path(os.path.join(dataset_path, "image_segmentation"))
        TILE_PATH.mkdir(parents=True, exist_ok=True)
        # output path
        OUT_PATH = Path(os.path.join(dataset_path, "image_tile_result"))
        OUT_PATH.mkdir(parents=True, exist_ok=True)
        data = st.Read10X(dataset_path,
                          count_file="filtered_feature_bc_matrix.h5")
        st.pp.tiling(data, TILE_PATH)
        st.pp.extract_feature(data, n_components=100)
        adata.obsm["X_morphology"] = data.obsm["X_morphology"]

    return adata


def norm_and_filter(adata):
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    adata = adata[:, adata.var["highly_variable"]]
    # if DLPFC , PCA OR NOT
    # VAE -->Z ----MCLUST
    return adata


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def permutation(feature):
    # fix_seed(FLAGS.random_seed)
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated


def construct_interaction(adata, n_neighbors=3):
    """
    Constructing spot-to-spot interactive graph
    this code is from GraphST
    """
    position = adata.obsm["spatial"]

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric="euclidean")
    n_spot = distance_matrix.shape[0]

    adata.obsm["distance_matrix"] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm["graph_neigh"] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm["adj"] = adj


def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm["spatial"]
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm["graph_orig"] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm["adj_orig"] = adj
    print("Graph constructed!")


def construct_distance_matrix(adata, metric="euclidean"):
    position = adata.obsm["spatial"]
    n_spot = position.shape[0]
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric=metric)
    # n_spot = distance_matrix.shape[0]
    return distance_matrix, n_spot



def construct_interaction_cosine(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm["spatial"]

    # calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(position)
    n_spot = similarity_matrix.shape[0]

    adata.obsm["similarity_matrix"] = similarity_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = similarity_matrix[i, :]
        neighbors = vec.argsort()[::-1][1: n_neighbors + 1]
        interaction[i, neighbors] = 1

    adata.obsm["graph_neigh_hat"] = interaction
    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    adata.obsm["adj_aug"] = adj
    print("Graph_hat constructed!")


def get_feature(adata, n_components=50):
    pca = PCA(n_components=n_components, random_state=1234)
    if issparse(adata.X):
        pca.fit(adata.X.A)
        embed = pca.transform(adata.X.A)
        raw_exp = adata.X.A
    else:
        raw_exp = adata.X
        pca.fit(adata.X)
        embed = pca.transform(adata.X)
    feat_pca = embed
    feat_orig = raw_exp
    feat_corr = permutation(feat_orig)
    feat_pca_corr = permutation(feat_pca)
    adata.obsm["feat_orig"] = feat_orig
    adata.obsm["feat_pca_orig"] = feat_pca
    adata.obsm["feat_corr"] = feat_corr
    adata.obsm["feat_pca_corr"] = feat_pca_corr


def get_feature_vae(adata, n_epochs=200, batch_size=128, n_components=50):
    n_input = adata.X.shape[1]
    n_hidden = 256
    n_latent = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(n_input, n_hidden, n_latent).to(device)
    x = torch.Tensor(adata.X).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    vae.train_vae(optimizer, x, n_epochs, batch_size)
    feat_vae, _, _ = vae.forward(x)
    feat_vae = feat_vae.detach().cpu().numpy()

    pca = PCA(n_components=n_components, random_state=1234)
    if issparse(feat_vae):
        pca.fit(feat_vae)
        embed = pca.transform(feat_vae)
        feat_feat_vae = feat_vae.A
    else:
        pca.fit(feat_vae)
        embed = pca.transform(feat_vae)
    adata.obsm["feat_pca_vae"] = embed
    adata.obsm["feat_vae"] = feat_vae


def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_cls = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm["label_cls"] = label_cls


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = (
        adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    )
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@numba.njit("f4(f4[:], f4[:])", nopython=True)
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    return np.sqrt(sum)


@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True, nopython=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49):
    # beta to control the range of neighbourhood when calculate grey vale for one spot
    beta_half = round(beta / 2)
    g = []
    for i in range(len(x_pixel)):
        max_x = image.shape[0]
        max_y = image.shape[1]
        nbs = image[
              max(0, x_pixel[i] - beta_half): min(max_x, x_pixel[i] + beta_half + 1),
              max(0, y_pixel[i] - beta_half): min(max_y, y_pixel[i] + beta_half + 1),
              ]
        g.append(np.mean(np.mean(nbs, axis=0), axis=0))
    c0, c1, c2 = [], [], []
    for i in g:
        c0.append(i[0])
        c1.append(i[1])
        c2.append(i[2])
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (
            np.var(c0) + np.var(c1) + np.var(c2)
    )
    return c3


def calculate_adj_matrix(
        x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=False
):
    # x,y,x_pixel, y_pixel are lists
    X = np.array([x, y]).T.astype(np.float32)
    adj = pairwise_distance(X)
    return adj


def calculate_p(adj, l):
    adj_exp = np.exp(-1 * (adj ** 2) / (2 * (l ** 2)))
    return np.mean(np.sum(adj_exp, 1)) - 1


def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    run = 0
    p_low = calculate_p(adj, start)
    p_high = calculate_p(adj, end)
    if p_low > p + tol:
        print("l not found, try smaller start point.")
        return None
    elif p_high < p - tol:
        print("l not found, try bigger end point.")
        return None
    elif np.abs(p_low - p) <= tol:
        print("recommended l = ", str(start))
        return start
    elif np.abs(p_high - p) <= tol:
        print("recommended l = ", str(end))
        return end
    while (p_low + tol) < p < (p_high - tol):
        run += 1
        print("Run " + str(run) + ": l [" + str(start) + ", " + str(end) + "], p [" + str(p_low) + ", " + str(
            p_high) + "]")
        if run > max_run:
            print("Exact l not found, closest values are:\n" + "l=" + str(start) + ": " + "p=" + str(
                p_low) + "\nl=" + str(end) + ": " + "p=" + str(p_high))
            return None
        mid = (start + end) / 2
        p_mid = calculate_p(adj, mid)
        if np.abs(p_mid - p) <= tol:
            print("recommended l = ", str(mid))
            return mid
        if p_mid <= p:
            start = mid
            p_low = p_mid
        else:
            end = mid
            p_high = p_mid


def mclust_R(
        adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2022
):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    can not use in windows
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


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


def compute_adata_components(adata, n_components=50, norm=False):
    if 'adj_orig' not in adata.obsm.keys():
        construct_interaction_KNN(adata)
        # if need  normalize adj\
        adata.obsm["graph_orig"] = adata.obsm["graph_orig"] + np.eye(adata.shape[0])
        adata.obsm["adj_orig"] = adata.obsm["adj_orig"] + np.eye(adata.shape[0])
        if norm:
            adata.obsm["adj_orig"] = normalize_adj(adata.obsm["adj_orig"])
    if 'label_cls' not in adata.obsm.keys():
        add_contrastive_label(adata)

    if 'feat' not in adata.obsm.keys():
        get_feature(adata, n_components=n_components)

    if 'feat_vae' not in adata.obsm.keys():
        get_feature_vae(adata, n_epochs=100, batch_size=256, n_components=n_components)

    if "adj_aug" not in adata.obsm.keys():
        construct_interaction_cosine(adata)
        # normalize adj
        # adata.obsm["graph_aug"] = adata.obsm["graph_aug"] + np.eye(adata.shape[0])
        adata.obsm["adj_aug"] = adata.obsm["adj_aug"] + np.eye(adata.shape[0])
        if norm:
            adata.obsm["adj_aug"] = normalize_adj(adata.obsm["adj_aug"])


def pseudo_Spatiotemporal_Map(adata, pSM_values_save_filepath="none", n_neighbors=20, resolution=1.0):
    """
    code source from SpaceFlow: Qing Nie
    Perform pseudo-Spatiotemporal Map for ST data
    :param pSM_values_save_filepath: the default save path for the pSM values
    :type pSM_values_save_filepath: class:`str`, optional, default: "./pSM_values.tsv"
    :param n_neighbors: The size of local neighborhood (in terms of number of neighboring data
    points) used for manifold approximation. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html` for detail
    :type n_neighbors: int, optional, default: 20
    :param resolution: A parameter value controlling the coarseness of the clustering.
    Higher values lead to more clusters. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html` for detail
    :type resolution: float, optional, default: 1.0
    """
    error_message = "No embedding found, please ensure you have run train() method before calculating pseudo-Spatiotemporal Map!"
    max_cell_for_subsampling = 5000
    try:
        print("Performing pseudo-Spatiotemporal Map")
        # adata = anndata.AnnData(emb)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='z')
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=resolution)
        sc.tl.paga(adata)
        if adata.shape[0] < max_cell_for_subsampling:
            if adata.X.shape[1] > 3000:
                adata = adata[:, adata.var["highly_variable"]]
                sub_adata_x = adata.X
        else:
            indices = np.arange(adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = adata.X[selected_ind, :]
        sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
        adata.uns['iroot'] = np.argmax(sum_dists)
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata)
        pSM_values = adata.obs['dpt_pseudotime'].to_numpy()
        save_dir = os.path.dirname(pSM_values_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt(pSM_values_save_filepath, pSM_values, fmt='%.5f', header='', footer='', comments='')
        print(
            f"pseudo-Spatiotemporal Map(pSM) calculation complete, pSM values of cells or spots saved at {pSM_values_save_filepath}!")
        pSM_values = pSM_values
    except NameError:
        print(error_message)
    except AttributeError:
        print(error_message)


def plot_pSM(self, pSM_figure_save_filepath="./pseudo-Spatiotemporal-Map.pdf", colormap='roma', scatter_sz=1., rsz=4.,
             csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
    """
    Plot the domain segmentation for ST data in spatial
    :param pSM_figure_save_filepath: the default save path for the figure
    :type pSM_figure_save_filepath: class:`str`, optional, default: "./Spatiotemporal-Map.pdf"
    :param colormap: The colormap to use. See `https://www.fabiocrameri.ch/colourmaps-userguide/` for name list of colormaps
    :type colormap: str, optional, default: roma
    :param scatter_sz: The marker size in points**2
    :type scatter_sz: float, optional, default: 1.0
    :param rsz: row size of the figure in inches, default: 4.0
    :type rsz: float, optional
    :param csz: column size of the figure in inches, default: 4.0
    :type csz: float, optional
    :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type wspace: float, optional
    :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
    :type hspace: float, optional
    :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
    :type left: float, optional
    :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
    :type right: float, optional
    :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
    :type bottom: float, optional
    :param top: the top position of the subplots of the figure in fraction, default: 0.9
    :type top: float, optional
    """
    error_message = "No pseudo Spatiotemporal Map data found, please ensure you have run the pseudo_Spatiotemporal_Map() method."
    try:
        fig, ax = self.prepare_figure(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right,
                                      bottom=bottom, top=top)
        x, y = self.adata_preprocessed.obsm["spatial"][:, 0], self.adata_preprocessed.obsm["spatial"][:, 1]
        st = ax.scatter(x, y, s=scatter_sz, c=self.pSM_values, cmap=f"cmc.{colormap}", marker=".")
        ax.invert_yaxis()
        clb = fig.colorbar(st)
        clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
        ax.set_title("pseudo-Spatiotemporal Map", fontsize=14)
        ax.set_facecolor("none")

        save_dir = os.path.dirname(pSM_figure_save_filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(pSM_figure_save_filepath, dpi=300)
        print(f"Plotting complete, pseudo-Spatiotemporal Map figure saved at {pSM_figure_save_filepath} !")
        plt.close('all')
    except NameError:
        print(error_message)
    except AttributeError:
        print(error_message)


def Moran_I(genes_exp, x, y, k=5, knn=True):
    XYmap = pd.DataFrame({"x": x, "y": y})
    if knn:
        XYnbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(XYmap)
        XYdistances, XYindices = XYnbrs.kneighbors(XYmap)
        W = np.zeros((genes_exp.shape[0], genes_exp.shape[0]))
        for i in range(0, genes_exp.shape[0]):
            W[i, XYindices[i, :]] = 1
        for i in range(0, genes_exp.shape[0]):
            W[i, i] = 0
    else:
        W = calculate_adj_matrix(x=x, y=y, histology=False)
    I = pd.Series(index=genes_exp.columns, dtype="float64")
    for k in genes_exp.columns:
        X_minus_mean = np.array(genes_exp[k] - np.mean(genes_exp[k]))
        X_minus_mean = np.reshape(X_minus_mean, (len(X_minus_mean), 1))
        Nom = np.sum(np.multiply(W, np.matmul(X_minus_mean, X_minus_mean.T)))
        Den = np.sum(np.multiply(X_minus_mean, X_minus_mean))
        I[k] = (len(genes_exp[k]) / np.sum(W)) * (Nom / Den)
    return I


def Geary_C(genes_exp, x, y, k=5, knn=True):
    XYmap = pd.DataFrame({"x": x, "y": y})
    if knn:
        XYnbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(XYmap)
        XYdistances, XYindices = XYnbrs.kneighbors(XYmap)
        W = np.zeros((genes_exp.shape[0], genes_exp.shape[0]))
        for i in range(0, genes_exp.shape[0]):
            W[i, XYindices[i, :]] = 1
        for i in range(0, genes_exp.shape[0]):
            W[i, i] = 0
    else:
        W = calculate_adj_matrix(x=x, y=y, histology=False)
    C = pd.Series(index=genes_exp.columns, dtype="float64")
    for k in genes_exp.columns:
        X = np.array(genes_exp[k])
        X_minus_mean = X - np.mean(X)
        X_minus_mean = np.reshape(X_minus_mean, (len(X_minus_mean), 1))
        Xij = np.array([X, ] * X.shape[0]).transpose() - np.array([X, ] * X.shape[0])
        Nom = np.sum(np.multiply(W, np.multiply(Xij, Xij)))
        Den = np.sum(np.multiply(X_minus_mean, X_minus_mean))
        C[k] = (len(genes_exp[k]) / (2 * np.sum(W))) * (Nom / Den)
    return C


def Cal_Spatial_Net_3D(adata, key_section='Section_id', verbose=True, num=8, p=1):
    """\
    Construct the spatial neighbor networks.
    """
    # guassian kernel
    adj_gaussian = calculate_adj_matrix(adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1], histology=False)
    num_section = np.unique(adata.obs[key_section]).shape[0]
    unique_sections = adata.obs[key_section].unique()
    section_counts = adata.obs[key_section].value_counts()
    # The cumulative sum of each section is computed in advance and used to determine the index position
    cumulative_counts = np.cumsum([section_counts[sec] for sec in unique_sections])
    end_indices = np.append(cumulative_counts, adata.shape[0])
    for it in range(num_section - 2):
        start_col = end_indices[it]
        start_row = end_indices[it + 1]
        adj_gaussian[start_row:, :start_col] = 1e10
        adj_gaussian[:start_col, start_row:] = 1e10

    l = search_l(p, adj_gaussian, start=0.01, end=1000, tol=0.01, max_run=100)
    adj_gaussian = np.exp(-1 * (adj_gaussian ** 2) / (2 * (l ** 2)))
    adata.obsm["combined_adjacency_gaussian"] = adj_gaussian
    print("combined_adjacency_gaussian constructed!")
    # KNN
    euclid_dis, spot_knn = construct_distance_matrix(adata, metric="euclidean")
    for it in range(num_section - 2):
        start_col = end_indices[it]
        start_row = end_indices[it + 1]
        euclid_dis[start_row:, :start_col] = 1e10
        euclid_dis[:start_col, start_row:] = 1e10
        # find k-nearest neighbors
    interaction_knn = np.zeros([spot_knn, spot_knn])
    for i in range(spot_knn):
        vec = euclid_dis[i, :]
        distance = vec.argsort()
        for t in range(1, num + 1):
            y = distance[t]
            interaction_knn[i, y] = 1
    adata.obsm["graph_knn"] = interaction_knn + np.eye(adata.shape[0])
    adj_KNN = interaction_knn
    adj_KNN = adj_KNN + adj_KNN.T
    adj_KNN = np.where(adj_KNN > 1, 1, adj_KNN)
    adj_KNN = preprocess_adj(adj_KNN)
    adata.obsm["combined_adjacency_KNN"] = adj_KNN
    print("combined_adjacency_KNN constructed!")
    # cosine
    cosine_dis, spot_cos = construct_distance_matrix(adata, metric="cosine")
    for it in range(num_section - 2):
        start_col = end_indices[it]
        start_row = end_indices[it + 1]
        cosine_dis[start_row:, :start_col] = 1e10
        cosine_dis[:start_col, start_row:] = 1e10
        # find k-nearest neighbors
    interaction_cos = np.zeros([spot_cos, spot_cos])
    for i in range(spot_knn):
        vec = cosine_dis[i, :]
        distance = vec.argsort()
        for t in range(1, num + 1):
            y = distance[t]
            interaction_cos[i, y] = 1
    adata.obsm["graph_cos"] = interaction_knn
    adj_cosine = interaction_cos
    adj_cosine = adj_cosine + adj_cosine.T
    adj_cosine = np.where(adj_cosine > 1, 1, adj_cosine)
    adj_cosine = preprocess_adj(adj_cosine)
    adata.obsm["combined_adjacency_cosine"] = adj_cosine
    print("combined_adjacency_cosine constructed!")
    print("multi view Combined adjacency matrix constructed!")


# cite from SpaGCN
def count_nbr(target_cluster, cell_id, x, y, pred, radius):
    adj_2d = calculate_adj_matrix(x=x, y=y, histology=False)
    cluster_num = dict()
    df = {'cell_id': cell_id, 'x': x, "y": y, "pred": pred}
    df = pd.DataFrame(data=df)
    df.index = df['cell_id']
    target_df = df[df["pred"] == target_cluster]
    row_index = 0
    num_nbr = []
    for index, row in target_df.iterrows():
        x = row["x"]
        y = row["y"]
        tmp_nbr = df[((df["x"] - x) ** 2 + (df["y"] - y) ** 2) <= (radius ** 2)]
        num_nbr.append(tmp_nbr.shape[0])
    return np.mean(num_nbr)


def search_radius(target_cluster, cell_id, x, y, pred, start, end, num_min=8, num_max=15, max_run=100):
    run = 0
    num_low = count_nbr(target_cluster, cell_id, x, y, pred, start)
    num_high = count_nbr(target_cluster, cell_id, x, y, pred, end)
    if num_min <= num_low <= num_max:
        print("recommended radius = ", str(start))
        return start
    elif num_min <= num_high <= num_max:
        print("recommended radius = ", str(end))
        return end
    elif num_low > num_max:
        print("Try smaller start.")
        return None
    elif num_high < num_min:
        print("Try bigger end.")
        return None
    while (num_low < num_min) and (num_high > num_min):
        run += 1
        print("Run " + str(run) + ": radius [" + str(start) + ", " + str(end) + "], num_nbr [" + str(
            num_low) + ", " + str(num_high) + "]")
        if run > max_run:
            print("Exact radius not found, closest values are:\n" + "radius=" + str(start) + ": " + "num_nbr=" + str(
                num_low) + "\nradius=" + str(end) + ": " + "num_nbr=" + str(num_high))
            return None
        mid = (start + end) / 2
        num_mid = count_nbr(target_cluster, cell_id, x, y, pred, mid)
        if num_min <= num_mid <= num_max:
            print("recommended radius = ", str(mid), "num_nbr=" + str(num_mid))
            return mid
        if num_mid < num_min:
            start = mid
            num_low = num_mid
        elif num_mid > num_max:
            end = mid
            num_high = num_mid


def find_neighbor_clusters(target_cluster, cell_id, x, y, pred, radius, ratio=1 / 2):
    cluster_num = dict()
    for i in pred:
        cluster_num[i] = cluster_num.get(i, 0) + 1
    df = {'cell_id': cell_id, 'x': x, "y": y, "pred": pred}
    df = pd.DataFrame(data=df)
    df.index = df['cell_id']
    target_df = df[df["pred"] == target_cluster]
    nbr_num = {}
    row_index = 0
    num_nbr = []
    for index, row in target_df.iterrows():
        x = row["x"]
        y = row["y"]
        tmp_nbr = df[((df["x"] - x) ** 2 + (df["y"] - y) ** 2) <= (radius ** 2)]
        # tmp_nbr=df[(df["x"]<x+radius) & (df["x"]>x-radius) & (df["y"]<y+radius) & (df["y"]>y-radius)]
        num_nbr.append(tmp_nbr.shape[0])
        for p in tmp_nbr["pred"]:
            nbr_num[p] = nbr_num.get(p, 0) + 1
    del nbr_num[target_cluster]
    nbr_num = [(k, v) for k, v in nbr_num.items() if v > (ratio * cluster_num[k])]
    nbr_num.sort(key=lambda x: -x[1])
    print("radius=", radius, "average number of neighbors for each spot is", np.mean(num_nbr))
    print(" Cluster", target_cluster, "has neighbors:")
    for t in nbr_num:
        print("Dmain ", t[0], ": ", t[1])
    ret = [t[0] for t in nbr_num]
    if len(ret) == 0:
        print("No neighbor domain found, try bigger radius or smaller ratio.")
    else:
        return ret


def rank_genes_groups(input_adata, target_cluster, nbr_list, label_col, adj_nbr=True, log=False):
    if adj_nbr:
        nbr_list = nbr_list + [target_cluster]
        adata = input_adata[input_adata.obs[label_col].isin(nbr_list)]
    else:
        adata = input_adata.copy()
    adata.var_names_make_unique()
    adata.obs["target"] = ((adata.obs[label_col] == target_cluster) * 1).astype('category')
    sc.tl.rank_genes_groups(adata, groupby="target", reference="rest", n_genes=adata.shape[1], method='wilcoxon')
    pvals_adj = [i[1] for i in adata.uns['rank_genes_groups']["pvals_adj"]]
    genes = [i[1] for i in adata.uns['rank_genes_groups']["names"]]
    if issparse(adata.X):
        obs_tidy = pd.DataFrame(adata.X.A)
    else:
        obs_tidy = pd.DataFrame(adata.X)
    obs_tidy.index = adata.obs["target"].tolist()
    obs_tidy.columns = adata.var.index.tolist()
    obs_tidy = obs_tidy.loc[:, genes]
    # 1. compute mean value
    mean_obs = obs_tidy.groupby(level=0).mean()
    # 2. compute fraction of cells having value >0
    obs_bool = obs_tidy.astype(bool)
    fraction_obs = obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()
    # compute fold change.
    fold_change = [np.power(2, i[1]) for i in adata.uns['rank_genes_groups']["logfoldchanges"]]
    df = {'genes': genes, 'in_group_fraction': fraction_obs.loc[1].tolist(),
          "out_group_fraction": fraction_obs.loc[0].tolist(),
          "in_out_group_ratio": (fraction_obs.loc[1] / fraction_obs.loc[0]).tolist(),
          "in_group_mean_exp": mean_obs.loc[1].tolist(), "out_group_mean_exp": mean_obs.loc[0].tolist(),
          "fold_change": fold_change, "pvals_adj": pvals_adj}
    df = pd.DataFrame(data=df)
    return df


def find_meta_gene(input_adata,
                   pred,
                   target_domain,
                   start_gene,
                   mean_diff=0,
                   early_stop=True,
                   max_iter=5,
                   use_raw=False):
    meta_name = start_gene
    adata = input_adata.copy()
    adata.obs["meta"] = adata.X[:, adata.var.index == start_gene]
    adata.obs["pred"] = pred
    num_non_target = adata.shape[0]
    for i in range(max_iter):
        # Select cells
        tmp = adata[((adata.obs["meta"] > np.mean(adata.obs[adata.obs["pred"] == target_domain]["meta"])) | (
                adata.obs["pred"] == target_domain))]
        tmp.obs["target"] = ((tmp.obs["pred"] == target_domain) * 1).astype('category').copy()
        if (len(set(tmp.obs["target"])) < 2) or (np.min(tmp.obs["target"].value_counts().values) < 5):
            print("Meta gene is: ", meta_name)
            return meta_name, adata.obs["meta"].tolist()
        # DE
        sc.tl.rank_genes_groups(tmp, groupby="target", reference="rest", n_genes=1, method='wilcoxon')
        adj_g = tmp.uns['rank_genes_groups']["names"][0][0]
        add_g = tmp.uns['rank_genes_groups']["names"][0][1]
        meta_name_cur = meta_name + "+" + add_g + "-" + adj_g
        print("Add gene: ", add_g)
        print("Minus gene: ", adj_g)
        # Meta gene
        adata.obs[add_g] = adata.X[:, adata.var.index == add_g]
        adata.obs[adj_g] = adata.X[:, adata.var.index == adj_g]
        adata.obs["meta_cur"] = (adata.obs["meta"] + adata.obs[add_g] - adata.obs[adj_g])
        adata.obs["meta_cur"] = adata.obs["meta_cur"] - np.min(adata.obs["meta_cur"])
        mean_diff_cur = np.mean(adata.obs["meta_cur"][adata.obs["pred"] == target_domain]) - np.mean(
            adata.obs["meta_cur"][adata.obs["pred"] != target_domain])
        num_non_target_cur = np.sum(tmp.obs["target"] == 0)
        if (early_stop == False) | ((num_non_target >= num_non_target_cur) & (mean_diff <= mean_diff_cur)):
            num_non_target = num_non_target_cur
            mean_diff = mean_diff_cur
            print("Absolute mean change:", mean_diff)
            print("Number of non-target spots reduced to:", num_non_target)
        else:
            print("Stopped!", "Previous Number of non-target spots", num_non_target, num_non_target_cur, mean_diff,
                  mean_diff_cur)
            print("Previous Number of non-target spots", num_non_target, num_non_target_cur, mean_diff, mean_diff_cur)
            print("Previous Number of non-target spots", num_non_target)
            print("Current Number of non-target spots", num_non_target_cur)
            print("Absolute mean change", mean_diff)
            print("===========================================================================")
            print("Meta gene: ", meta_name)
            print("===========================================================================")
            return meta_name, adata.obs["meta"].tolist()
        meta_name = meta_name_cur
        adata.obs["meta"] = adata.obs["meta_cur"]
        print("===========================================================================")
        print("Meta gene is: ", meta_name)
        print("===========================================================================")
    return meta_name, adata.obs["meta"].tolist()
