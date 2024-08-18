import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from typing import Optional, Literal, Dict
from .layers import GraphConvolution, GCN, MLPDecoder
from .layers import Attention, Discriminator, AvgReadout
from .utils import *
from sklearn.metrics import adjusted_rand_score as ari_score


class MGAE(nn.Module):
    """
    Multi-view graph autoencoder model.

    Parameters
    ----------
    n_input : int
        Number of input genes.
    n_hidden : int, optional
        Number of nodes per hidden layer.
    n_latent : int, optional
        Dimensionality of the latent space.
    dropout : float, optional
        Dropout rate for regularization.
    use_pca : bool, optional
        Whether to use PCA for dimensionality reduction before encoding.
    fusion_mode : str, optional
        Fusion mode for combining multiple views.
    alpha : float, optional
        Alpha is the degree of freedom of the Student’s t-distribution.
    """

    def __init__(
            self,
            n_input: int,
            n_hidden: int = 128,
            n_latent: int = 10,
            n_components: int = 50,
            dropout: float = 0.1,
            use_pca: bool = False,
            fusion_mode: str = "holistic",
            use_emb_x_rec: bool = True,
            use_emb_g_rec: bool = True,
            w_morph: float = 0.0,
            alpha: float = 1.0,
    ):
        super(MGAE, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.use_pca = use_pca
        self.fusion_mode = fusion_mode
        self.use_emb_x_rec = use_emb_x_rec
        self.use_emb_g_rec = use_emb_g_rec
        self.alpha = alpha
        self.mu = None
        self.w_morph = w_morph

        self.gc_orig = GCN(n_input, n_hidden, n_latent, dropout=dropout)
        self.gc_x = GCN(n_input, n_hidden, n_latent, dropout=dropout)
        self.gc_g = GCN(n_input, n_hidden, n_latent, dropout=dropout)
        self.gc_pca_orig = GraphConvolution(n_components, n_latent)
        self.gc_pca_x = GraphConvolution(n_components, n_latent)
        self.gc_pca_g = GraphConvolution(n_components, n_latent)

        self.att1 = Attention(n_latent)
        self.att2 = Attention(n_latent)
        self.att3 = Attention(n_latent)
        self.decoder = MLPDecoder(n_latent, n_hidden, n_input, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(n_components, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_latent))
        self.disc = Discriminator(n_latent)
        self.avg_read = AvgReadout()
        self.sigmoid = nn.Sigmoid()

        self.recon_x_loss = nn.MSELoss()
        self.contrast_loss = nn.BCEWithLogitsLoss()

    def recon_g_loss(self, emb, adj, compute_g_loss="cosine"):
        # Compute the outer product of the embedding
        emb_product = torch.matmul(emb, emb.T)
        emb_product = torch.sigmoid(emb_product)  # or another suitable function
        # Zero out the diagonal elem
        emb_product = emb_product * (1 - torch.eye(emb_product.shape[0], device=emb_product.device)) # test
        adj = adj * (1 - torch.eye(adj.shape[0], device=adj.device))
        # Flatten the matrices
        emb_flat = emb_product.view(-1)
        adj_flat = adj.reshape(-1)
        # Compute cosine similarity and return the loss
        if compute_g_loss=="cosine":
            cosine_similarity = F.cosine_similarity(emb_flat, adj_flat, dim=0)
            loss = 1 - cosine_similarity
        elif compute_g_loss=="cross_entropy":
            loss = F.binary_cross_entropy(emb_flat, adj_flat, reduction='mean')

        return loss

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    def compute_loss(
            self,
            model_output: Dict[str, torch.Tensor],
            x: torch.Tensor,
            adj: torch.Tensor,
            label_cls: torch.Tensor,
            target_p: torch.Tensor,
            cl_flag: float = 1.0,
            compute_g_loss: Literal["cosine", "cross_entropy"] = "cosine",
            w_recon_x: float = 1.0,
            w_recon_g: float = 1.0,
            w_contrast: float = 1.0,
            w_cluster: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute and return the model loss.

        Parameters
        ----------
        model_output : dict
            Model output containing various intermediate results and prediction outputs.
        x : torch.Tensor
            Input data feature vectors of shape (n_spot, n_emb).
        adj : torch.Tensor
            Adjacency matrix of shape (n_spot, n_spot).
        label_cls : torch.Tensor
            Cell type labels of shape (n_spot,).
        target_p : torch.Tensor
            Target distribution P(z) of shape (n_spot, n_emb).
        compute_g_loss : str, optional
            Method to compute the graph reconstruction loss, default is "cosine".
        cl_flag : float, optional
            Flag to compute the clustering loss, default is 1.0.
        w_recon_x : float, optional
            Weight for the feature reconstruction loss, default is 1.0.
        w_recon_g : float, optional
            Weight for the graph reconstruction loss, default is 1.0.
        w_contrast : float, optional
            Weight for the contrastive loss, default is 1.0.
        w_cluster : float, optional
            Weight for the clustering loss, default is 1.0.

        Returns
        -------
        dict
            Dictionary containing various loss components and the total loss.
        """
        # Extract necessary components from model output
        x_rec, emb_g_rec, ret_orig, ret_corr, soft_q = (
            model_output[key] for key in ["x_rec", "emb_g_rec", "ret_orig", "ret_corr", "q"]
        )

        # Compute individual loss components
        recon_x_loss = self.recon_x_loss(x_rec, x)
        recon_g_loss = self.recon_g_loss(emb_g_rec, adj, compute_g_loss=compute_g_loss)
        contrast_loss = self.contrast_loss(ret_orig, label_cls) + self.contrast_loss(ret_corr, label_cls)
        cluster_loss = self.cluster_loss(target_p, soft_q)

        # Compute total loss
        total_loss = (
                w_recon_x * recon_x_loss +
                w_recon_g * recon_g_loss +
                w_contrast * contrast_loss +
                cl_flag * w_cluster * cluster_loss
        )

        # Return loss components and total loss in a dictionary
        return {
            'loss': total_loss,
            'recon_x_loss': recon_x_loss,
            'recon_g_loss': recon_g_loss,
            'contrast_loss': contrast_loss,
            'cluster_loss': cluster_loss,
        }

    def forward(self, input_tensors: Dict[str, torch.Tensor], compute_q: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Parameters
        ----------
        input_tensors : dict
            A dictionary containing various input tensors for the model:
            - 'feat_orig': torch.Tensor
                The original feature matrix.
            - 'feat_corr': torch.Tensor
                The corrupted feature matrix.
            - 'feat_vae': torch.Tensor
                The augmented feature matrix.
            - 'feat_pca_orig': torch.Tensor
                The PCA-transformed original feature matrix.
            - 'feat_pca_corr': torch.Tensor
                The PCA-transformed corrupted feature matrix.
            - 'feat_pca_vae': torch.Tensor
                The PCA-transformed augmented feature matrix.
            - 'adj_orig': torch.Tensor
                The processed symmetric adjacency matrix for k-nearest neighbor graph.
            - 'adj_aug': torch.Tensor
                The processed symmetric adjacency matrix based on augmented similarity matrix.
            - 'graph_orig': torch.Tensor
                The original asymmetric adjacency matrix for k-nearest neighbor graph.
            - 'label_cls': torch.Tensor
                The cell type labels.
        compute_q : bool, optional
            Whether to compute q for clustering loss. Default is False.

        Returns
        -------
        dict
            Dictionary containing various intermediate results and outputs of the model.
            - 'emb': Common z for downstream tasks.
            - 'emb_x_rec': emb_x_rec for expression reconstruction.
            - 'emb_g_rec': emb_g_rec for graph reconstruction.
            - 'x_rec': Reconstructed feature matrix for expression.
            - 'q': q soft cluster assignment for clustering loss.
            - 'ret_orig': ret_orig for contrastive loss.
            - 'ret_corr': ret_corr for contrastive loss.
        """
        # Extracting tensors from the input dictionary
        feat_orig = input_tensors['feat_orig']
        feat_corr = input_tensors['feat_corr']
        feat_vae = input_tensors['feat_vae']
        feat_pca_orig = input_tensors['feat_pca_orig']
        feat_pca_corr = input_tensors['feat_pca_corr']
        feat_pca_vae = input_tensors['feat_pca_vae']
        adj_orig = input_tensors['adj_orig']
        adj_aug = input_tensors['adj_aug']
        graph_orig = input_tensors['graph_orig']
        if "X_morphology" in input_tensors.keys():
            morphology = input_tensors["X_morphology"]
            morphology = self.mlp(morphology)

        # label_cls = input_tensors['label_cls']

        if self.use_pca:
            emb_orig = self.gc_pca_orig(feat_pca_orig, adj_orig)
            emb_corr = self.gc_pca_orig(feat_pca_corr, adj_orig)
            emb_aug_x = self.gc_pca_x(feat_pca_vae, adj_orig)
            emb_aug_g = self.gc_pca_g(feat_pca_orig, adj_aug)
        else:
            emb_orig = self.gc_orig(feat_orig, adj_orig)
            emb_corr = self.gc_orig(feat_corr, adj_orig)
            emb_aug_x = self.gc_x(feat_vae, adj_orig)
            emb_aug_g = self.gc_g(feat_orig, adj_aug)
        g_orig = self.avg_read(emb_orig, graph_orig)
        g_orig = self.sigmoid(g_orig)
        g_corr = self.avg_read(emb_corr, graph_orig)
        g_corr = self.sigmoid(g_corr)
        ret_orig = self.disc(g_orig, emb_orig, emb_corr)
        ret_corr = self.disc(g_corr, emb_corr, emb_orig)

        q = None
        if self.fusion_mode == "holistic":
            emb_stack = torch.stack([emb_orig, emb_aug_x, emb_aug_g], dim=1)
            emb, _ = self.att1(emb_stack)
            emb_x_rec = emb
            emb_g_rec = emb
        elif self.fusion_mode == "fractional":
            emb_stack = torch.stack([emb_orig, emb_aug_g], dim=1)
            emb_x_rec, _ = self.att1(emb_stack)
            emb_stack = torch.stack([emb_orig, emb_aug_x], dim=1)
            emb_g_rec, _ = self.att2(emb_stack)
            emb = torch.stack([emb_x_rec, emb_g_rec], dim=1)
            if self.use_emb_x_rec and self.use_emb_g_rec:
                emb, _ = self.att3(emb)
            elif not self.use_emb_x_rec and self.use_emb_g_rec:
                emb = emb_g_rec
            elif self.use_emb_x_rec and not self.use_emb_g_rec:
                emb = emb_x_rec
            else:
                raise ValueError("At least one of use_emb_x_rec and use_emb_g_rec must be True.")
        elif self.fusion_mode == "vanilla":
            emb = feat_pca_vae  # or whichever embedding you want to use
            emb_x_rec = emb
            emb_g_rec = emb
        else:
            raise ValueError(
                f"Invalid fusion_mode: {self.fusion_mode}. Expected 'holistic', 'fractional', or 'vanilla'.")

        x_rec = self.decoder(emb_x_rec)
        if "X_morphology" in input_tensors.keys():
            emb = emb + self.w_morph * morphology
        if compute_q:
            q = 1.0 / (1.0 + torch.sum((emb.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
            # q = q.pow((self.alpha + 1.0) / 2.0)
            q = q ** (self.alpha + 1.0) / 2.0
            q = q / torch.sum(q, dim=1, keepdim=True)
        return dict(
            emb=emb,
            emb_x_rec=emb_x_rec,
            emb_g_rec=emb_g_rec,
            x_rec=x_rec,
            q=q,
            ret_orig=ret_orig,
            ret_corr=ret_corr,
        )

    def compute_mu(self, emb, cluster_method, n_clusters, res=0.4, if_search_res=True):
        y_pred = None
        if cluster_method == "kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            kmeans = KMeans(n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(emb.detach().cpu().numpy())
        elif cluster_method == "leiden":
            adata_clu = sc.AnnData(emb.detach().cpu().numpy())
            if if_search_res:
                res = search_res(adata_clu, n_clusters)
            print("Initializing cluster centers with leiden, resolution = ", res)
            sc.pp.neighbors(adata_clu, n_neighbors=10)
            sc.tl.leiden(adata_clu, resolution=res)
            y_pred = adata_clu.obs["leiden"].astype(int).to_numpy()
            n_clusters = len(np.unique(y_pred))

        mu = torch.nn.Parameter(torch.Tensor(n_clusters, self.n_latent)).to(emb.device)
        emb = pd.DataFrame(emb.detach().cpu().numpy(), index=np.arange(0, emb.shape[0]))
        group = pd.Series(y_pred, index=np.arange(0, emb.shape[0]), name="group")
        merge_embed = pd.concat([emb, group], axis=1)
        cluster_centers = np.asarray(merge_embed.groupby("group").mean())
        mu.data.copy_(torch.Tensor(cluster_centers))
        return mu, y_pred
