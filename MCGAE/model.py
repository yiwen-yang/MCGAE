"""
MCGAE: A Versatile Model for Spatial Transcriptomics Data Analysis.
(Multi-view Contrastive Graph AutoEncoder)

Developers
----------
- Developer 1: [Yiwen Yang] <yangyw1@shanghaitech.edu.cn>
- Developer 2: [Chengming Zhang] <zhangchengming@g.ecc.u-tokyo.ac.jp>
- ...

License
-------
None

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import issparse
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch.optim as optim
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from typing import Optional, Literal
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score as ari_score
import matplotlib.pyplot as plt

from .layers import GraphConvolution as gc
from .layers import *
from .module import MGAE
from sklearn.decomposition import PCA
from .utils import *


class MCGAE(nn.Module):
    """
    MCGAE is used to model spatial transcriptomics data.

    **Acknowledgments**
    This code is inspired by the python library of scvi-tools.
    Gayoso, A., Lopez, R., Xing, G. et al. A Python library for probabilistic analysis of single-cell omics data.
    Nat Biotechnol 40, 163–166 (2022). https://doi.org/10.1038/s41587-021-01206-w.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing the spatial data.
    n_latent : int, optional
        Dimensionality of the latent space.
    n_components : int, optional
        Number of components for PCA dimensionality reduction.
    use_pca : bool, optional
        Whether to use PCA for dimensionality reduction before encoding.
fusion_mode : Literal["holistic", "fractional", "vanilla"], optional
    The strategy for fusing the latent spaces (z) obtained from the encoder.
    - "holistic": all z's are fused into a single z, which is then used to reconstruct both x and A.
    - "fractional": different z's are fused in pairs to reconstruct x and A separately, and then the resulting z's are fused again.
    - "vanilla": only a single z is used for all operations, without considering other latent spaces.
    random_seed : int, optional
        A seed for the random number generator to ensure reproducibility.
    **model_kwargs : dict
        Additional keyword arguments for the :class:`~MCGAE.module`.
    """
    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 50,
            n_components: int = 50,
            use_pca: bool = True,
            fusion_mode: Literal["holistic", "fractional", "vanilla"] = "holistic",
            random_seed: int = 42,
            w_morph: float = 0,
            **model_kwargs,
    ):
        super(MCGAE, self).__init__()
        set_seed(random_seed)
        self.adata = adata
        self.n_spots = adata.n_obs
        self.n_latent = n_latent
        valid_modes = ["holistic", "fractional", "vanilla"]
        if fusion_mode not in valid_modes:
            raise ValueError(f"Invalid fusion_mode. Expected one of {valid_modes}.")
        self.fusion_mode = fusion_mode
        self.w_morph = w_morph
        self.device = None
        self.loss_history = None

        # model initialization
        self.module = MGAE(
            n_input=self.adata.shape[1],
            n_latent=self.n_latent,
            n_components=n_components,
            use_pca=use_pca,
            fusion_mode=self.fusion_mode,
            w_morph=self.w_morph,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: int = 400,
            lr: float = 1e-3,
            device: Optional[str] = None,
            opt: Literal["Adam", "SGD", "RMSprop"] = "Adam",
            weight_decay: float = 5e-4,
            cluster_method: Literal["leiden", "kmeans"] = "leiden",
            cl_start_epoch: int = 0,  # New parameter
            p_update_interval: int = 5,
            y_predict_interval: int = 5,
            loss_print_interval: int = 20,
            tol: float = 1e-6,
            res: float = 0.4,
            n_clusters: int = 4,
            adj_diag: float = 0.0,
            **kwargs,
    ):
        """
         Trains the model using a graph neural network.

         Parameters
         ----------
         max_epochs : int, optional
             Number of passes through the dataset.
         lr : float
             Learning rate for optimization.
         device : str, optional
             Which device to use for computation. Options are 'cpu', 'cuda', or None.
             If None, will use GPU if available, otherwise CPU.
         opt : {'Adam', 'SGD', 'RMSprop'}, optional
             Optimizer for model training.
         weight_decay : float, optional
             Weight decay (L2 penalty).
         cluster_method : {'leiden', 'kmeans'}, optional
             Method for clustering.
         cl_start_epoch : int, optional
             Epoch to start applying clustering loss.
         p_update_interval : int, optional
             Interval for updating the target distribution P(z).
         y_predict_interval : int, optional
             Interval for predicting cluster labels.
         loss_print_interval : int, optional
             Interval for printing the loss values.
         tol : float, optional
             Tolerance for early stopping.
         res : float, optional
             Resolution for louvain clustering.
         n_clusters : int, optional
             Number of clusters to generate.
         adj_diag : float, optional
                Value to add to the diagonal of the adjacency matrix.
         **kwargs
             Other keyword arguments for specific training methods.
         """
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this system.")
        device = torch.device(device)
        self.device = device
        if device.type == "cuda":
            self.module.cuda(device=device)

        # Move data to device and optionally add identity matrix
        keys = ["feat_orig", "feat_corr", "feat_vae", "feat_pca_orig", "feat_pca_corr", "feat_pca_vae",
                "adj_orig", "adj_aug", "graph_orig", "label_cls"]
        input_tensors = {key: torch.FloatTensor(self.adata.obsm[key]).to(self.device) for key in keys}
        if "X_morphology" in self.adata.obsm.keys():
            input_tensors["X_morphology"] = torch.FloatTensor(self.adata.obsm["X_morphology"]).to(self.device)
        input_tensors["adj_orig"] += torch.eye(self.n_spots, device=self.device) * (adj_diag - 1)
        optimizer = None
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # 对角线替换
        model_output = self.module(input_tensors, compute_q=False)
        emb = model_output['emb']
        mu, y_pred = self.module.compute_mu(emb, cluster_method, n_clusters, res)
        self.module.mu = mu
        y_pred_list = []
        y_pred_last = y_pred
        loss_history = {'loss': [], 'recon_x_loss': [], 'recon_g_loss': [], 'contrast_loss': [], 'cluster_loss': []}
        self.module.train()
        p = None
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            # 1. Model Forward Pass
            model_output = self.module(input_tensors, compute_q=True)
            q = model_output['q']

            # 2. Special Handling for First Epoch & Logging and Metrics Calculation
            if epoch == 0 or epoch % 20 == 0:
                p = self.module.target_distribution(q).data
                # if epoch % 20 == 0:
                #     y = self.adata.obs["label"]
                #     y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                #     ari = ari_score(y, y_pred)
                #     print(f"\nCurrent epoch is: {epoch}, and ARI is: {ari}")

            # 3. Conditional Logic for Cluster Loss Start Epoch
            if epoch >= cl_start_epoch and (epoch - cl_start_epoch) % p_update_interval == 0:
                p = self.module.target_distribution(q).data

            # 4. Loss Calculation and Backpropagation
            cl_flag = 1.0 if epoch >= cl_start_epoch else 0.0
            adj = input_tensors['adj_orig']
            feat_orig, label_cls = input_tensors['feat_orig'], input_tensors['label_cls']
            loss_dict = self.module.compute_loss(model_output, feat_orig, adj, label_cls, p, **kwargs)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optional: Print loss at specified intervals
            if epoch % loss_print_interval == 0 or epoch == (max_epochs - 1):
                tqdm.write(f"Epoch: {epoch}, Loss: {loss}")

            # 5. Loss History Logging
            loss_history['loss'].append(loss.item())
            loss_history['recon_x_loss'].append(loss_dict['recon_x_loss'].item())
            loss_history['recon_g_loss'].append(loss_dict['recon_g_loss'].item())
            loss_history['contrast_loss'].append(loss_dict['contrast_loss'].item())
            loss_history['cluster_loss'].append(loss_dict['cluster_loss'].item())
            self.loss_history = loss_history

            # 6. Convergence Check
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            y_pred_list.append(y_pred)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / len(y_pred_last)
            y_pred_last = y_pred

            if epoch > 400 and delta_label < tol:
                print(f'delta_label {delta_label} < tol {tol}')
                print("Reach tolerance threshold. Stopping training.")
                print(f"Total epoch: {epoch}")
                break

    @torch.no_grad()
    def get_model_output(self, adata: Optional[AnnData] = None):
        """
        Return model output.

        Parameters:
        - adata: Optional, an AnnData object. If None, self.adata will be used.

        Returns:
        - output: A dictionary containing various model outputs.
        """
        # Ensure the model is in evaluation mode
        if self.module.training:
            self.module.eval()

        # Use provided adata or default to self.adata
        adata_used = adata if adata is not None else self.adata

        # Move data to device and optionally add identity matrix
        keys = ["feat_orig", "feat_corr", "feat_vae", "feat_pca_orig", "feat_pca_corr", "feat_pca_vae",
                "adj_orig", "adj_aug", "graph_orig", "label_cls"]
        input_tensors = {key: torch.FloatTensor(adata_used.obsm[key]).to(self.device) for key in keys}
        input_tensors["graph_orig"] += torch.eye(self.n_spots, device=self.device)
        if "X_morphology" in self.adata.obsm.keys():
            input_tensors["X_morphology"] = torch.FloatTensor(self.adata.obsm["X_morphology"]).to(self.device)

        # Get model output
        model_output = self.module(input_tensors, compute_q=True)

        # Extract and process necessary outputs
        q = model_output['q']
        y_pred = torch.argmax(q, dim=1).cpu().numpy()

        # Construct output dictionary ensuring tensors are moved to CPU and converted to NumPy arrays
        output = {
            'emb': model_output['emb'].cpu().numpy(),
            'emb_x_rec': model_output['emb_x_rec'].cpu().numpy(),
            'emb_g_rec': model_output['emb_g_rec'].cpu().numpy(),
            'x_rec': model_output['x_rec'].cpu().numpy(),
            'y_pred': y_pred
        }

        return output

    def plot_train_loss(self, fig_size=(8, 8), save_path=None, fig_show=True, pause_time=10):
        # Set figure size
        fig = plt.figure(figsize=fig_size)
        if self.loss_history is None:
            raise ValueError("You should train the model first!")
        epoch_losses = self.loss_history
        # Plot a subplot of each loss
        for i, loss_name in enumerate(epoch_losses.keys()):
            # Gets the value of the current loss
            loss_values = epoch_losses[loss_name]
            # Create subplot
            ax = fig.add_subplot(3, 2, i + 1)
            # Draw subplot
            ax.plot(range(len(loss_values)), loss_values)
            # Set the subplot title
            ax.set_title(loss_name)
            # Set the subplot x-axis and y-axis labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
        # adjust the distance and edges between sub-graphs
        plt.tight_layout()
        # Save the figure if a save path is provided
        if save_path is not None:
            plt.savefig(save_path)
        # Show the figure without blocking if show is True
        if fig_show:
            plt.draw()
            plt.pause(pause_time)
            plt.close(fig)  # Close the figure window