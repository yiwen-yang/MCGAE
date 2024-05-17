<h1><center>MCGAE Tutorial</center></h1>



### 1. Installation
The installation should take a few minutes on a normal computer. To install MCGAE package you must make sure that your python version is over 3.9 If you don’t know the version of python you can check it by:


```python
import platform
platform.python_version()
```




  '3.9.16'




Note: Because MCGAE pends on pytorch, you should make sure torch is correctly installed.
<br>
Now you can install the current release of MCGAE by the following way:


#### 1.1 Github
Download the package from Github and install it locally:


```python
git clone https://github.com/yiwen-yang/MCGAE
cd MCGAE/
```
### 2. Import modules

```python
import os
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
import itertools
# Importing custom modules
from MCGAE.model import MCGAE
from MCGAE.utils import load_dataset, norm_and_filter, compute_adata_components, search_res, refine_label, set_seed

# Suppressing runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
```


### 3. Read in data
The current version of MCGAE requres three input data.
1. The gene expression matrix(n by k): expression_matrix.h5;
2. Spatial coordinateds of samplespositions.txt;
3. Histology image(optional): histology.tif, can be tif or png or jepg.

The gene expreesion data can be stored as an AnnData object. AnnData stores a data matrix .X together with annotations of observations .obs, variables .var and unstructured annotations .uns. 


```python
# reading and preprocessing
BASE_DIR = r"D:\Work\MCGAE project\MCGAE-master"
file_path = os.path.join(BASE_DIR, "benchmark", "Colorectal Cancer Liver")
dir_path = os.path.join(file_path, "raw_data")
adata = load_dataset(dir_path, use_image=True)
adata = norm_and_filter(adata)
n_clusters = 20
print(n_clusters)
set_seed(1234)
compute_adata_components(adata, n_components=100)
save_obj_z = pd.DataFrame()
```

### 4. Integrate gene expression and histology into a Graph


```python
model = MCGAE(
    adata,
    n_latent=50,
    n_components=100,
    use_pca=True,
    fusion_mode="fractional",
    use_emb_x_rec=True,
    use_emb_g_rec=True,
    dropout=0.01,#0.01
    random_seed=2,
    w_morph=0.9,
)


model.train(
    weight_decay=5e-4,
    # weight_decay=0.0,
    w_recon_x=0.5,
    w_recon_g=0.1,
    w_contrast=0.1,
    w_cluster=1, # because of ablation,importance of KL loss
    n_clusters=n_clusters,
    cl_start_epoch=100,
    compute_g_loss="cross_entropy", 
)
temp = model.get_model_output()
emb, y_pred = temp["emb"], temp["y_pred"]
adata.obsm["z"] = emb
adata.obs["pred"] = y_pred
res = search_res(adata, n_clusters, rep="z", start=0.3, end=3, increment=0.02)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10, random_state=1234)
sc.tl.leiden(adata, key_added="leiden", resolution=res, random_state=1234)
new_type = refine_label(adata, key='leiden', radius=30)
adata.obs['leiden'] = new_type
sc.pl.spatial(adata, img_key="hires", color='leiden',
              show=False, title="MCGAE")
plt.tight_layout()
```
