
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
recommend conda to install


```bash
git clone https://github.com/yiwen-yang/MCGAE
cd MCGAE/
conda env create -f environment.yml
conda activate mcgae_env
```
```python
from MCGAE.model import MCGAE
```