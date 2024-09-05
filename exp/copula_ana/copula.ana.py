import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..')
sys.path.append(_parent_path)

import pandas as pd
import torch    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import t, gaussian_kde
from scipy.interpolate import interp1d
import yaml

from multicopula_model import EllipticalCopula
from tools.evaluation_m import MMD_kernel, calculate_w_distances
import alg.models_fcpflow_lin as fcpf
import tools.tools_train as tl

def compute_empirical_marginal_cdf(a):
    sorted_dim1 = np.sort(a[:, 0])
    sorted_dim2 = np.sort(a[:, 1])

    cdf_dim1 = np.arange(1, len(sorted_dim1) + 1) / len(sorted_dim1)
    cdf_dim2 = np.arange(1, len(sorted_dim2) + 1) / len(sorted_dim2)

    cdf_func_dim1 = interp1d(sorted_dim1, cdf_dim1, kind='linear', bounds_error=False, fill_value=(0, 1))
    cdf_func_dim2 = interp1d(sorted_dim2, cdf_dim2, kind='linear', bounds_error=False, fill_value=(0, 1))

    return cdf_func_dim1, cdf_func_dim2

# define the data path and read 
path = os.path.join(_parent_path, 'data/nl_data_1household.csv')
data = pd.read_csv(path).iloc[:,3:-2].values
data = data[~pd.isna(data).any(axis=1)]
pca_data = PCA(n_components=2).fit_transform(data)
print(pca_data.shape)

# define the copula model and sample
copula = EllipticalCopula(pca_data.T)
copula.fit()
samples = copula.sample(pca_data.shape[0])

# define the fcpflow model and sample
with open(os.path.join(_parent_path,'exp/copula_ana/FCPFlow/config.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
model = fcpf.FCPflow(config['FCPflow']['num_blocks'], config['FCPflow']['num_channels'], 
                        config['FCPflow']['sfactor'], config['FCPflow']['hidden_dim'], config['FCPflow']['condition_dim'])
_, scaler = tl.create_data_loader(pca_data, config['FCPflow']['batch_size'], True)
model.load_state_dict(torch.load(os.path.join(_parent_path, 'exp/copula_ana/FCPFlow/FCPflow_model.pth')))
z = torch.randn(pca_data .shape[0], pca_data .shape[1])
cond_test = torch.ones((pca_data.shape[0], 1))  
gen_test = model.inverse(z, cond_test)
re_data = torch.cat((gen_test, cond_test), dim=1)
re_data = scaler.inverse_transform(re_data.cpu().detach().numpy())

# plot the data
plt.figure()
plt.scatter(pca_data[:,0], pca_data[:,1], label='Real data', alpha=0.3)
plt.scatter(samples[0], samples[1], label='t-Copula', alpha=0.3)
plt.scatter(re_data[:,0], re_data[:,1], label='FCPFlow', alpha=0.3)
save_path = os.path.join(_parent_path, 'exp/copula_ana/copula_ana.png')
plt.legend()
plt.savefig(save_path)
plt.close()

  
