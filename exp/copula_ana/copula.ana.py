import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..')
sys.path.append(_parent_path)

import pandas as pd
import torch    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import yaml
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'tools/TIMES.TTF'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

from multicopula_model import EllipticalCopula
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
samples, inv_collection = copula.sample(pca_data.shape[0])
samples = samples.T

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
re_data = scaler.inverse_transform(gen_test.cpu().detach().numpy())

# compute the empirical cdf
cdf_pca_data_1, cdf_pca_data_2 = compute_empirical_marginal_cdf(pca_data)

# evaluate the cdf of real data and samples
cdf_real_1 = cdf_pca_data_1(pca_data[:, 0])
cdf_real_2 = cdf_pca_data_2(pca_data[:, 1])
cdf_sample_1 = cdf_pca_data_1(samples[:, 0])
cdf_sample_2 = cdf_pca_data_2(samples[:, 1])
cdf_re_1 = cdf_pca_data_1(re_data[:, 0])
cdf_re_2 = cdf_pca_data_2(re_data[:, 1])

# plot the data
# plt.figure()
# plt.scatter(pca_data[:,0], pca_data[:,1], label='Real data', alpha=0.3)
# plt.scatter(samples[:,0], samples[:,1], label='t-Copula', alpha=0.3)
# plt.scatter(re_data[:,0], re_data[:,1], label='FCPFlow', alpha=0.3)
# save_path = os.path.join(_parent_path, 'exp/copula_ana/copula_ana.png')
# plt.legend()
# plt.savefig(save_path)
# plt.close()

# plot the cdf
_size = 20
plt.figure(figsize=(6, 6))
plt.scatter(cdf_real_1, cdf_real_2, label='Real data', alpha=0.5)
plt.scatter(cdf_sample_1, cdf_sample_2, label='t-Copula', alpha=0.2)
plt.scatter(cdf_re_1, cdf_re_2, label='FCPFlow', alpha=0.5)
plt.legend(fontsize=_size-6)
# also apply this to legend's font
for text in plt.gca().get_legend().get_texts():
    text.set_fontproperties(font_prop)
    text.set_fontsize(_size-6)
# Apply the custom font properties to the x and y tick labels
for label in plt.gca().get_xticklabels():
    label.set_fontproperties(font_prop)  # Apply font properties to x-ticks
    label.set_fontsize(_size-6)  # Ensure the size is applied

for label in plt.gca().get_yticklabels():
    label.set_fontproperties(font_prop)  # Apply font properties to y-ticks
    label.set_fontsize(_size-6)  # Ensure the size is applied
plt.xlabel('Cumulative Probability $u_1$ [-]', fontsize=_size, fontproperties=font_prop)
plt.ylabel('Cumulative Probability $u_2$ [-]', fontsize=_size, fontproperties=font_prop)

plt.savefig(os.path.join(_parent_path, 'exp/copula_ana/cdf_plot.png'))
plt.close()


  
