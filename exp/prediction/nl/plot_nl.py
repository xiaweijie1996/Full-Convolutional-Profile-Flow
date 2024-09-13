import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(_parent_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import yaml

import alg.models_fcpflow_lin as fcpf
import tools.tools_train as tl
import tools.tools_pre as tp
import alg.cwgan_gp_model as md
import alg.nice_model as nm
import alg.vae_model as vm

# read the data
data_path = os.path.join(_parent_path, 'data/train_nl_pred.csv')
np_array_train = pd.read_csv(data_path).iloc[:,1:].values
data_path = os.path.join(_parent_path, 'data/test_nl_pred.csv')
np_array_test = pd.read_csv(data_path).iloc[:,1:].values
all_data = np.concatenate((np_array_train, np_array_test), axis=0)
dataloader_train, scaler = tl.create_data_loader(np_array_train, np_array_train.shape[0], True)
np_array_test = scaler.transform(np_array_test)

# experiment configuration
_row_peak_indx_max = np.unravel_index(np.argmax(np_array_test[:, 24:]), np_array_test[:, 24:].shape)[0]-1 # 0, 12, -1
_sample_num = 100
cond = torch.tensor(np_array_test[_row_peak_indx_max,:24]).view(1,-1).repeat(_sample_num,1)
pre = torch.tensor(np_array_test)

# ------------load the FCPflow model------------
with open(os.path.join(_parent_path, 'exp/prediction/nl/FCPFlow/config_pre.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
model = fcpf.FCPflow(config['FCPflow']['num_blocks'], config['FCPflow']['num_channels'], 
                        config['FCPflow']['sfactor'], config['FCPflow']['hidden_dim'], config['FCPflow']['condition_dim'])
model.load_state_dict(torch.load(os.path.join(_parent_path, 'exp/prediction/nl/FCPFlow/FCPflow_model.pth')))
model.eval()
z = torch.randn(_sample_num, config['FCPflow']['condition_dim'])
re_data = model.inverse(z, cond)
re_data_fcpflow = torch.cat((cond, re_data), dim=1)
# ------------load the FCPflow model------------


# ------------ load the CWGAN-GP ------------
with open(os.path.join(_parent_path, 'exp/prediction/nl/WGANGP/config_cwgan_pre.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
input_shape = config['CWGAN']['input_shape']
cond_dim = config['CWGAN']['condition_dim']
latent_dim = config['CWGAN']['latent_dim']
hidden_dim = config['CWGAN']['hidden_dim']
generator = md.Generator(input_shape, cond_dim , hidden_dim ,latent_dim)
generator.load_state_dict(torch.load(os.path.join(_parent_path, 'exp/prediction/nl/WGANGP/CWGANGP.pth')))
generator.eval()
z = torch.randn(cond.shape[0], latent_dim)
recon = generator(z, cond)
recon = recon.cpu().detach()
re_data_wgangp = torch.cat([cond.cpu().detach(), recon], dim=1)
# ------------ load the CWGAN-GP ------------

# ------------ load the NICE ------------
with open(os.path.join(_parent_path, 'exp/prediction/nl/NICE/config_nice_pre.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
nice = nm.NICE(config['NICE']['num_blocks'], config['NICE']['sfactor'], 'linear', config['NICE']['num_channels'],
                         config['NICE']['hidden_dim'], config['NICE']['condition_dim'])
nice.load_state_dict(torch.load(os.path.join(_parent_path, 'exp/prediction/nl/NICE/NICE_model.pth')))
nice.eval()
z = torch.randn(_sample_num, config['NICE']['condition_dim'])
re_data = nice.inverse(z, cond)
re_data_nice = torch.cat((cond, re_data), dim=1)
# ------------ load the NICE ------------

# ------------ load the VAE ------------
with open(os.path.join(_parent_path, 'exp/prediction/nl/VAE/config_vae_pre.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
vae = vm.VAE(input_shape=config['VAE']['input_shape'], cond_dim=config['VAE']['condition_dim'], 
                latent_dim=config['VAE']['latent_dim'], hidden_dim=config['VAE']['hidden_dim'])
vae.load_state_dict(torch.load(os.path.join(_parent_path, 'exp/prediction/nl/VAE/VAE_model.pth')))
vae.eval()
z = torch.randn(_sample_num, config['VAE']['latent_dim'])
z = torch.cat([z, cond], dim=1)
re_data = vae.decode(z)
re_data_vae = torch.cat((cond, re_data), dim=1)
# ------------ load the VAE ------------

# ------------ plot the data ------------
save_path = os.path.join(_parent_path, 'exp/prediction/nl', 'nl_peak_all.png')
re_data_list = None
tp.plot_pre(pre, re_data_fcpflow, scaler, 24, re_data_list, _sample_index=_row_peak_indx_max, path=save_path)

# plot all the data
save_path = os.path.join(_parent_path, 'exp/prediction/nl', 'nl_all.png')
plt.figure(figsize=(20,10))
plt.plot(all_data[:,24:].T, color='blue', alpha=0.1)
plt.savefig(save_path)