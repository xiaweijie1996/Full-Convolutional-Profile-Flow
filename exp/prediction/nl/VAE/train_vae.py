import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..','..','..')
sys.path.append(_parent_path)

import pandas as pd
import torch
import wandb
import yaml
import numpy as np
import time

import tools.tools_train as tl
import alg.vae_model as md
from tools.evaluation_m import MMD_kernel, calculate_w_distances
import tools.tools_vae as tvae

# ------------------- Define the model -------------------
# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path,'exp/prediction/nl/VAE/config_vae_pre.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
        
# define the data loader
data_path = os.path.join(_parent_path, 'data/train_nl_pred.csv')
# read the data from the second column to the end
np_array_train = pd.read_csv(data_path).iloc[:,1:].values
data_path = os.path.join(_parent_path, 'data/test_nl_pred.csv')
np_array_test = pd.read_csv(data_path).iloc[:,1:].values

# stack one extra column of zeros to the data as the condition
dataloader_train, scaler = tl.create_data_loader(np_array_train, np_array_train.shape[0], True)
dataloader_test, _ = tl.create_data_loader(np_array_test, np_array_test.shape[0], True)

# define the model
input_shape = config['VAE']['input_shape']
cond_dim = config['VAE']['condition_dim']
latent_dim = config['VAE']['latent_dim']
hidden_dim = config['VAE']['hidden_dim']
model = md.VAE(input_shape=input_shape, cond_dim =cond_dim ,latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.Adam(model .parameters(), lr=1e-3, weight_decay=0)

parem = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
print('number of parameters {}'.format(parem))

# ------------------- train the model -------------------
path = os.path.join(_parent_path, 'exp/prediction/nl/VAE')
tvae.train_vae_pre(model, dataloader_train, optimizer, scaler, latent_dim, cond_dim, device, path, epochs=500, log_wandb=False)