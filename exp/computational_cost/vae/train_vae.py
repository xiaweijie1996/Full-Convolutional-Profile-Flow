import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..','..')
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
with open(os.path.join(_parent_path,'exp/computational_cost/vae/config_vae.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
        
# define the data loader
data_path = os.path.join(_parent_path, 'data/nl_data_1household.csv')
np_array = pd.read_csv(data_path).iloc[:,3:-2].values
np_array = np_array[~pd.isna(np_array).any(axis=1)]

# stack one extra column of zeros to the data as the condition
np_array = np.hstack((np_array, np.ones((np_array.shape[0], 1))))
print('data shape: ', np_array.shape)
dataloader, scaler = tl.create_data_loader(np_array, config['VAE']['batch_size'], True)

# define the model
input_shape = config['VAE']['input_shape']
cond_dim = config['VAE']['condition_dim']
latent_dim = config['VAE']['latent_dim']
hidden_dim = config['VAE']['hidden_dim']
model = md.VAE(input_shape=input_shape, cond_dim =cond_dim ,latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.Adam(model .parameters(), lr=1e-3, weight_decay=0)

parem = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
print('number of parameters {}'.format(parem))

wandb.init(project="com_cost")
wandb.log({"number of parameters": parem})

# ------------------- train the model -------------------
tvae.train_vae(model, dataloader, optimizer, scaler, latent_dim, cond_dim, device, _parent_path, epochs=260000, log_wandb=True)