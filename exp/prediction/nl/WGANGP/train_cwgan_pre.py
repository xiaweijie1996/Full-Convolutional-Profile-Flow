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
import alg.cwgan_gp_model as md
from tools.evaluation_m import MMD_kernel, calculate_w_distances
import tools.tools_wgangp as twgan

# ------------------- Define the model -------------------
# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path,'exp/prediction/nl/WGANGP/config_cwgan_pre.yaml')) as file:
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
input_shape = config['CWGAN']['input_shape']
cond_dim = config['CWGAN']['condition_dim']
latent_dim = config['CWGAN']['latent_dim']
hidden_dim = config['CWGAN']['hidden_dim']
generator = md.Generator(input_shape, cond_dim , hidden_dim ,latent_dim).to(device)
discriminator = md.Discriminator(input_shape, cond_dim , hidden_dim).to(device)
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=config['CWGAN']['lr_max'], weight_decay=1e-4)
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=config['CWGAN']['lr_max'])

parem1 = sum(p.numel() for p in generator.parameters() if p.requires_grad)
param2 = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
parem = parem1 + param2
print('number of parameters of generator {}'.format(parem1))

# ------------------- train the model -------------------
twgan.train_cwgan_pre(generator, discriminator, dataloader_train, optimizer_gen, optimizer_dis, 
                scaler, latent_dim, cond_dim, device, _parent_path, epochs=10001, log_wandb=False)