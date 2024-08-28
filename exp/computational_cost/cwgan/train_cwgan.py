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
import alg.cwgan_gp_model as md
from tools.evaluation_m import MMD_kernel, calculate_w_distances
import tools.tools_wgangp as twgan

# ------------------- Define the model -------------------
# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path,'exp/computational_cost/cwgan/config_cwgan.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
        
# define the data loader
data_path = os.path.join(_parent_path, 'data/nl_data_1household.csv')
np_array = pd.read_csv(data_path).iloc[:,3:-2].values
np_array = np_array[~pd.isna(np_array).any(axis=1)]
print('data shape: ', np_array.shape)

# stack one extra column of zeros to the data as the condition
np_array = np.hstack((np_array, np.ones((np_array.shape[0], 1))))
dataloader, scaler = tl.create_data_loader(np_array, config['CWGAN']['batch_size'], True)

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
print('number of parameters of discriminator {}'.format(param2))
print('number of parameters {}'.format(parem))

wandb.init(project="com_cost")
wandb.log({"number of parameters": parem1})

# ------------------- train the model -------------------
twgan.train_cwgan(generator, discriminator, dataloader, optimizer_gen, optimizer_dis, 
                scaler, latent_dim, cond_dim, device, _parent_path, epochs=100001, log_wandb=True)