#!/usr/bin/env python3

import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(_parent_path)

import pandas as pd
import torch
import wandb
import yaml
import numpy as np

import alg.models_fcpflow_lin as fcpf
import alg.tools_train as tl

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path, 'exp/con_rlp_generation/exp_uk/config_uk.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
        
# define the data loader
data_path = os.path.join(_parent_path, 'data', 'uk_data_cleaned_ind_train.csv')
# read the data from the second column to the end
np_array_train = pd.read_csv(data_path, index_col=0).values
data_path = os.path.join(_parent_path, 'data', 'uk_data_cleaned_ind_test.csv')
np_array_test = pd.read_csv(data_path,  index_col=0).values

# stack one extra column of zeros to the data as the condition
dataloader_train, scaler = tl.create_data_loader(np_array_train, config['FCPflow']['batch_size'], True)
dataloader_test, _ = tl.create_data_loader(np_array_train, config['FCPflow']['batch_size'], True)


# train the model
model = fcpf.FCPflow(config['FCPflow']['num_blocks'], config['FCPflow']['num_channels'], 
                        config['FCPflow']['sfactor'], config['FCPflow']['hidden_dim'], config['FCPflow']['condition_dim']).to(device)

print('number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['FCPflow']['lr'], weight_decay=config['FCPflow']['w_decay'])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.001)

# define the wandb
wandb.init(project="fcpflow_uk")
wandb.watch(model)

# train the model
path = os.path.join(_parent_path, 'exp/con_rlp_generation/exp_uk')
tl.train(path, model, dataloader_train, optimizer, 4000001, config['FCPflow']['condition_dim'], device, scaler, dataloader_test, scheduler, 100, True)

