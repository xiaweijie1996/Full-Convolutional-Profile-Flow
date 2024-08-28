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

import alg.nice_model as nm
import tools.tools_nice as tn
import tools.tools_train as tl

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path,'exp/computational_cost/nice/config_nice.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
        
# define the data loader
data_path = os.path.join(_parent_path, 'data/nl_data_1household.csv')
np_array = pd.read_csv(data_path).iloc[:,3:-2].values
np_array = np_array[~pd.isna(np_array).any(axis=1)]
print('data shape: ', np_array.shape)

# stack one extra column of zeros to the data as the condition
np_array = np.hstack((np_array, np.ones((np_array.shape[0], 1))))
dataloader, scaler = tl.create_data_loader(np_array, config['NICE']['batch_size'], True)

# train the model
model = nm.NICE(config['NICE']['num_blocks'], config['NICE']['num_channels'], 
                        config['NICE']['sfactor'], config['NICE']['hidden_dim'], config['NICE']['condition_dim']).to(device)

print('number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['NICE']['lr_max'], weight_decay=config['NICE']['w_decay'])
scheduler = None

# define the wandb
# wandb.init(project="com_cost")
# wandb.log({"number of parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

# train the model
path = os.path.join(_parent_path, 'exp/computational_cost/nice')
tn.train_com_cost(path, model, dataloader, optimizer, 1000001, config['NICE']['condition_dim'], 
                  device, scaler, dataloader, scheduler, 100, False)

