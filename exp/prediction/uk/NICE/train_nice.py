#!/usr/bin/env python3

import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..','..','..')
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
with open(os.path.join(_parent_path,'exp/prediction/uk/NICE/config_nice_pre.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
        
# define the data loader
data_path = os.path.join(_parent_path, 'data/train_uk_pred.csv')
# read the data from the second column to the end
np_array_train = pd.read_csv(data_path).iloc[:,1:].values
data_path = os.path.join(_parent_path, 'data/test_uk_pred.csv')
np_array_test = pd.read_csv(data_path).iloc[:,1:].values

# stack one extra column of zeros to the data as the condition
dataloader_train, scaler = tl.create_data_loader(np_array_train, np_array_train.shape[0], True)
dataloader_test, _ = tl.create_data_loader(np_array_test, np_array_test.shape[0], True)

# train the model
model = nm.NICE(config['NICE']['num_blocks'], config['NICE']['sfactor'], 'linear', config['NICE']['num_channels'],
                         config['NICE']['hidden_dim'], config['NICE']['condition_dim']).to(device)

print('number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['NICE']['lr_max'], weight_decay=config['NICE']['w_decay'])
scheduler = None

# # define the wandb
# wandb.init(project="com_cost")
# wandb.log({"number of parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

# train the model
path = os.path.join(_parent_path, 'exp/prediction/uk/NICE')
tn.train_nice_pre(path, model, dataloader_train, optimizer, 100001, config['NICE']['condition_dim'], 
                  device, scaler, dataloader_train, scheduler, 200, False)