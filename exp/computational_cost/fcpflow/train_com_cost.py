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
import tools.tools_train as tl

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path,'exp/computational_cost/fcpflow/config_com_cost.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
        
# define the data loader
data_path = os.path.join(_parent_path, 'data/nl_data_1household.csv')
np_array = pd.read_csv(data_path).iloc[:,3:-2].values
np_array = np_array[~pd.isna(np_array).any(axis=1)]
print('data shape: ', np_array.shape)

# stack one extra column of zeros to the data as the condition
np_array = np.hstack((np_array, np.ones((np_array.shape[0], 1))))
dataloader, scaler = tl.create_data_loader(np_array, config['FCPflow']['batch_size'], True)

# train the model
model = fcpf.FCPflow(config['FCPflow']['num_blocks'], config['FCPflow']['num_channels'], 
                        config['FCPflow']['sfactor'], config['FCPflow']['hidden_dim'], config['FCPflow']['condition_dim']).to(device)

print('number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['FCPflow']['lr_max'], weight_decay=config['FCPflow']['w_decay'])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up=config['FCPflow']['lr_step_size'], 
                                              base_lr=config['FCPflow']['lr_min'], max_lr=config['FCPflow']['lr_max'],
                                              cycle_momentum=False )
# define the wandb
wandb.init(project="com_cost")
wandb.log({"number of parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

# train the model
path = os.path.join(_parent_path, 'exp/computational_cost/fcpflow')
tl.train_com_cost(path, model, dataloader, optimizer, 1000001, config['FCPflow']['condition_dim'], 
                  device, scaler, dataloader, scheduler, 100, True)

