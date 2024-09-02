#!/usr/bin/env python3

import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(_parent_path)

import pandas as pd
import torch
import wandb
import yaml

import alg.models_fcpflow_lin as fcpf
import tools.tools_train as tl

torch.cuda.empty_cache()

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import the configuration
with open(os.path.join(_parent_path,'exp/data_requirement_ana/60_train_data/config_60_fcp.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
        
# define the data loader
data_path = os.path.join(_parent_path, 'data/nl_data_1household.csv')
np_array = pd.read_csv(data_path).iloc[:,3:].values
np_array = np_array[~pd.isna(np_array).any(axis=1)]
split_ratio = 0.8
amount_train_data = 0.6
_length_train = int(np_array.shape[0] * split_ratio * amount_train_data)
_lenggth_test = np_array.shape[0] - int(np_array.shape[0] * split_ratio)
print('data shape: ', np_array.shape)

# stack one extra column of zeros to the data as the condition
train_dataloader, scaler = tl.create_data_loader(np_array[:_length_train,:], _length_train, True)
test_dataloader, _ = tl.create_data_loader(np_array[-_lenggth_test:,:], _lenggth_test, True)

# train the model
model = fcpf.FCPflow(config['FCPflow']['num_blocks'], config['FCPflow']['num_channels'], 
                        config['FCPflow']['sfactor'], config['FCPflow']['hidden_dim'], config['FCPflow']['condition_dim']).to(device)

print('number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['FCPflow']['lr_max'], weight_decay=config['FCPflow']['w_decay'])
scheduler = None

# define the wandb
wandb.init(project="amount_of_data_ana")
wandb.log({"number of parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

# train the model
path = os.path.join(_parent_path, 'exp/data_requirement_ana/60_train_data')
tl.train_data_ana(path, model, train_dataloader, optimizer, 100001, config['FCPflow']['condition_dim'], 
                  device, scaler, test_dataloader, scheduler, 100, True)

