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

# read the data
# define the data loader
data_path = os.path.join(_parent_path, 'data/train_nl_pred.csv')
np_array_train = pd.read_csv(data_path).iloc[:,1:].values
data_path = os.path.join(_parent_path, 'data/test_nl_pred.csv')
np_array_test = pd.read_csv(data_path).iloc[:,1:].values
all_data = np.concatenate((np_array_train, np_array_test), axis=0)
dataloader_train, scaler = tl.create_data_loader(np_array_train, np_array_train.shape[0], True)
np_array_test = scaler.transform(np_array_test)

# load the FCPflow model
with open(os.path.join(_parent_path, 'exp/prediction/nl/FCPFlow/config_nl_pre.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
model = fcpf.FCPflow(config['FCPflow']['num_blocks'], config['FCPflow']['num_channels'], 
                        config['FCPflow']['sfactor'], config['FCPflow']['hidden_dim'], config['FCPflow']['condition_dim'])
model.load_state_dict(torch.load(os.path.join(_parent_path, 'exp/prediction/nl/FCPFlow/FCPflow_model.pth')))

# find the row index with the max value and second max value in np_array_test[:,24:]
_row_peak_indx_max = np.unravel_index(np.argmax(np_array_test[:, 24:]), np_array_test[:, 24:].shape)[0]
_row_peak_indx_min = _row_peak_indx_max + 2

z = torch.randn(500, config['FCPflow']['condition_dim'])
cond = torch.tensor(np_array_test[_row_peak_indx_max,:24]).view(1,-1).repeat(500,1)
re_data = model.inverse(z, cond)
re_data = torch.cat((re_data, cond), dim=1)
pre = torch.tensor(np_array_test)

# plot the data
save_path = os.path.join(_parent_path, 'exp/prediction/nl', 'nl_peak_max.png')
tp.plot_pre(pre, re_data, scaler, 24, _sample_index=_row_peak_indx_max, path=save_path)

z = torch.randn(500, config['FCPflow']['condition_dim'])
cond = torch.tensor(np_array_test[_row_peak_indx_min,:24]).view(1,-1).repeat(500,1)
re_data = model.inverse(z, cond)
re_data = torch.cat((re_data, cond), dim=1)
pre = torch.tensor(np_array_test)


save_path = os.path.join(_parent_path, 'exp/prediction/nl', 'nl_peak_min.png')
tp.plot_pre(pre, re_data, scaler, 24, _sample_index=_row_peak_indx_min, path=save_path)


# plot all the data
save_path = os.path.join(_parent_path, 'exp/prediction/nl', 'nl_all.png')
plt.figure(figsize=(20,10))
plt.plot(all_data[:,24:].T, color='blue', alpha=0.1)
plt.savefig(save_path)
