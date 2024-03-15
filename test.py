# import packages and modules
import alg.tools as tl
import alg.models_fctflow_lin as fctf
import pandas as pd
import torch

# load the data
path_train = r'data\usa_data_cleaned_annual_train.csv'
df_train = pd.read_csv(path_train, index_col=0)
df_train = df_train.dropna()
df_train = df_train.iloc[:100,:] # select the first 200 samples to train for this small example

paht_test = r'data\usa_data_cleaned_annual_test.csv'
df_test = pd.read_csv(paht_test, index_col=0)
df_test = df_test.dropna()
test_set =df_test.values[:30,:] # select the first 20 samples as test for this small example

# create the data loader 
dataLoader, scaler = tl.create_data_loader(df_train.values, batch_size=df_train.shape[0], shuffle=True) # I'm too lazy to wait, so I secretly load all the data once
data_plot = torch.tensor(scaler.transform(df_train.values[:,:]))
 # plot the real data, the y-axis is the value of the feature, the x-axis is the time, the color of profile indicates the total daily energy consumption
tl.plot_figure(data_plot , scaler, 2)

# define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train the model
while True:
        
    # hyperparameters
    num_blocks = 4 # a small number of blocks for this small example, better not exceed 15 blocks to make your life easier
    net_type = 'fullconv'  
    sfactor = 0.3 # scaling factor, 0.1 to 1 is a good range
    hidden_dim = 96*2 # hidden dimension of the s and t networks
    lr = 0.001
    w_decay = 0 # weight decay of the optimizer

    # define the model
    num_channels = 96 # resolution of the time series 
    condition_dim = 2  # annual energy consumption and daily energy consumption
    model = fctf.FCTflow(num_blocks, num_channels, net_type, sfactor, hidden_dim, condition_dim)
        
    # print the number of parameters
    print('Number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    
    # train the model 
    tl.train(model, dataLoader, optimizer, 50000, condition_dim, device, scaler, lr, test_set, pgap=200)
    
    break