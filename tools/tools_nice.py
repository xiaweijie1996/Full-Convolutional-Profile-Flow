from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
import time

import tools.tools_train as tl
from tools.evaluation_m import MMD_kernel, calculate_w_distances, calculate_energy_distances, ks_distance

torch.set_default_dtype(torch.float64)

def log_likelihood(x, type='Gaussian'):
    """Compute log likelihood for x under a uniform distribution in [0,1]^D.
    Args:
    - x (torch.Tensor): Input tensor of shape (batch_size, D)
    Returns:
    - log_likelihood (torch.Tensor): Log likelihood for each sample in the batch. Shape: (batch_size,)
    """
    if type == 'Uniform':
        # Check if all values in x are within the interval [0,1]
        is_inside = ((x >= -1) & (x <= 1)).all(dim=1).float()
        log_likelihood = 0 * is_inside - (x*x).mean(dim=1)*(1-is_inside)
    if type == 'Gaussian':
        log_likelihood = -0.5 * x.pow(2).sum(dim=1)
    return log_likelihood


def train_com_cost(path, model, train_loader, optimizer, epochs, cond_dim ,device, scaler, test_loader, scheduler, pgap=100, _wandb=True):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        for _, data in enumerate(train_loader):
            model.train()
            pre = data[0].to(device) # + torch.randn_like(data[0].to(device))/(256)
            
            # split the data into data and conditions
            cond = pre[:,-cond_dim:]
            data = pre[:,:-cond_dim] # + torch.rand_like(data[:,:-cond_dim])/(256) 
            
            gen, logdet = model(data, cond)
            
            # compute the log likelihood loss
            llh = log_likelihood(gen, type='Gaussian')
            loss = -llh.mean()-logdet
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
        # ----------------- test the model -----------------
        model.eval()
        
        print('epoch: ', epoch, 'loss: ', loss.item())

        # plot the generated data
        z = torch.randn(data.shape[0], data.shape[1]).to(device)
        gen_test = model.inverse(z, cond)
        re_data = torch.cat((gen_test, cond), dim=1)
        re_data = re_data.detach()
        
        orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
        orig_data_re = scaler.inverse_transform(re_data.cpu().detach().numpy())
        
        # comput energy distance
        _dis1 = MMD_kernel(orig_data_pre, orig_data_re)
        _dis2 = calculate_w_distances(orig_data_pre, orig_data_re)
        _dis3 = calculate_energy_distances(orig_data_pre, orig_data_re)
        # _dis4 = calculate_autocorrelation_mse(orig_data_pre, orig_data_re)
        _dis5 = ks_distance(orig_data_pre, orig_data_re)
        

        # ----------------- plot the generated data -----------------
        if epoch % pgap ==0: 
            save_path = path + '/NICE_generated.png'
            tl.plot_figure(pre, re_data, scaler, cond_dim, save_path)
        # ----------------- plot the generated data -----------------

        if _wandb:
            wandb.log({
                'time': time.time() - start_time,
                'epoch': epoch,
                'MMD': _dis1,
                'Wasserstein': _dis2,
                'Energy': _dis3,
                # 'Autocorrelation': _dis4,
                'KS': _dis5,
                'loss': loss.item(),
            })

        # ----------------- test the model -----------------
        


