from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb
# from scipy.stats import energy_distance
# import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error

torch.set_default_dtype(torch.float64)

def create_data_loader(numpy_array, batch_size=32, shuffle=True):
    """
    Converts a NumPy array to a PyTorch DataLoader.
    Parameters:
    numpy_array (np.ndarray): The input data in NumPy array format.
    batch_size (int): Size of each batch in the DataLoader.
    shuffle (bool): Whether to shuffle the data.
    Returns:
    DataLoader: A PyTorch DataLoader containing the input data.
    """ 
    # check if nan exists in the data, if nan drop
    if np.isnan(numpy_array).any():
        print('There are nan in the data, drop them')
        numpy_array = numpy_array[~np.isnan(numpy_array).any(axis=1)]

    # scalr the 
    scaler = StandardScaler()
    numpy_array = scaler.fit_transform(numpy_array)
    
    # Convert the NumPy array to a PyTorch Tensor
    tensor_data = torch.Tensor(numpy_array)

    # Create a TensorDataset from the Tensor
    dataset = TensorDataset(tensor_data)
    # Create a DataLoader from the Dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader, scaler


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

 
def kendalltau_corr(data):
    """compute the Kendall Tau correlation between one time series data"""
    data = np.array(data)
    correlation_matrix = np.zeros((data.shape[1], data.shape[1]))
    
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            correlation_matrix[i, j] = kendalltau(data[:, i], data[:, j])[0]
            
    return correlation_matrix


def calculate_autocorrelation_mse(dataset1, dataset2):
    """
    Calculate the correlation and mean square error between two dataset.

    """
    dataset1 = np.array(dataset1)
    dataset2 = np.array(dataset2)
    
    # compute the correlation matrix of data1
    correlation_matrix1 = kendalltau_corr(dataset1)
    
    # compute the correlation matrix of data2
    correlation_matrix2 = kendalltau_corr(dataset2)
    
    # compute the mean square error
    mse = mean_squared_error(correlation_matrix1,correlation_matrix2)
    
    return mse
    

def plot_figure(pre, re_data, scaler, con_dim, path='Generated Data Comparison.png'):
    # Inverse transform to get the original scale of the data
    orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
    orig_data_re = scaler.inverse_transform(re_data.cpu().detach().numpy())
    
    cmap = plt.get_cmap('RdBu_r')
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Two rows for comparison
    
    if con_dim > 0:
        # Original data plot
        _cond_pre = orig_data_pre[:, :-con_dim].sum(axis=1)
        for i, condition in zip(orig_data_pre[:, :-con_dim], _cond_pre):
            color = cmap((condition - _cond_pre.min()) / (_cond_pre.max() - _cond_pre.min()))
            axs[0].plot(i, color=color, alpha=0.1)
        axs[0].set_title('Original Data')
        
        # Reconstructed/Generated data plot
        _cond_re = orig_data_re[:, :-con_dim].sum(axis=1)
        for i, condition in zip(orig_data_re[:, :-con_dim], _cond_re):
            color = cmap((condition - _cond_re.min()) / (_cond_re.max() - _cond_re.min()))
            axs[1].plot(i, color=color, alpha=0.1)
        axs[1].set_title('Reconstructed/Generated Data')

        # Add colorbars to each subplot
        for ax, _cond in zip(axs, [_cond_pre, _cond_re]):
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=_cond.min(), vmax=_cond.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Condition Value scaled', rotation=270, labelpad=20)
        
    else:
        # Original data plot
        for i in orig_data_pre:
            axs[0].plot(i, color='blue', alpha=0.1)
        axs[0].set_title('Original Data')

        # Reconstructed/Generated data plot
        for i in orig_data_re:
            axs[1].plot(i, color='red', alpha=0.1)
        axs[1].set_title('Reconstructed/Generated Data')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def adjust_learning_rate(optimizer, epoch, initial_lr, epochs):
    lr = initial_lr
    change_point = 5000
    if epoch >= change_point:
        lr = initial_lr - (initial_lr-initial_lr/100)*((epoch+1-change_point)/(epochs-change_point))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    
# define a function to train the model
def train(path, model, train_loader, optimizer, epochs, cond_dim ,device, scaler, test_loader, scheduler, pgap=100, _wandb=True):
    model.train()
    loss_mid = 0
    for epoch in range(epochs):
        for _, data in enumerate(train_loader):
            pre = data[0].to(device) # + torch.randn_like(data[0].to(device))/(256)
            
            # split the data into data and conditions
            cond = pre[:,-cond_dim:]
            data = pre[:,:-cond_dim] # + torch.rand_like(data[:,:-cond_dim])/(256) 
            
            gen, logdet = model(data, cond)
            
            # compute the log likelihood loss
            llh = log_likelihood(gen, type='Gaussian')
            loss = -llh.mean()-logdet
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # weight clipping
            for p in model.parameters():
                p.data.clamp_(-1, 1)
                
            # adjust_learning_rate(optimizer, epoch, lr, epochs)
        
        # ----------------- moniter loss -----------------
        print(epoch, 'loss: ', loss.item())
        if _wandb:
            wandb.log({'loss': loss.item()})
        # ----------------- moniter loss -----------------
            
        # ----------------- test the model -----------------
        model.eval()
        
        # test the model
        pre = next(iter(test_loader))[0].to(device)
        cond_test = pre[:,-cond_dim:]
        data_test = pre[:,:-cond_dim]
        
        gen_test, logdet_test = model(data_test, cond_test)
        llh_test = log_likelihood(gen_test, type='Gaussian')
        loss_test = -llh_test.mean()-logdet_test
        
        # save the model
        if loss_test.item() < loss_mid:
            save_path = path + '/FCPflow_model.pth'
            torch.save(model.state_dict(), save_path)
            loss_mid = loss_test.item()
        # ----------------- test the model -----------------
        
        # ----------------- plot the generated data -----------------
        if epoch % pgap == 0:
            # plot the generated data
            z = torch.randn(data_test.shape[0], data_test.shape[1]).to(device)
            gen_test = model.inverse(z, cond_test)
            re_data = torch.cat((gen_test, cond_test), dim=1)
            re_data = re_data.detach()
            save_path = path + '/FCPflow_generated.png'
            plot_figure(pre, re_data, scaler, cond_dim, save_path)
        # ----------------- plot the generated data -----------------
        model.train()
        

def compute_quantile(pre_data):
    q = np.arange(0.01, 1, 0.01)
    quantile_value = np.quantile(pre_data, q, axis=0)
    return quantile_value 

def pinball_loss_compute(y, f, q):
    if y >= f:
        return (y - f) * q
    else:
        return (f - y) * (1 - q)
    
def plloss(pre_data, true_data):

    quantile_values = compute_quantile(pre_data)
    q = np.arange(0.01, 1, 0.01)
    
    loss = []
    for i in range(len(q)):
        ind_loss = [pinball_loss_compute(y, f, q[i]) for y, f in zip(true_data, quantile_values[i,:])]
        loss.append(np.mean(ind_loss))
    pinball_loss = np.mean(loss)

    return pinball_loss