import os

import time
import torch
import wandb
import pandas as pd

import tools.tools_train as tl
import alg.vae_model as md
from tools.evaluation_m import MMD_kernel, calculate_w_distances, calculate_energy_distances, ks_distance


def train_vae(model, dataloader, optimizer, scaler, latent_dim, cond_dim, device, _parent_path, epochs=10001, log_wandb=False):
    # Initialize wandb if logging is enabled
    start_time = time.time()
    path = os.path.join(_parent_path, 'exp/prediction/nl/VAE')
    mid_dis1 = 1000
    for epoch in range(epochs):
        model.train()
        for _, data in enumerate(dataloader):
            pre = data[0].to(device)
            
            # Split the data into data and conditions
            cond = pre[:, -cond_dim:]
            data = pre[:, :-cond_dim]
            
            # Forward pass  
            recon_batch, mu, logvar = model(data, cond)
            loss = md.loss_function_vae(recon_batch, data, mu, logvar, 10)
            
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
        
        # Evaluation and Logging
        model.eval()
        z = torch.randn(cond.shape[0], latent_dim).to(device)
        z = torch.cat([z, cond], dim=1)
        recon = model.decode(z)
        re_data = torch.cat([recon, cond], dim=1)
        
        # Inverse scaling to original data scale
        orig_data_re = scaler.inverse_transform(re_data.cpu().detach().numpy())
        orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
                    
        # check if nan in the data and cancel the nan values
        orig_data_re = orig_data_re[~pd.isna(orig_data_re).any(axis=1)]
        orig_data_pre = orig_data_pre[~pd.isna(orig_data_pre).any(axis=1)]
        
        # Compute the MMD (Maximum Mean Discrepancy)
        _dis1 = MMD_kernel(orig_data_pre, orig_data_re)

        # Print the loss
        print(f'Epoch {epoch}, Generator Loss: {loss.item()}', f'MMD: {_dis1}')
        
        # Save plots every 100 epochs
        if epoch % 100 == 0:
            save_path = os.path.join(path, 'vae.png')
            tl.plot_figure(pre, re_data, scaler, cond_dim, save_path)
            
                    
        if log_wandb:
            wandb.log({
                'time': time.time() - start_time,
                'epoch': epoch,
                'MMD': _dis1,
                'loss': loss.item(),
            })
            
        # save the model
        if _dis1 < mid_dis1:
            loss_mid = _dis1
            torch.save(model.state_dict(), os.path.join(path, 'VAE_model.pth'))
            

    
    # Finish the wandb run if logging
    if log_wandb:
        wandb.finish()

def train_vae_pre(model, dataloader, optimizer, scaler, latent_dim, cond_dim, device, _parent_path, epochs=10001, log_wandb=False):
    # initialize wandb if logging is enabled
    start_time = time.time()
    path = os.path.join(_parent_path, 'exp/prediction/uk/VAE')
    mid_dis1 = 1000
    for epoch in range(epochs):
        model.train()
        for _, data in enumerate(dataloader):
            pre = data[0].to(device)
            
            # split the data into data and conditions
            cond = pre[:, :cond_dim]
            data = pre[:, cond_dim:]
            
            # forward pass  
            recon_batch, mu, logvar = model(data, cond)
            loss = md.loss_function_vae(recon_batch, data, mu, logvar, 10)
            
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
        
        # evaluation and Logging
        model.eval()
        z = torch.randn(cond.shape[0], latent_dim).to(device)
        z = torch.cat([z, cond], dim=1)
        recon = model.decode(z)
        re_data = torch.cat([cond, recon], dim=1)
        
        # inverse scaling to original data scale
        orig_data_re = scaler.inverse_transform(re_data.cpu().detach().numpy())
        orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
                    
        # check if nan in the data and cancel the nan values
        orig_data_re = orig_data_re[~pd.isna(orig_data_re).any(axis=1)]
        orig_data_pre = orig_data_pre[~pd.isna(orig_data_pre).any(axis=1)]
        
        # compute the MMD (Maximum Mean Discrepancy)
        _dis1 = MMD_kernel(orig_data_pre, orig_data_re)

        # print the loss
        print(f'Epoch {epoch}, Generator Loss: {loss.item()}', f'MMD: {_dis1}')
        
        # save plots every 100 epochs
        if epoch % 100 == 0:
            save_path = os.path.join(path, 'vae.png')
            tl.plot_pre(pre, re_data, scaler, cond_dim, _sample_index=0, path=save_path)
            
        if log_wandb:
            wandb.log({
                'time': time.time() - start_time,
                'epoch': epoch,
                'MMD': _dis1,
                'loss': loss.item(),
            })
            
        # save the model
        if _dis1 < mid_dis1:
            mid_dis1 = _dis1
            print('model saved at epoch: ', epoch)
            torch.save(model.state_dict(), os.path.join(path, 'VAE_model.pth'))
            
    # finish the wandb run if logging
    if log_wandb:
        wandb.finish()
