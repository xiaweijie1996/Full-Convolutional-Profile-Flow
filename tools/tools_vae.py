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
    path = os.path.join(_parent_path, 'exp/computational_cost/vae')
    loss_mid = 1000
    for epoch in range(epochs):
        model.train()
        for _, data in enumerate(dataloader):
            pre = data[0].to(device)
            
            # Split the data into data and conditions
            cond = pre[:, -cond_dim:]
            data = pre[:, :-cond_dim]
            
            # Forward pass  
            recon_batch, mu, logvar = model(data, cond)
            loss = md.loss_function_vae(recon_batch, data, mu, logvar, 2)
            
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
            
        # Print the loss
        # print(f'Epoch {epoch}, Generator Loss: {loss.item()}')
        
        # Evaluation and Logging
        model.eval()
        z = torch.rand(cond.shape[0], latent_dim).to(device)
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
        # _dis2 = calculate_w_distances(orig_data_pre, orig_data_re)
        # _dis3 = calculate_energy_distances(orig_data_pre, orig_data_re)
        # _dis5 = ks_distance(orig_data_pre, orig_data_re)


        # Save plots every 100 epochs
        # if epoch % 100 == 0:
        #     save_path = os.path.join(path, 'vae.png')
        #     tl.plot_figure(pre, re_data, scaler, cond_dim, save_path)
            
                    
        if log_wandb:
            wandb.log({
                'time': time.time() - start_time,
                'epoch': epoch,
                'MMD': _dis1,
                # 'Wasserstein': _dis2,
                # 'Energy': _dis3,
                # # 'Autocorrelation': _dis4,
                # 'KS': _dis5,
                'loss': loss.item(),
            })
            
        # Save the model every 
        # if loss_dis.item() < loss_mid :
        #     loss_mid = loss_dis.item()
        #     torch.save(generator.state_dict(), os.path.join(path, 'generator.pth'))
        #     torch.save(discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))
            
    
    # Finish the wandb run if logging
    if log_wandb:
        wandb.finish()
