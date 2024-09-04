import os

import time
import torch
import wandb
import pandas as pd
import numpy as np

import tools.tools_train as tl
import alg.cwgan_gp_model as md
from tools.evaluation_m import MMD_kernel, pinball_loss

def train_cwgan(generator, discriminator, dataloader, optimizer_gen, optimizer_dis, 
                scaler, latent_dim, cond_dim, device, _parent_path, epochs=10001, log_wandb=False):
    # Initialize wandb if logging is enabled
    start_time = time.time()
    path = os.path.join(_parent_path, 'exp/prediction/nl/WGANGP')
    loss_mid = 1000
    for epoch in range(epochs):
        for _, data in enumerate(dataloader):
            generator.train()
            discriminator.train()
            pre = data[0].to(device)
            
            # Split the data into data and conditions
            cond = pre[:, -cond_dim:]
            data = pre[:, :-cond_dim]
            
            # Generate fake data
            z = torch.randn(data.shape[0], latent_dim).to(device)
            gen_fake = generator(z, cond)
            
            # Discriminator forward pass
            fake_label = discriminator(gen_fake.detach(), cond)
            real_label = discriminator(data, cond)
            
            # Gradient penalty
            grad_p = md.compute_gradient_penalty(discriminator, cond, data, gen_fake.detach(), device)
            
            # Calculate the WGAN loss for discriminator
            loss_dis = -torch.mean(real_label) + torch.mean(fake_label) + 10 * grad_p
            
            # Update the discriminator
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()
            
            # Train the generator
            for _ in range(3):
                z = torch.randn(pre.shape[0], latent_dim).to(device)
                gen_fake = generator(z, cond)
                
                fake_label = discriminator(gen_fake, cond)
                loss_gen = -torch.mean(fake_label)
                
                optimizer_gen.zero_grad()
                loss_gen.backward()
                optimizer_gen.step()
            
        # Print the loss
        print(f'Epoch {epoch}, Generator Loss: {loss_gen.item()}, Discriminator Loss: {loss_dis.item()}')
        
        # Evaluation and Logging
        generator.eval()
        z = torch.randn(cond.shape[0], latent_dim).to(device)
        recon = generator(z, cond)
        recon = recon.cpu().detach()
        re_data = torch.cat([recon, cond.cpu().detach()], dim=1)
        
        # Inverse scaling to original data scale
        orig_data_re = scaler.inverse_transform(re_data.numpy())
        orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
                    
        # check if nan in the data and cancel the nan values
        orig_data_re[orig_data_re < 0] = 0
        
        # Compute the MMD (Maximum Mean Discrepancy)
        _dis1 = MMD_kernel(orig_data_pre, orig_data_re)
        # _dis2 = calculate_w_distances(orig_data_pre, orig_data_re)
        # _dis3 = calculate_energy_distances(orig_data_pre, orig_data_re)
        # _dis5 = ks_distance(orig_data_pre, orig_data_re)


        # Save plots every 100 epochs
        if epoch % 100 == 0:
            save_path = os.path.join(path, 'CWGAN_generated.png')
            tl.plot_figure(pre, re_data, scaler, cond_dim, save_path)
            
                    
        if log_wandb:
            wandb.log({
                'time': time.time() - start_time,
                'epoch': epoch,
                'MMD': _dis1,
                # 'Wasserstein': _dis2,
                # 'Energy': _dis3,
                # # 'Autocorrelation': _dis4,
                # 'KS': _dis5,
                'loss_dis': loss_dis.item(),
                'loss_gen': loss_gen.item()
            })
            
        # Save the model every 
        if _dis1 < loss_mid :
            loss_mid = _dis1
            torch.save(generator.state_dict(), os.path.join(path, 'generator.pth'))
            # torch.save(discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))
            
    
    # Finish the wandb run if logging
    if log_wandb:
        wandb.finish()


def train_cwgan_pre(generator, discriminator, dataloader, optimizer_gen, optimizer_dis, 
                scaler, latent_dim, cond_dim, device, path, epochs=10001, log_wandb=False):
    # Initialize wandb if logging is enabled
    loss_mid = 1000
    for epoch in range(epochs):
        for _, data in enumerate(dataloader):
            generator.train()
            discriminator.train()
            pre = data[0].to(device)
            
            # Split the data into data and conditions
            cond = pre[:, :cond_dim]
            data = pre[:, cond_dim:]
            
            # Generate fake data
            z = torch.randn(data.shape[0], latent_dim).to(device)
            gen_fake = generator(z, cond)
            
            # Discriminator forward pass
            fake_label = discriminator(gen_fake.detach(), cond)
            real_label = discriminator(data, cond)
            
            # Gradient penalty
            grad_p = md.compute_gradient_penalty(discriminator, cond, data, gen_fake.detach(), device)
            
            # Calculate the WGAN loss for discriminator
            loss_dis = -torch.mean(real_label) + torch.mean(fake_label) + 10 * grad_p
            
            # Update the discriminator
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()
            
            # Train the generator
            for _ in range(3):
                z = torch.randn(pre.shape[0], latent_dim).to(device)
                gen_fake = generator(z, cond)
                
                fake_label = discriminator(gen_fake, cond)
                loss_gen = -torch.mean(fake_label)
                
                optimizer_gen.zero_grad()
                loss_gen.backward()
                optimizer_gen.step()
            
        generator.eval()
        z = torch.randn(cond.shape[0], latent_dim).to(device)
        recon = generator(z, cond)
        recon = recon.cpu().detach()
        re_data = torch.cat([cond.cpu().detach(), recon], dim=1)
        
        # Inverse scaling to original data scale
        orig_data_re = scaler.inverse_transform(re_data.numpy())
        orig_data_pre = scaler.inverse_transform(pre.cpu().detach().numpy())
                    
        # Compute the MMD (Maximum Mean Discrepancy)
        _dis1 = MMD_kernel(orig_data_pre[:, cond_dim:], orig_data_re[:,cond_dim:])
        print(f'Epoch {epoch}, PL: {_dis1}', 'loss_dis:', loss_dis.item(), 'loss_gen:', loss_gen.item())
        
        if log_wandb:
            wandb.log({
                'epoch': epoch,
                'PL': _dis1,
                'loss_dis': loss_dis.item(),
                'loss_gen': loss_gen.item()
            })
            
        # Save the model every 
        if _dis1 < loss_mid :
            print('Saving model')
            loss_mid = loss_dis.item()
            torch.save(generator.state_dict(), os.path.join(path, 'CWGANGP.pth'))
            loss_mid = _dis1
            
        # Save plots every 100 epochs
        if epoch % 100 == 0:
            save_path = os.path.join(path, 'CWGAN_generated.png')
            tl.plot_pre(pre, re_data, scaler, cond_dim, 0, save_path)
            
    # Finish the wandb run if logging
    if log_wandb:
        wandb.finish()
