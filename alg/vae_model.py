import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_shape=24, cond_dim =24 ,latent_dim=12, hidden_dim=12):
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.out_scale1 = hidden_dim # control the model complexity
        
        self.encoder = nn.Sequential(
                # First layer
                nn.Linear(self.input_shape + self.cond_dim, self.out_scale1*10),
                nn.BatchNorm1d(self.out_scale1*10),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale1*10,self.out_scale1*10),
                nn.BatchNorm1d(self.out_scale1*10),
                nn.LeakyReLU(),
                
        
                # third layer
                nn.Linear(self.out_scale1*10,self.out_scale1),
                nn.BatchNorm1d(self.out_scale1),
                nn.LeakyReLU(),
                )
        
        
        self.mu = nn.Linear(self.out_scale1, self.latent_dim) 
        self.logvar = nn.Linear(self.out_scale1, self.latent_dim)

        self.decoder = nn.Sequential(
                # First layer
                nn.Linear(self.latent_dim + self.cond_dim, self.out_scale1*10),
                nn.BatchNorm1d(self.out_scale1*10),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale1*10,self.out_scale1*10),
                nn.BatchNorm1d(self.out_scale1*10),
                nn.LeakyReLU(),
                
                # third layer
                nn.Linear(self.out_scale1*10,self.input_shape),
            )
            
            
    def encode(self, x):
        out = self.encoder(x)
        return self.mu(out), self.logvar(out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        mu, logvar = self.encode(x) 
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, cond], dim=1)
        return self.decode(z), mu, logvar


def loss_function_vae(recon_x, x, mu, logvar, beta):
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld * beta

