import torch
import torch.nn as nn

def compute_gradient_penalty(D, cond, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.shape[0], 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, cond)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# define the generator of the gan
class Generator(nn.Module):
    def __init__(self, input_dim = 24, cond_dim = 24, hidden_dim = 12, z_dim= 12):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.out_scale = hidden_dim # for 15m
        
        
        self.models = nn.Sequential(
                        
                # first layer
                nn.Linear(self.z_dim + self.cond_dim, self.out_scale*10),
                nn.BatchNorm1d(self.out_scale*10),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale*10, self.out_scale*10),
                nn.BatchNorm1d(self.out_scale*10),
                nn.LeakyReLU(),
                
                # last layer
                nn.Linear(self.out_scale*10, self.input_dim),
                # nn.Tanh()  
            )
        
    def forward(self, z, conds):
        z = torch.cat([z, conds], dim=1)
        out = self.models(z)
        return out

# define the discriminator of the gan
class Discriminator(nn.Module):
    def __init__(self, input_dim = 24, cond_dim = 24, hidden_dim = 12):
        super(Discriminator, self).__init__()

        self.out_scale = hidden_dim 
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        
        self.models =  nn.Sequential(
                
                # first layer
                nn.Linear(self.input_dim + self.cond_dim, self.out_scale*10),
                nn.LayerNorm(self.out_scale*10),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale*10, self.out_scale*10),
                nn.LayerNorm(self.out_scale*10),
                nn.LeakyReLU(),
                
                # last layer
                nn.Linear(self.out_scale*10, 1),
                # nn.Sigmoid() No sigmoid for wgan
            )
        
    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        score = self.models(x)
        return score


