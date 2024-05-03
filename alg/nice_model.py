#%%
import torch.nn as nn
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

class ResiudalBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, condition_dim, output_dim):
        super(ResiudalBlock, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        
        self.net1 = nn.Sequential(
            nn.Linear(self.input_dim+self.condition_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, input):
        output = self.net1(input)+input[:,:self.input_dim] 
        output = self.net2(output)+output
        return output


class ConditionalAffineCouplingLayer(nn.Module):
    def __init__(self, net_type, sfactor, input_dim, hidden_dim, condition_dim, output_dim):
        super(ConditionalAffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        self.net_type = net_type
        self.sfactor = sfactor

        # Conditional scale and translation networks
        self.scale_net1 = self._create_conditional_network()
        self.translate_net1 = self._create_conditional_network()
        self.scale_net2 = self._create_conditional_network()
        self.translate_net2 = self._create_conditional_network()
        # self.u_net1 = self._create_conditional_network_scale()
        # self.u_net2 = self._create_conditional_network_scale()
        self.sigma_net1 = self._create_conditional_network()
        self.sigma_net2 = self._create_conditional_network()


    def _create_conditional_network(self):
        # Network that accepts both input and condition
        if self.net_type == 'linear':
            return nn.Sequential(
                nn.Linear(int(self.input_dim/2)+self.condition_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.LeakyReLU(),
                
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.LeakyReLU(),
                
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.LeakyReLU(),
                
                nn.Linear(self.hidden_dim, int(self.output_dim/2)),
                nn.BatchNorm1d(int(self.output_dim/2)),
                nn.LeakyReLU(),
            )
        elif self.net_type == 'residual':
            return ResiudalBlock(int(self.input_dim/2), self.hidden_dim, self.condition_dim, int(self.output_dim/2))
            
    def forward(self, x, condition):
        # First Affine Transformation with mask1
        x11 = x[:,0::2]
        x12 = x[:,1::2]
        s1 = self.scale_net1(torch.cat([x11, condition], dim=1))
        s1 = torch.atan(s1/self.sfactor)*2*self.sfactor/torch.pi
        t1 = self.translate_net1(torch.cat([x11, condition], dim=1))
        # u1 = self.u_net1(torch.cat([x11, condition], dim=1))
        # sigma1 = self.sigma_net1(torch.cat([x11, condition], dim=1))
        # sigma1 = torch.sign(sigma1)
        x12_trans = x12 # - sigma1
        # log_det_trans_1 = torch.sum(torch.log(torch.abs(sigma1)), dim=[1])
        
        x12_exp = x12_trans*torch.exp(s1)+t1
        log_det_exp_1 = torch.sum(s1, dim=[1])
        x2 = torch.empty_like(x)
        x2[:,0::2] = x11
        x2[:,1::2] = x12_exp
        log_det_1 = log_det_exp_1# + log_det_trans_1
        

        # Second Affine Transformation with mask2
        x21 = x2[:,0::2]
        x22 = x2[:,1::2]
        s2 = self.scale_net2(torch.cat([x22, condition], dim=1))
        s2 = torch.atan(s2/self.sfactor)*2*self.sfactor/torch.pi
        t2 = self.translate_net2(torch.cat([x22, condition], dim=1))
        # u2 = self.u_net2(torch.cat([x22, condition], dim=1))
        # sigma2 = self.sigma_net2(torch.cat([x22, condition], dim=1))
        # sigma2 = torch.atan(sigma2)*2/torch.pi
        x21_trans = x21  #- sigma2
        # log_det_trans_2 = torch.sum(torch.log(torch.abs(sigma2)), dim=[1])
        
        x21_exp = x21_trans*torch.exp(s2)+t2
        log_det_exp_2 = torch.sum(s2, dim=[1])
        y = torch.empty_like(x2)
        y[:,0::2] = x21_exp
        y[:,1::2] = x22
        log_det_2 = log_det_exp_2  #+ log_det_trans_2
        
        # Compute log-determinant
        log_det = log_det_1 + log_det_2

        return y, log_det.mean() # (sigma1 , sigma2)

    def inverse(self, y, condition):
        # First inverse Affine Transformation with mask1
        x21_exp = y[:,0::2]
        x22 = y[:,1::2]
        s2 = self.scale_net2(torch.cat([x22, condition], dim=1))
        s2 = torch.atan(s2/self.sfactor)*2*self.sfactor/torch.pi
        t2 = self.translate_net2(torch.cat([x22, condition], dim=1))
        # u2 = self.u_net2(torch.cat([x22, condition], dim=1))
        # sigma2 = self.sigma_net2(torch.cat([x22, condition], dim=1))
        # sigma2 = torch.atan(sigma2)*2/torch.pi
        x21_trans = (x21_exp-t2)/torch.exp(s2)
        # sigma2 = 1
        x21 = x21_trans #+ sigma2
        
        x2 = torch.empty_like(y)
        x2[:,0::2] = x21
        x2[:,1::2] = x22
                
        # Second inverse Affine Transformation with mask2
        x11 = x2[:,0::2]
        x12_exp = x2[:,1::2]
        s1 = self.scale_net1(torch.cat([x11, condition], dim=1)) 
        s1 = torch.atan(s1/self.sfactor)*2*self.sfactor/torch.pi
        t1 = self.translate_net1(torch.cat([x11, condition], dim=1))
        # u1 = self.u_net1(torch.cat([x11, condition], dim=1))
        # sigma1 = self.sigma_net1(torch.cat([x11, condition], dim=1))
        # sigma1 = torch.atan(sigma1)*2/torch.pi
        x12_trans = (x12_exp-t1)/torch.exp(s1)
        x12 = x12_trans #+ sigma1
    
        x = torch.empty_like(y)
        x[:,0::2] = x11
        x[:,1::2] = x12
        
        return x
    
# net_type, sfactor, input_dim, hidden_dim, condition_dim, output_dim
class NICEflow(nn.Module): # Fully convolutional time flow
    def __init__(self, num_blocks , sfactor, net_type, num_channels, hidden_dim, condition_dim ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.net_type = net_type
        self.input_dim = num_channels
        self.hidden_dim = hidden_dim
        self.output_dim = num_channels
        self.condition_dim = condition_dim
        self.sfactor = sfactor
  
        
        self.blocks = nn.ModuleList([ConditionalAffineCouplingLayer(self.net_type, self.sfactor,  self.num_channels, 
                                                    self.hidden_dim,
                                                  self.condition_dim, self.num_channels) for _ in range(self.num_blocks)])
        # self.Tahhlayer = Tanhlayer()
        
    def forward(self, x, condition):
        log_det = 0
        for block in self.blocks:
            x, log_det1 = block(x, condition)
            log_det += log_det1
        # x, log_det_tahn = self.Tahhlayer(x)
        return x, log_det  # + log_det_tahn
    
    def inverse(self, y, condition):
        # y = self.Tahhlayer.inverse(y)
        for block in reversed(self.blocks):
            y = block.inverse(y, condition)
        return y
   

# check
# data = torch.randn(10, 24)
# cond = torch.randn(10, 24)

# model = NICEflow(6*3, 0.7, 'linear', 24, 24, 24)

# gen, logdet = model(data, cond)
# print(gen.shape)
# print(logdet.shape)

# data_ = model.inverse(gen, cond)

# print((data - data_).mean())