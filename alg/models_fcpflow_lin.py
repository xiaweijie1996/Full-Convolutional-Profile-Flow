#%%
import torch.nn as nn
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

class InvertibleNorm(nn.Module):
    def __init__(self, num_channels):
        super(InvertibleNorm, self).__init__()
        self.num_channels = num_channels
        # Initialize running mean and standard deviation
        self.register_buffer('running_mean', torch.zeros(1, num_channels, 1))
        self.register_buffer('running_std', torch.ones(1, num_channels, 1))
        self.initialized = False
        self.momentum = 0.1

    def forward(self, input):
        if self.training:
            # Calculate mean and std dev for the current batch
            mean = torch.mean(input, dim=[0, 2], keepdim=True)
            std = torch.std(input, dim=[0, 2], keepdim=True) + 1e-10

            # Normalize input using calculated mean and std dev
            normalized_input = (input - mean) / std

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * std
        else:
            # Normalize input using running statistics during evaluation
            normalized_input = (input - self.running_mean) / self.running_std

        # log-determinant of the Jacobian
        self.scale = 1 / (std + 1e-10)
        log_det = (
            torch.sum(torch.log(torch.abs(self.scale.squeeze() + 1e-10)))
        )
        return normalized_input, log_det

    def inverse(self, output):
        # Use running_mean and running_std for the inverse normalization
        return (output * self.running_std) + self.running_mean
    

class InvertibleWConv(nn.Module):
    def __init__(self, num_channels):
        super(InvertibleWConv, self).__init__()
        self.num_channels = num_channels

        # Initialize weights with a random rotation matrix
        w_init = torch.qr(torch.randn(num_channels, num_channels))[0]
        self.w = nn.Parameter(w_init)

    def forward(self, input):
        out = F.conv1d(input, self.w.view(self.num_channels, self.num_channels, 1))
        log_det = torch.slogdet(self.w)[1]
        return out, log_det

    def inverse(self, y):
        # Compute the inverse of the weights
        w_inv = torch.inverse(self.w)
        return F.conv1d(y, w_inv.view(self.num_channels, self.num_channels, 1))

            
class Simple1DfullConvNet(nn.Module):
    def __init__(self, in_c, h_c, out_c, linear = False):
        super().__init__()
        self.in_c = in_c
        self.h_c = h_c
        self.out_c = out_c
        self.linear = linear
        self.bias = True
        self.model = nn.Sequential( 
            nn.Linear(in_features=self.in_c, out_features=self.h_c, bias=self.bias),
            nn.BatchNorm1d(self.h_c),
            nn.LeakyReLU(),
            
            nn.Linear(in_features=self.h_c, out_features=self.h_c, bias=self.bias),
            nn.BatchNorm1d(self.h_c),
            nn.LeakyReLU(),
            
            nn.Linear(in_features=self.h_c, out_features=self.h_c, bias=self.bias),
            nn.BatchNorm1d(self.h_c),
            nn.LeakyReLU(),
            
            nn.Linear(in_features=self.h_c, out_features=self.h_c, bias=self.bias),
            nn.BatchNorm1d(self.h_c),
            nn.LeakyReLU(),
        )
        
        self.linear_model = nn.Linear(self.h_c, self.out_c)
        # self.leakrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        if not self.linear:
                # x = x.reshape(x.shape[0], self.in_c, 1)
                x = self.model(x)
                x = x.view(x.shape[0], -1)
                x = self.linear_model(x)
                x = self.leakrelu(x)
                return x
        else:
                # x = x.reshape(x.shape[0], self.in_c, 1)
                x = self.model(x)
                x = x.view(x.shape[0], -1)
                x = self.linear_model(x)
                return x


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
        self.scale_net1 = self._create_conditional_network1()
        self.translate_net1 = self._create_conditional_network2()
        self.scale_net2 = self._create_conditional_network1()
        self.translate_net2 = self._create_conditional_network2()

    def _create_conditional_network1(self):
        # Network that accepts both input and condition
        in_channel = int(self.input_dim/2)+self.condition_dim
        hidden_channel = self.hidden_dim
        out_channel = int(self.output_dim/2)
       
        if self.net_type == 'fullconv':
            return Simple1DfullConvNet(in_channel, hidden_channel, out_channel, True)
        
    def _create_conditional_network2(self):
        # Network that accepts both input and condition
        in_channel = int(self.input_dim/2)+self.condition_dim
        hidden_channel = self.hidden_dim
        out_channel = int(self.output_dim/2)
        if self.net_type == 'fullconv':
            return Simple1DfullConvNet(in_channel, hidden_channel, out_channel, True)
    
    def _positional_encoding(self, x):
        length = x.shape[1]
        d_model = 1
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Compute the positional encoding values
        pe = torch.zeros(length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.reshape(1, length)
        
    def forward(self, x, condition):
        # First Affine Transformation with mask1
        x = x + self._positional_encoding(x)
        x11 = x[:,0::2]
        x12 = x[:,1::2]
        s1 = self.scale_net1(torch.cat([x11, condition], dim=1))
        s1 = torch.atan(s1/self.sfactor)*2*self.sfactor/torch.pi
        t1 = self.translate_net1(torch.cat([x11, condition], dim=1))
        x12_trans = x12 #- sigma1
        
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
        x21_trans = x21 # - sigma2

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
        x21_trans = (x21_exp-t2)/torch.exp(s2)
        x21 = x21_trans 
        x2 = torch.empty_like(y)
        x2[:,0::2] = x21
        x2[:,1::2] = x22
                
        # Second inverse Affine Transformation with mask2
        x11 = x2[:,0::2]
        x12_exp = x2[:,1::2]
        s1 = self.scale_net1(torch.cat([x11, condition], dim=1)) 
        s1 = torch.atan(s1/self.sfactor)*2*self.sfactor/torch.pi
        t1 = self.translate_net1(torch.cat([x11, condition], dim=1))
        x12_trans = (x12_exp-t1)/torch.exp(s1)
        x12 = x12_trans # + sigma1
    
        x = torch.empty_like(y)
        x[:,0::2] = x11
        x[:,1::2] = x12
        
        x = x - self._positional_encoding(x)
        return x
    
        
class FCPflowblock(nn.Module): # Fully convolutional time Flow Block
    def __init__(self, num_channels, net_type, sfactor, hidden_dim, condition_dim):
        super(FCPflowblock, self).__init__()
        self.num_channels = num_channels
        self.sfactor = sfactor
        self.input_dim = num_channels
        self.hidden_dim = hidden_dim
        self.output_dim = num_channels
        self.condition_dim = condition_dim
        self.net_type = net_type
        
        # define the layers
        self.actnorm = InvertibleNorm(self.num_channels)
        self.inv_conv = InvertibleWConv(self.num_channels)
        self.coupling_layer = ConditionalAffineCouplingLayer(self.net_type, self.sfactor, self.input_dim, self.hidden_dim, self.condition_dim, self.output_dim)

    def forward(self, x, condition):
        # reshape x to (batch_size, num_channels, 1)
        x = x.unsqueeze(2)
        
        x, log_det1 = self.actnorm(x)
        x, log_det2 = self.inv_conv(x)
        
        # reshape x to (batch_size, num_channels)
        x = x.squeeze(2)
        x, log_det3 = self.coupling_layer(x, condition)
        # x, log_det4 = self.tanh(x)
        
        return x,   log_det2 + log_det3 + log_det1  # + log_det4   
    
    def inverse(self, y, condition):
        x = self.coupling_layer.inverse(y, condition)
        x = x.unsqueeze(2)
        x = self.inv_conv.inverse(x)
        x = self.actnorm.inverse(x)
        x = x.squeeze(2)
        return x
    

class FCPflow(nn.Module): # Fully convolutional time flow
    def __init__(self, num_blocks ,num_channels, net_type, sfactor, hidden_dim, condition_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.net_type = net_type
        self.sfactor = sfactor
        self.input_dim = num_channels
        self.hidden_dim = hidden_dim
        self.output_dim = num_channels
        self.condition_dim = condition_dim
        
        self.blocks = nn.ModuleList([FCPflowblock(self.num_channels, self.net_type,
                                                  self.sfactor, self.hidden_dim,
                                                  self.condition_dim) for _ in range(self.num_blocks)])
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
   
