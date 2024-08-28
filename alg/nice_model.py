import torch.nn as nn
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

class Simple1DfullConvNet(nn.Module):
    def __init__(self, in_c, h_c, out_c, linear = False):
        super().__init__()
        self.in_c = in_c
        self.h_c = h_c
        self.out_c = out_c
        self.linear = linear
        self.bias = False
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
    def __init__(self, sfactor, input_dim, hidden_dim, condition_dim, output_dim):
        super(ConditionalAffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        # self.net_type = net_type
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
        return Simple1DfullConvNet(in_channel, hidden_channel, out_channel, True)
        
    def _create_conditional_network2(self):
        # Network that accepts both input and condition
        in_channel = int(self.input_dim/2)+self.condition_dim
        hidden_channel = self.hidden_dim
        out_channel = int(self.output_dim/2)
        return Simple1DfullConvNet(in_channel, hidden_channel, out_channel, True)

    def forward(self, x, condition):
        # Device 
        device = x.device
        
        # First Affine Transformation with mask1
        x = x # + self._positional_encoding(x).to(device)
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
        log_det_1 = log_det_exp_1 # + log_det_trans_1

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
        
        x = x # - self._positional_encoding(x).to(x.device)
        return x
    
        
class FCPflowblock(nn.Module): # Fully convolutional time Flow Block
    def __init__(self, num_channels, sfactor, hidden_dim, condition_dim):
        super(FCPflowblock, self).__init__()
        self.num_channels = num_channels
        self.sfactor = sfactor
        self.input_dim = num_channels
        self.hidden_dim = hidden_dim
        self.output_dim = num_channels
        self.condition_dim = condition_dim
        # self.net_type = net_type
        
        self.coupling_layer = ConditionalAffineCouplingLayer(self.sfactor, self.input_dim, self.hidden_dim, self.condition_dim, self.output_dim)

    def forward(self, x, condition):
        # reshape x to (batch_size, num_channels, 1)
        x = x.unsqueeze(2)
        x = x.squeeze(2)
        
        x, log_det3 = self.coupling_layer(x, condition)
        # x, log_det4 = self.tanh(x)
        
        return x,   log_det3
    
    def inverse(self, y, condition):
        x = self.coupling_layer.inverse(y, condition)
        x = x.unsqueeze(2)
        x = x.squeeze(2)
        return x

class NICE(nn.Module): # Fully convolutional time flow
    def __init__(self, num_blocks ,num_channels, sfactor, hidden_dim, condition_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        # self.net_type = net_type
        self.sfactor = sfactor
        self.input_dim = num_channels
        self.hidden_dim = hidden_dim
        self.output_dim = num_channels
        self.condition_dim = condition_dim
        
        self.blocks = nn.ModuleList([FCPflowblock(self.num_channels,
                                                  self.sfactor, self.hidden_dim,
                                                  self.condition_dim) for _ in range(self.num_blocks)])
    def forward(self, x, condition):
        log_det = 0
        for block in self.blocks:
            x, log_det1 = block(x, condition)
            log_det += log_det1
        return x, log_det 
    
    def inverse(self, y, condition):
        for block in reversed(self.blocks):
            y = block.inverse(y, condition)
        return y


if __name__ == '__main__':
    # check the model
    # model = FCPflow(8, 48, 0.3, 48, 48).to('cuda')  
    # x = torch.randn(604, 48).to('cuda')

    # condition = torch.randn(604, 48).to('cuda')
    # y, log_det = model(x, condition)
    # x_re = model.inverse(y, condition)
    # print('x: ', x.shape)
    # print('y: ', y.shape)
    # print('difference: ', torch.norm(x - x_re))
    
    # check here the model
    # x = torch.randn(200, 96, 1)
    # inv_layer1 = InvertibleWConv(96)
    # # inv_layer2 = InvertibleWConv_1(96)
    # inv_layer2.w = inv_layer1.w
    # y_1, log_det1 = inv_layer1(x)
    # y_2, log_det2 = inv_layer2(x)
    # x_re_1 = inv_layer1.inverse(y_1)
    # print('x, x_re_1 difference: ', torch.norm(x - x_re_1))
    # x_re_2 = inv_layer2.inverse(y_2)
    # print('x, x_re_2 difference: ', torch.norm(x - x_re_2))
    
    pass

