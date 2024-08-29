import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..',)
sys.path.append(_parent_path)

import torch
import numpy as np
from typing import Union
import warnings
warnings.filterwarnings("ignore", message=".*torch.qr is deprecated.*")

import alg.models_fcpflow_lin as fcpflow_model
import tools.tools_train as tl

class FCPflowPipeline:
    def __init__(self,
        #--------Define the model--------
        num_blocks: Union[int, float] = 6, # number of blocks in the model
        num_channels: Union[int, float] = 48, # resolution of the time series 
        hidden_dim: Union[int, float] = 64, # dimension of the hidden layers
        condition_dim: Union[int, float] = 9, # dimension of the condition vector, large than 1
        sfactor: Union[int, float] = 0.7, 
        ):
        
        # Assign parameters to instance variables
        self.num_blocks = num_blocks
        self.sfactor = sfactor
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.condition_dim = condition_dim
        self._define_learning_set()

    def _define_learning_set(self,
        #--------Define the training--------  
        lr_min: Union[int, float] = 0.0001,
        lr_max: Union[int, float] = 0.0005,
        lr_step_size: Union[int, float] = 2000,
        w_decay: Union[int, float] = 0.0,
        batch_size: Union[int, float] = 2000):
        
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_step_size = lr_step_size
        self.w_decay = w_decay
        self.batch_size = batch_size
        print('Learning set defined')
    
    def _define_model(self):
        # Define the model
        
        self.model = fcpflow_model.FCPflow(num_blocks=self.num_blocks, num_channels=self.num_channels, 
                                           hidden_dim=self.hidden_dim, condition_dim=self.condition_dim, 
                                           sfactor=self.sfactor)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('FCPflow Model defined' )
        print('Number of parameters:', num_params)

    
    def data_processing(self, train_array, val_array):
        # Define the data loaders
        self.dataloader_train, self.scaler = tl.create_data_loader(train_array, self.batch_size, True)
        if val_array is None:
            self.dataloader_test = self.dataloader_train
        else:
            self.dataloader_test, _ = tl.create_data_loader(val_array, self.batch_size, True)
            
    
    def train_model(self, num_epochs, train_array, val_array, save_path, device='gpu', train_scheduler=False):
        print('the data has to be numpy arrays which has shape (n_samples, legnth of RLP + length of condition vector)') 
        # Train the model
        self.data_processing(train_array, val_array)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_max, weight_decay=self.w_decay)
        if train_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up=self.lr_step_size, 
                                                        base_lr=self.lr_min, max_lr=self.lr_max,
                                                        cycle_momentum=False)
        else:
            scheduler = None
        tl.train( save_path,self.model, self.dataloader_train, optimizer, num_epochs, self.condition_dim, 
                device, self.scaler, self.dataloader_test, scheduler, 100, _wandb=False, _plot=True, _save=True)
        
        print('Model trained')
    
    def load_model(self, path):
        # Load the model, load dict
        self.model.load_state_dict(torch.load(path, weights_only=True))
        print('Model loaded')
        return self.model
    
    def sample_from_trained_model(self, condition_array, device = 'cpu'):
        self.model.eval()
        # Scale the condition array
        _mid = torch.randn((condition_array.shape[0], self.num_channels)).cpu().numpy()
        _condion = np.hstack((_mid, condition_array))
        condition_array = self.scaler.transform(_condion)[:,-self.condition_dim:]
        
        z = torch.randn(condition_array.shape[0], self.num_channels).to(device)
        condition_array = torch.tensor(condition_array).to(device)
        
        gen_test = self.model.inverse(z, condition_array)
        re_data = torch.cat((gen_test, condition_array), dim=1)
        re_data = re_data.detach()
        re_data = self.scaler.inverse_transform(re_data)
        return re_data


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Initialize the pipeline
    pipeline = FCPflowPipeline(
        num_blocks = 2, # number of blocks in the model
        num_channels = 24, # resolution of the time series 
        hidden_dim = 12, # dimension of the hidden layers
        condition_dim = 1, # dimension of the condition vector, large than 1
        sfactor = 0.3, 
    )
    
    save_path = 'fcpflow_package' 
      
    pipeline._define_learning_set() # Define the learning set
    pipeline._define_model() # Define the model

    # Prepare the data
    data_path = 'data/nl_data_1household.csv'
    np_array = pd.read_csv(data_path).iloc[:,3:-2].values
    np_array = np_array[~pd.isna(np_array).any(axis=1)]
    np_array = np.hstack((np_array, np.ones((np_array.shape[0], 1))))
    
    pipeline.train_model(1, np_array, None, save_path, device='cpu', train_scheduler=False)
    
    model_path = save_path +'FCPflow_model.pth'
    model = pipeline.load_model(model_path)
    pipeline.data_processing(np_array, None)
    
    # Sample from the trained model
    condition_array = np_array[:10,-1:]
    samples = pipeline.sample_from_trained_model(condition_array, device = 'cpu')
    
    # plot the samples
    print(samples.shape)
    plt.plot(samples[:,:-1].T)
    plt.savefig(save_path+'sample.png')
    
    
    