import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..',)
sys.path.append(_parent_path)

import torch
import torch.nn as nn
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
    
    def define_model(self):
        # Define the model
        
        self.model = fcpflow_model.FCPflow(num_blocks=self.num_blocks, num_channels=self.num_channels, 
                                           hidden_dim=self.hidden_dim, condition_dim=self.condition_dim, 
                                           sfactor=self.sfactor)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('FCPflow Model defined' )
        print('Number of parameters: ', num_params)
        return self.model
    
    def train_model(self, model, num_epochs, train_array, val_array, save_path, device='gpu',train_scheduler=False):
        print('the data has to be numpy arrays which has shape (n_samples, legnth of RLP + length of condition vector)') 
        # Train the model
        self.model = model
        
        # Define the data loaders
        dataloader_train, scaler = tl.create_data_loader(train_array, self.batch_size, True)
        if val_array is None:
            dataloader_test = self.dataloader_train
        else:
            dataloader_test, _ = tl.create_data_loader(val_array, self.batch_size, True)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_max, weight_decay=self.w_decay)
        if train_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, step_size_up=self.lr_step_size, 
                                                        base_lr=self.lr_min, max_lr=self.lr_max,
                                                        cycle_momentum=False)
        else:
            scheduler = None
        tl.train( save_path, model, dataloader_train, optimizer, num_epochs, self.condition_dim, 
                device, scaler, dataloader_test, scheduler, 100, _wandb=False, _plot=True, _save=True)
        
        print('Model trained')
    
    def load_model(self, path):
        # Load the model
        self.model = torch.load(path)
        print('Model loaded')
        return self.model


if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = FCPflowPipeline()
    # Define the learning set
    pipeline._define_learning_set()
    # Define the model
    model = pipeline.define_model()