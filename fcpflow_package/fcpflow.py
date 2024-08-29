import os
import sys
_parent_path =os.path.join(os.path.dirname(__file__), '..','..','..')
sys.path.append(_parent_path)

import torch
import torch.nn as nn
from typing import Union

import alg.models_fcpflow_lin as fcpflow_model
import tools.tools_train as tl

class FCPflowPipeline:
    def __init__(self,
        #--------Define the model--------
        num_blocks: Union[int, float] = 6, # number of blocks in the model
        num_channels: Union[int, float] = 48, # resolution of the time series 
        hidden_dim: Union[int, float] = 64, # dimension of the hidden layers
        condition_dim: Union[int, float] = 9, # dimension of the condition vector
        sfactor: Union[int, float] = 0.7, 
        
        #--------Define the training--------  
        lr_min: Union[int, float] = 0.0001,
        lr_max: Union[int, float] = 0.0005,
        lr_step_size: Union[int, float] = 2000,
        w_decay: Union[int, float] = 0.0,
        batch_size: Union[int, float] = 2000
        ):
        
        # Assign parameters to instance variables
        self.num_blocks = num_blocks
        self.sfactor = sfactor
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.condition_dim = condition_dim
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_step_size = lr_step_size
        self.w_decay = w_decay
        self.batch_size = batch_size
        
    def define_model(self):
        # Define the model
        self.model = fcpflow_model.FCPflow(num_blocks=self.num_blocks, num_channels=self.num_channels, 
                                           hidden_dim=self.hidden_dim, condition_dim=self.condition_dim, 
                                           sfactor=self.sfactor)
        return self.model

if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = FCPflowPipeline()
    # Define the model
    model = pipeline.define_model()
    print(model)