"""
ResNet
Yiyuan Yang
April 16th 2020
"""

import torch
import torch.nn as nn

class ResNet(nn.Module):
    """
    My Own Implementation of the Famous ResNet
    """
    def __init__(
        self, 
        model_config
    ):
        super().__init__()
    
    def forward(
        self, 
        data_batch
    ):
        return data_batch
