"""
ResNet
Yiyuan Yang
April 16th 2020
"""

import torch
import torch.nn as nn
from model.model_components.resnet_components import residual_block

config_1 = {
    "out_channel": 32,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "bias": True
}

config_2 = {
    "in_channel": 32,
    "out_channel": 64,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "bias": True
}

config_3 = {
    "in_channel": 64,
    "kernel_size": 128,
    "stride": 1,
    "padding": 1,
    "bias": True
}

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
