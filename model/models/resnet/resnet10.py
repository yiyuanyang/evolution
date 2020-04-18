"""
ResNet
Yiyuan Yang
April 16th 2020
"""

import torch
import torch.nn as nn
from model.model_components.resnet_components.residual_block import ResidualBlock

class ResNet10(nn.Module):
    """
    My Own Implementation of the Famous ResNet
    """
    def __init__(
        self, 
        in_channels,
    ):
        super().__init__()
        self.residual_block_1 = ResidualBlock(in_channels, 32, 3, 1, 1, True, True)
        self.residual_block_2 = ResidualBlock(32, 64, 3, 1, 1, True, True)
        self.residual_block_3 = ResidualBlock(64, 64, 3, 1, 1, True, True)
        self.residual_block_4 = ResidualBlock(64, 128, 3, 1, 1, True, True)
        self.residual_block_5 = ResidualBlock(128, 128, 3, 1, 1, True, True)
    
    def forward(
        self, 
        x
    ):
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.residual_block_4(x)
        x = self.residual_block_5(x)
        return x
        
            

