"""
    Content: ResNet10
    Author: Yiyuan Yang
    Date: April. 16th 2020
"""

import torch
import torch.nn as nn
from Evolution.model.model_components.resnet_components.residual_block import ResidualBlock

class ResNet10(nn.Module):
    """
    My Own Implementation of the Famous ResNet
    """
    def __init__(
        self, 
        in_channels,
    ):
        super().__init__()
        self.residual_block_1 = ResidualBlock(
            in_channels=in_channels, 
            out_channels=64, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.residual_block_2 = ResidualBlock(
            in_channels=64, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.residual_block_3 = ResidualBlock(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.residual_block_4 = ResidualBlock(
            in_channels=128, 
            out_channels=128, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.residual_block_5 = ResidualBlock(
            in_channels=128, 
            out_channels=256, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
    
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
        
            

