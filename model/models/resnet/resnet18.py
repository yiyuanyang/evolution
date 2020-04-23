"""
    Content: Resnet18
    Author: Yiyuan Yang
    April. 22nd 2020
"""

import torch
import torch
from Evolution.model.model_components.resnet_components.residual_block import ResidualBlock


class ResNet18(nn.Module):
    """
    My Own Implementation of the Famous ResNet
    """
    def __init__(
        self,
        in_channels
    ):
        super().__init__()
        self.residual_block_1 = ResidualBlock(
            in_channels=in_channels, 
            out_channels=64, 
            kernel_size=3, 
            stride=1, 
            padding=1
            )