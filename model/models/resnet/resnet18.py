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
        self.residual_block_1 = ResidualBlock(in_channels, 32, 3, 1, 1, True)