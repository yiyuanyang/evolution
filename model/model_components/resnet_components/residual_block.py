"""
    Content: Basic Residual Block for ResNet
    Author: Yiyuan Yang
    Date: April 16th 2020
"""

import torch.nn as nn
import torch

class ResidualBlock(torch.nn.Module):
    def __init__(
        self, 
        config
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = self._conv2d(config)
        self.conv2 = self._conv2d(config)
        self.bn1 = nn.BatchNorm2d()
        self.bn2 = nn.BatchNorm2d()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
    
    def forward(
        self, 
        x
    ):
        skip = 


    def _conv2d(
        self, 
        config
    ):
        return nn.Conv2d(
            in_channels=config["in_channel"],
            out_channels=config["out_channel"],
            kernel_size=config["kernel_size"],
            stride=config["stride"],
            padding=config["padding"],
            bias = config["bias"],
            padding_mode=config["padding_mode"]
        )
    
    def _downsample(
        self,
        config
    ):
        re

