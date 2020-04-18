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
        self.bn1 = nn.BatchNorm2d(config['in_channel'])
        self.bn2 = nn.BatchNorm2d(config['out_channel'])
        self.relu = nn.ReLU()
    
    def forward(
        self, 
        x
    ):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += x
        out = self.relu(out)

        return out


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

