"""
    Content: Basic Residual Block for ResNet
    Author: Yiyuan Yang
    Date: April 16th 2020
"""

import torch.nn as nn
import torch

class ResidualBlock(torch.nn.Module):
    """
        A Residual Block
    """

    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias = True,
    ):
        super(ResidualBlock, self).__init__()
        if stride != 1:
            self.downsample = True

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if self.downsample:
            self.downsample_block = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
    
    def forward(
        self, 
        x
    ):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.downsample:
            out += self.downsample_block(x)
        else:
            out += x

        return out


