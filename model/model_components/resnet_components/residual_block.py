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
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias = True,
        downsample = False,
    ):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        if self.downsample:
            self.downsample_block = nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride * 2,
                padding=1
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
        
        out += x
        out = self.relu(out)

        if self.downsample:
            out = self.downsample_block(out)

        return out


