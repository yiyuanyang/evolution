"""
    Content: Downsample Block For ResNet
    Author: Yiyuan Yang
    Date: April 17th 2020
"""

import torch.nn as nn
import torch 

class DownsampleBlock(torch.nn.Module):
    def __init__(
        self, 
        config
    ):
        super(DownsampleBlock, self).__init__()
        self.downsample = self._conv2d(config)

    
    def forward(
        self, 
        x
    ):
        return self.downsample(x)

    def _conv2d(
        self, 
        config
    ):
        return nn.Conv2d(
            in_channels=config["in_channel"],
            out_channels=config["out_channel"],
            kernel_size=config["kernel_size"],
            stride=config["stride"] * 2,
            padding=config["padding"],
            bias = config["bias"],
        )

