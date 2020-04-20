"""
ResNet
Yiyuan Yang
April 16th 2020
"""

import torch
import torch.nn as nn
import numpy as np 
from Evolution.model.models.resnet.resnet10 import ResNet10

class ResNet10Flatten(nn.Module):
    """
    My Own Implementation of the Famous ResNet
    """
    def __init__(
        self, 
        in_channels,
        image_size,
        num_classes
    ):
        super().__init__()
        final_image_size = int(image_size / (2**5))
        self.ResNet_Skeleton = ResNet10(in_channels)
        self.final_layer = nn.Conv2d(
            in_channels=128, 
            out_channels=num_classes,
            kernel_size=final_image_size,
            stride=1,
            padding=0)
        self.final_batchnorm = nn.BatchNorm2d(
            num_classes
        )
    
    def forward(
        self, 
        x
    ):
        x = self.ResNet_Skeleton(x)
        x = self.final_layer(x)
        x = self.final_batchnorm(x)
        x = x.view(x.size(0), -1)
        return x
        
            

