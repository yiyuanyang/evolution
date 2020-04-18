"""
ResNet
Yiyuan Yang
April 16th 2020
"""

import torch
import torch.nn as nn
import numpy as np 
from model.models.resnet.resnet10 import ResNet10

class ResNet10_Flatten(nn.Module):
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
        final_image_size = int(image_size / (2**4))
        self.ResNet_Skeleton = ResNet10(in_channels)
        self.final_layer = nn.Conv2d(
            in_channels=128, 
            out_channels=num_classes,
            kernel_size=final_image_size,
            stride=1,
            padding=0)
    
    def forward(
        self, 
        x
    ):
        x = self.ResNet_Skeleton(x)
        x = self.final_layer(x)
        x = x.view()
        
            

