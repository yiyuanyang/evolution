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
        final_image_size = int(image_size / (2**3))
        self.ResNet_Skeleton = ResNet10(in_channels)
        self.final_layer = nn.Linear(256 * final_image_size * final_image_size, 10)
    
    def forward(
        self, 
        x
    ):
        x = self.ResNet_Skeleton(x)
        x = x.view(x.size(0), -1)
        x = self.final_layer(x)
        return x
        
            

