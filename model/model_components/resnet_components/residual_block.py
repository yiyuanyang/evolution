"""
    Content: Basic Residual Block for ResNet
    Author: Yiyuan Yang
    Date: April 16th 2020
"""

import torch.nn as nn
import torch

class BasicBlock(torch.nn.Module):
    """
        Re-implementing a basic residual block for shallow resnet
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, dilation=1, normalization=None):

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        assert_downsample(in_channels, out_channels, stride)

        self.conv1 = conv_layer(in_channels, out_channels, kernel_size, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.conv_layer(out_channels, out_channels, kernel_size)
        self.bn2 = norm_layer(out_channels)

        if stride !=1 :
            self.downsample = nn.Sequential(
                Conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels)
            )

    
    def forward(
        self, 
        x
    ):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.stride != 1:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, dilation=1, norm_layer=None):
        if norm_layer is None:
            norm_layer=nn.BatchNorm2d
        bottleneck_channels = int(out_channels / 4)

        self.conv1 = Conv1x1(in_channels, bottleneck_channels)
        self.bn1 = norm_layer(bottleneck_channels)
        self.conv2 = conv_layer(
            in_channels=bottleneck_channels, 
            out_channels=bottleneck_channels,
            stride=stride,
            dilation=dilation)
        self.bn2 = norm_layer(bottleneck_channels)
        self.conv3 = Conv1x1(bottleneck_channels, out_channels)
        self.bn3 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride !=1 :
            self.downsample = nn.Sequential(
                Conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels)
            )
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride != 1:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out



        


def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
    return nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        stride=stride,
        padding=dilation,
        bias=False,
        dilation=dilation)

def Conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )

def assert_downsample(in_channels, out_channels, stride):
    assert (
        (stride == 1 and in_channels == out_channels) 
        or (stride != 1 and in_channels != out_channels),
        "Inconsistent downsampling with in/out channels:" + 
        " {in_channels},{out_channels} and stride {stride}".format(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
        ))



