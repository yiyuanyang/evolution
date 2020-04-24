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

    def __init__(self, in_channels, channels, kernel_size=3, 
                 stride=1, norm_layer=None):

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        assert_downsample(in_channels, channels, stride)

        self.conv1 = conv_layer(in_channels, channels, kernel_size, stride)
        self.bn1 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.conv_layer(channels, channels, kernel_size)
        self.bn2 = norm_layer(channels)

        if stride !=1 :
            self.downsample = True
            self.downsample_block = nn.Sequential(
                Conv1x1(in_channels, channels, stride),
                norm_layer(channels)
            )

    def log_weights(self, logger):
        logger.log_residual_block_statistics(self)

    
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
        
        if self.downsample:
            identity = self.downsample_block(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, kernel_size=3, 
                 stride=1, norm_layer=None):
            
        self.expansion = 4

        if norm_layer is None:
            norm_layer=nn.BatchNorm2d

        self.conv1 = Conv1x1(in_channels, channels)
        self.bn1 = norm_layer(channels)
        self.conv2 = conv_layer(
            in_channels=channels, 
            out_channels=channels,
            stride=stride)
        self.bn2 = norm_layer(channels)
        self.conv3 = Conv1x1(channels, channels * self.expansion)
        self.bn3 = norm_layer(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if stride !=1 :
            self.downsample = True
            self.downsample_block = nn.Sequential(
                Conv1x1(
                    in_channels, 
                    channels * self.expansion, 
                    stride
                ),
                norm_layer(channels * self.expansion)
            )
        self.stride = stride

    def log_weights(self, logger):
        logger.log_residual_block_statistics(self)


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
            identity = self.downsample_block(x)
        
        out += identity
        out = self.relu(out)

        return out



        


def conv_layer(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        stride=stride,
        padding=kernel_size//2,
        bias=False)

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



