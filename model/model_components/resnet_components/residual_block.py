"""
    Content: Basic Residual Block for ResNet
    Author: Yiyuan Yang
    Date: April 16th 2020
"""

import torch.nn as nn
import torch
from Evolution.utils.breed import model_breeding


class BasicBlock(torch.nn.Module):
    """
        Re-implementing a basic residual block for shallow resnet
    """

    expansion = 1

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size=3,
                 stride=1,
                 norm_layer=None):

        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = norm_layer
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d

        self.conv1 = conv_layer(self.in_channels, self.channels,
                                self.kernel_size, self.stride)
        self.bn1 = self.norm_layer(self.channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(self.channels, self.channels, self.kernel_size)
        self.bn2 = self.norm_layer(self.channels)

        self.downsample = False
        if stride != 1:
            self.downsample = True
            self.downsample_block = nn.Sequential(
                Conv1x1(self.in_channels, self.channels, self.stride),
                self.norm_layer(self.channels))

    def log_weights(self, logger):
        logger.log_residual_block_statistics(self)

    def forward(self, x):
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

    def breed(self,
              other_block,
              policy="average",
              max_weight_mutation=0.00005):
        """
            Generate a new Basic Block based on current block
            and input block
        """

        new_block = BasicBlock(self.in_channels, self.channels,
                               self.kernel_size, self.stride, self.norm_layer)

        new_block.conv1 = model_breeding.breed_conv(
            left_conv=self.conv1,
            right_conv=other_block.conv1,
            in_channels=self.in_channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            policy=policy,
            max_weight_mutation=max_weight_mutation)
        new_block.conv2 = model_breeding.breed_conv(
            left_conv=self.conv2,
            right_conv=other_block.conv2,
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            policy=policy,
            max_weight_mutation=max_weight_mutation)

        if self.downsample:
            new_block.downsample_block = nn.Sequential(
                model_breeding.breed_conv(
                    left_conv=self.downsample_block[0],
                    right_conv=other_block.downsample_block[0],
                    in_channels=self.in_channels,
                    out_channels=self.channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    policy=policy,
                    max_weight_mutation=max_weight_mutation),
                self.norm_layer(self.channels))

        return new_block


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size=3,
                 stride=1,
                 norm_layer=None):

        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_layer = norm_layer

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = Conv1x1(in_channels, channels)
        self.bn1 = norm_layer(channels)
        self.conv2 = conv_layer(in_channels=channels,
                                out_channels=channels,
                                stride=stride)
        self.bn2 = norm_layer(channels)
        self.conv3 = Conv1x1(channels, channels * self.expansion)
        self.bn3 = norm_layer(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = False
        if stride != 1:
            self.downsample = True
            self.downsample_block = nn.Sequential(
                Conv1x1(in_channels, channels * self.expansion, stride),
                norm_layer(channels * self.expansion))
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

    def breed(self,
              other_block,
              policy="average",
              max_weight_mutation=0.00005):
        """
            Generate a new Basic Block based on current block
            and input block
        """

        new_block = Bottleneck(self.in_channels, self.channels,
                               self.kernel_size, self.stride, self.norm_layer)

        new_block.conv1 = model_breeding.breed_conv(
            left_conv=self.conv1,
            right_conv=other_block.conv1,
            in_channels=self.in_channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            policy=policy,
            max_weight_mutation=max_weight_mutation)

        new_block.conv2 = model_breeding.breed_conv(
            left_conv=self.conv2,
            right_conv=other_block.conv2,
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            policy=policy,
            max_weight_mutation=max_weight_mutation)

        new_block.conv3 = model_breeding.breed_conv(
            left_conv=self.conv3,
            right_conv=other_block.conv3,
            in_channels=self.channels,
            out_channels=self.channels * self.expansion,
            kernel_size=self.kernel_size,
            stride=self.stride,
            policy=policy,
            max_weight_mutation=max_weight_mutation)

        if self.downsample:
            new_block.downsample_block = nn.Sequential(
                model_breeding.breed_conv(
                    left_conv=self.downsample_block,
                    right_conv=other_block.downsample_block,
                    in_channels=self.in_channels,
                    out_channels=self.channels * self.expansion,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    policy=policy,
                    max_weight_mutation=max_weight_mutation),
                self.norm_layer(self.channels))

        return new_block


def conv_layer(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=kernel_size // 2,
                     bias=False)


def Conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
