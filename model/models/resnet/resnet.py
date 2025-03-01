"""
    Content: ResNet
    Author: Yiyuan Yang
    Date: April. 16th 2020
"""

import torch
import torch.nn as nn
from yyycode.model.model_components.resnet_components.residual_block import BasicBlock, Bottleneck


model_types = ['resnet10', 'resnet18', 'resnet34', 'resnet50']


def gen_model(
    config,
    norm_layer=None
):
    if "kernel_sizes" in config.keys():
        kernel_sizes = config["kernel_sizes"]
    else:
        kernel_sizes = None
    model_type = config["model_type"]
    in_channels = config["in_channels"]
    image_size = config["image_size"]
    num_classes = config["num_classes"]
    if model_type == 'resnet10':
        return resnet10(in_channels, image_size, num_classes, kernel_sizes, norm_layer)
    elif model_type == 'resnet18':
        return resnet18(in_channels, image_size, num_classes, kernel_sizes, norm_layer)
    elif model_type == 'resnet34':
        return resnet34(in_channels, image_size, num_classes, kernel_sizes, norm_layer)
    elif model_type == 'resnet50':
        return resnet50(in_channels, image_size, num_classes, kernel_sizes, norm_layer)


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        in_channels,
        image_size,
        num_classes,
        kernel_sizes,
        norm_layer=None
    ):

        super(ResNet, self).__init__()

        self.block = block
        self.layers = layers
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.norm_layer = norm_layer
        if kernel_sizes is None:
            kernel_sizes = [7, 3, 3, 3, 3]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.dynamic_in_channels = 64

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.dynamic_in_channels,
            self.kernel_sizes[0],
            stride=2,
            padding=self.kernel_sizes[0] // 2,
            bias=False)
        self.bn1 = norm_layer(self.dynamic_in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block=self.block,
            channels=64,
            num_blocks=self.layers[0],
            kernel_size=self.kernel_sizes[1],
            stride=1
        )
        self.layer2 = self._make_layer(
            block=self.block,
            channels=128,
            num_blocks=self.layers[1],
            kernel_size=self.kernel_sizes[2],
            stride=2
        )
        self.layer3 = self._make_layer(
            block=self.block,
            channels=256,
            num_blocks=self.layers[2],
            kernel_size=self.kernel_sizes[3],
            stride=2
        )
        self.layer4 = self._make_layer(
            block=self.block,
            channels=512,
            num_blocks=self.layers[3],
            kernel_size=self.kernel_sizes[4],
            stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def log_weights(self, logger):
        logger.log("Logging resnet weights")
        for i in range(len(self.layer1)):
            self.layer1[i].log_weights(logger)
        for i in range(len(self.layer2)):
            self.layer2[i].log_weights(logger)
        for i in range(len(self.layer3)):
            self.layer3[i].log_weights(logger)
        for i in range(len(self.layer4)):
            self.layer4[i].log_weights(logger)

    def _make_layer(
        self,
        block,
        channels,
        num_blocks,
        kernel_size,
        stride
    ):

        layers = []
        layers.append(
            block(
                in_channels=self.dynamic_in_channels,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=self.norm_layer
            )
        )
        self.dynamic_in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=self.dynamic_in_channels,
                    channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_layer=self.norm_layer
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet10(
    in_channels,
    image_size,
    num_classes,
    kernel_sizes=None,
    norm_layer=None
):
    return ResNet(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        in_channels=in_channels,
        image_size=image_size,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        norm_layer=norm_layer)


def resnet18(
    in_channels,
    image_size,
    num_classes,
    kernel_sizes=None,
    norm_layer=None
):
    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
        image_size=image_size,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        norm_layer=norm_layer)


def resnet34(
    in_channels,
    image_size,
    num_classes,
    kernel_sizes=None,
    norm_layer=None
):
    return ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        image_size=image_size,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        norm_layer=norm_layer)


def resnet50(
    in_channels,
    image_size,
    num_classes,
    kernel_sizes=None,
    norm_layer=None
):
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        image_size=image_size,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        norm_layer=norm_layer)
