"""
    Content: ResNet10
    Author: Yiyuan Yang
    Date: April. 16th 2020
"""

import torch
import torch.nn as nn
from Evolution.model.model_components.resnet_components.residual_block import BasicBlock, Bottleneck


model_types = ['resnet10', 'resnet18', 'resnet34', 'resnet50']


def gen_model(
    config,
    kernel_sizes = None, 
    norm_layer=None
):
    model_type = config["model_type"]
    in_channels = config["in_channels"]
    image_size = config["image_size"]
    num_classes = config["num_classes"]
    if model_type == 'resnet10':
        return resnet10(in_channels, image_size, num_classes, kernel_sizes, norm_layers)
    elif model_type == 'resnet18':
        return resnet18(in_channels, image_size, num_classes, kernel_sizes, norm_layers)
    elif model_type == 'resnet34':
        return resnet34(in_channels, image_size, num_classes, kernel_sizes, norm_layers)
    elif model_type == 'resnet50':
        return resnet50(in_channels, image_size, num_classes, kernel_sizes, norm_layers)



class ResNet(nn.Module):

    def __init__(
        self, 
        block, 
        layers, 
        in_channels,
        image_size,
        num_classes, 
        kernel_sizes = [7,3,3,3,3], 
        norm_layer=None):

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.__norm_layer = norm_layer
        self.num_layers = len(layers)
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(
            in_channels, 
            self.in_channels, 
            kernel_sizes[0],
            stride=2,
            padding = kernel_sizes[0] // 2,
            bias = False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.num_layers >= 1:
            self.layer1 = self._make_layer(
                block=block, 
                channels=64, 
                num_blocks=layers[0],
                kernel_size=kernel_sizes[1],
                stride=1
            )
        if self.num_layers >= 2:
            self.layer2 = self._make_layer(
                block=block, 
                channels=128, 
                num_blocks=layers[1],
                kernel_size=kernel_sizes[2],
                stride=2
            )
        if self.num_layers >= 3:
            self.layer3 = self._make_layer(
                block=block, 
                channels=256, 
                num_blocks=layers[2],
                kernel_size=kernel_sizes[3],
                stride=2
            )
        if self.num_layers >= 4:
            self.layer4 = self._make_layer(
                block=block, 
                channels=512, 
                num_blocks=layers[3],
                kernel_size=kernel_sizes[4],
                stride=2
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    
    def log_weights(self, logger):
        logger.log("Logging resnet weights")
        if self.num_layers > 1:
            self.layer1.log_weights(logger)
        if self.num_layers > 2:
            self.layer2.log_weights(logger)
        if self.num_layers >= 3:
            self.layer3.log_weights(logger)
        if self.num_layers >= 4:
            self.layer4.log_weights(logger)        



    def _make_layer(
        self, 
        block, 
        channels, 
        num_blocks, 
        kernel_size,
        stride
    ):
        norm_layer = self.__norm_layer
        
        layers = []
        layers.append(
            block(
                in_channels=self.in_channels,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer
            )
        )
        self.in_channel = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_layer=norm_layer
                )
        )
        return nn.Sequential(*layers)


    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.num_layers > 1:
            x = self.layer1(x)
        if self.num_layers > 2:
            x = self.layer2(x)
        if self.num_layers >= 3:
            x = self.layer3(x)
        if self.num_layers >= 4:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x
    

def resnet10(        
    in_channels,
    image_size,
    num_classes, 
    kernel_sizes = None, 
    norm_layer=None
):
    return ResNet(
        block=BasicBlock, 
        layers=[1,1,1,1],
        in_channels=in_channels,
        image_size=image_size,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        norm_layer=norm_layer)


def resnet18(        
    in_channels,
    image_size,
    num_classes, 
    kernel_sizes = None, 
    norm_layer=None
):
    return ResNet(
        block=BasicBlock, 
        layers=[2,2,2,2],
        in_channels=in_channels,
        image_size=image_size,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        norm_layer=norm_layer)


def resnet34(        
    in_channels,
    image_size,
    num_classes, 
    kernel_sizes = None, 
    norm_layer=None
):
    return ResNet(
        block=BasicBlock, 
        layers=[3,4,6,3],
        in_channels=in_channels,
        image_size=image_size,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        norm_layer=norm_layer)


def resnet50(        
    in_channels,
    image_size,
    num_classes, 
    kernel_sizes = None, 
    norm_layer=None
):
    return ResNet(
        block=Bottleneck, 
        layers=[3,4,6,3],
        in_channels=in_channels,
        image_size=image_size,
        num_classes=num_classes,
        kernel_sizes=kernel_sizes,
        norm_layer=norm_layer)


        
            

