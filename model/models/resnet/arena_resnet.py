"""
    Content: ResNet here have the ability to adapt and do gradient descent
    Author: Yiyuan Yang
    Date: April. 16th 2020
"""

import torch
import torch.nn as nn
from Evolution.model.model_components.resnet_components.residual_block import BasicBlock, Bottleneck
from Evolution.survival.breed import model_breeding
from Evolution.utils.lineage.lineage_tree import Lineage
from Evolution.utils.weights_understanding.func import func
import copy


model_types = ['resnet10', 'resnet18', 'resnet34', 'resnet50']


def gen_model(model_config):
    if "kernel_sizes" not in model_config.keys():
        model_config["kernel_sizes"] = None
    if "norm_layer" not in model_config.keys():
        model_config["norm_layer"] = None

    model_type = model_config["model_type"]
    if model_type == 'resnet10':
        return resnet10(model_config)
    elif model_type == 'resnet18':
        return resnet18(model_config)
    elif model_type == 'resnet34':
        return resnet34(model_config)
    elif model_type == 'resnet50':
        return resnet50(model_config)


def gen_model_config(
    block,
    layers,
    in_channels,
    image_size,
    num_classes,
    kernel_sizes,
    norm_layer
):
    return {
        "block": block,
        "layers": layers, 
        "in_channels": in_channels,
        "image_size": image_size,
        "num_classes": num_classes,
        "kernel_sizes": kernel_sizes,
        "norm_layer": norm_layer
    }


def gen_model_status_config(
    initial_model = True,
    model_id = 0, # if there has been 400 models existed during evolution, which one is this?
    arena_id = 0, # if there are say 100 total models in arena, which one is this?
    lineage = [None, None],
    age = 0
):
    return {
        "initial_model": initial_model, 
        "model_id": model_id,
        "lineage": lineage, 
        "age": age
    }


class ResNet(nn.Module):

    def __init__(
        self, 
        model_config
    ):
        super(ResNet, self).__init__()
        self._init_config(model_config)
        self._init_model()


    def _init_config(self, model_config):
        # Apply the model config
        self.block = model_config["block"]
        self.layers = model_config["layers"]
        self.in_channels = model_config["in_channels"]
        self.image_size = model_config["image_size"]
        self.num_classes = model_config["num_classes"]
        self.kernel_sizes = model_config["kernel_sizes"]
        self.norm_layer = model_config["norm_layer"]
        if self.kernel_sizes is None:
            self.kernel_sizes = [7,3,3,3,3]
        if self.norm_layer is None:
            self.norm_layer = nn.BatchNorm2d


    def _init_model(self):
        self.dynamic_in_channels = 64
        self.conv1 = nn.Conv2d(
            self.in_channels, 
            self.dynamic_in_channels, 
            self.kernel_sizes[0],
            stride=2,
            padding = self.kernel_sizes[0] // 2,
            bias = False)
        self.bn1 = self.norm_layer(self.dynamic_in_channels)
        self.relu = nn.ReLU(inplace = True)
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
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * self.block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def breed_net(
        self, 
        other_net, 
        logger,
        policy = "average",
        max_weight_mutation = 0.00005
    ):
        """
            Given a right hand side net, breed a new net that with some policy,
            combines the two
        """
        logger.log("Parent 1:")
        self.log_weights(logger)
        logger.log("Parent 2:")
        other_net.log_weights(logger)

        with torch.no_grad():
            temp = self.conv1
            self.conv1 = model_breeding.breed_conv(
                left_conv=self.conv1,
                right_conv=other_net.conv1,
                in_channels=self.in_channels,
                out_channels=64,
                policy=policy,
                max_weight_mutation=max_weight_mutation
            )
            del temp
            temp = self.layer1
            self.layer1 = self.layer1.breed(other_block=other_net.layer1, policy=policy, max_weight_mutation=max_weight_mutation)
            del temp
            temp = self.layer2
            self.layer2 = self.layer1.breed(other_block=other_net.layer2, policy=policy, max_weight_mutation=max_weight_mutation)
            del temp
            temp = self.layer3
            self.layer3 = self.layer1.breed(other_block=other_net.layer3, policy=policy, max_weight_mutation=max_weight_mutation)
            del temp
            temp = self.layer4
            self.layer4 = self.layer1.breed(other_block=other_net.layer4, policy=policy, max_weight_mutation=max_weight_mutation)
            del temp

            # We are not changing the fully connected layer
            #self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            #self.fc = nn.Linear(512 * self.block.expansion, self.num_classes)
            logger.log("Child")
            self.log_weights(logger)
            return self


    def log_weights(self, logger):
        logger.log("Logging resnet weights")
        func.conv_statistics(self.conv1, "conv1", logger)
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
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    

def resnet10(model_config):
    model_config["block"] = BasicBlock
    model_config["layers"] = [1,1,1,1]
    return ResNet(model_config)


def resnet18(model_config):
    model_config["block"] = BasicBlock
    model_config["layers"] = [2,2,2,2]
    return ResNet(model_config)


def resnet34(model_config):
    model_config["block"] = BasicBlock
    model_config["layers"] = [3,4,6,3]
    return ResNet(model_config)


def resnet50(model_config):
    model_config["block"] = Bottleneck
    model_config["layers"] = [3,4,6,3]
    return ResNet(model_config)


        
            

