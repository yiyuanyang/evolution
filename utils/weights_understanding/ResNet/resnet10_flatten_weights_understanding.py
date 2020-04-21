"""
    Content: This is used to calculate statistical distribution
    of the weights inside a resnet10 model
    Author: Yiyuan Yang
    Date: April. 20th 2020
"""

import numpy as np
import torch
from Evolution.utils.weights_understanding.func import func

def calculate_statistics(model, name, logger):
    residual_block_1 = model.ResNet_Skeleton.residual_block_1
    residual_block_2 = model.ResNet_Skeleton.residual_block_2
    residual_block_3 = model.ResNet_Skeleton.residual_block_3
    residual_block_4 = model.ResNet_Skeleton.residual_block_4
    residual_block_5 = model.ResNet_Skeleton.residual_block_5
    final_layer = model.final_layer
    func.residual_block_statistics(residual_block_1, "Res Block 1", logger)
    func.residual_block_statistics(residual_block_2, "Res Block 2", logger)
    func.residual_block_statistics(residual_block_3, "Res Block 3", logger)
    func.residual_block_statistics(residual_block_4, "Res Block 4", logger)
    func.residual_block_statistics(residual_block_5, "Res Block 5", logger)
    func.conv_statistics(final_layer, "final layer", logger)
    
    