"""
    Content: This is used to calculate statistical distribution
    of the weights inside a resnet10 model
    Author: Yiyuan Yang
    Date: April. 20th 2020
"""

import numpy as np
import torch

def calculate_statistics(model, name):
    residual_block_1 = model.ResNet_Skeleton.residual_block_1
    residual_block_2 = model.ResNet_Skeleton.residual_block_2
    residual_block_3 = model.ResNet_Skeleton.residual_block_3
    residual_block_4 = model.ResNet_Skeleton.residual_block_4
    residual_block_5 = model.ResNet_Skeleton.residual_block_5
    final_layer = model.final_layer

    msg = "Statistics Of Model By Layer\n=======\n{name}".format(name=name)
    block_msg = "\nStatistics Of Model By Block\n======\n"
    msg, block_msg = analyze_residual_block(
        residual_block_1, 
        "Residual Block 1".format(name=name),
        msg,
        block_msg)
    msg, block_msg = analyze_residual_block(
        residual_block_2, 
        "Residual Block 2".format(name=name),
        msg,
        block_msg)
    msg, block_msg = analyze_residual_block(
        residual_block_3, 
        "Residual Block 3".format(name=name),
        msg,
        block_msg)
    msg, block_msg = analyze_residual_block(
        residual_block_4, 
        "Residual Block 4".format(name=name),
        msg,
        block_msg)
    msg, block_msg = analyze_residual_block(
        residual_block_5, 
        "Residual Block 5".format(name=name),
        msg,
        block_msg)
    msg, block_msg = analyze_conv_layer(
        final_layer, 
        "{name} final layer".format(name=name),
        msg,
        block_msg)
    return msg + block_msg

def analyze_residual_block(residual_block, name, msg, block_msg):
    conv1 = residual_block.conv1
    conv2 = residual_block.conv2
    conv_short = residual_block.conv_short
    downsample_block = residual_block.downsample_block

    msg += "\n=======\n{name}".format(name=name)
    msg_1, weight_1, bias_1 = analyze_conv_layer(
        conv1, 
        "Conv Layer 1".format(name=name))
    msg_2, weight_2, bias_2 = analyze_conv_layer(
        conv2, 
        "Conv Layer 2".format(name=name))
    msg_3, weight_3, bias_3 = analyze_conv_layer(
        conv_short, 
        "Conv Short".format(name=name))
    msg_4, weight_4, bias_4 = analyze_conv_layer(
        downsample_block, 
        "Downsample Block".format(name=name))

    weight = torch.cat(
        tensors=(
            torch.flatten(weight_1), 
            torch.flatten(weight_2), 
            torch.flatten(weight_3), 
            torch.flatten(weight_4)
        ),
        dim=0
    )

    bias = torch.cat(
        tensors=(
            torch.flatten(bias_1), 
            torch.flatten(bias_2), 
            torch.flatten(bias_3), 
            torch.flatten(bias_4)
        ),
        dim=0
    )

    block_msg += statistics(
        weight, 
        "{name}_weight".format(name=name)
        ) + \
        statistics(
            weight, 
            "{name}_bias".format(name=name)
        )

    return msg + msg_1 + msg_2 + msg_3 + msg_4, block_msg

def analyze_conv_layer(conv, name):
    weight = conv.weight.data

    msg = "\n=======\n{name}".format(name=name)
    msg += conv_statistics(
        weight, 
        "Weights".format(name=name)
    )
    
    if conv.bias is not None:
        bias = conv.bias.data
        msg += statistics(
            bias
        )
    
    grad = conv.weight.grad
    msg += conv_statistics(
        grad, 
        "Gradient".format(name=name)
    )
    
    return msg, torch.flatten(weight), torch.flatten(conv.bias.data)

def conv_statistics(weight, name):
    """
        Weight Here can also be gradient
    """
    per_kernel_average = torch.mean(weight, dim=(1,2,3))
    num_weights = torch.flatten(weight).shape[0]
    msg = "\n=======\n{name} with {num_weights} of weights".format(
        name=name, 
        num_weights=num_weights,
    )
    msg += statistics(per_kernel_average)
    return msg

def statistics(arr):
    max_value = torch.max(arr)
    min_value = torch.min(arr)
    range_value = max_value - min_value
    mean_value = torch.mean(arr)
    stdev = torch.std(arr)

    return "\nmax {max_value}, min {min_value}, range {range_value}, mean {mean_value}, stdev {stdev}".format(
            max_value=np.format_float_scientific(max_value.cpu().numpy(), precision=2),
            min_value=np.format_float_scientific(min_value.cpu().numpy(), precision=2),
            range_value=np.format_float_scientific(range_value.cpu().numpy(), precision=2),
            mean_value=np.format_float_scientific(mean_value.cpu().numpy(), precision=2),
            stdev=np.format_float_scientific(stdev.cpu().numpy(), precision=2)
    )
    