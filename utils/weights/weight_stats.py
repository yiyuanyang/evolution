"""
    Content: Functions that are helpful to
    analyze network weights
    Author: Yiyuan Yang
    Date: April. 20th 2020
"""

import torch


def tensor_statistics(tensor):
    """
        Given multi-dimensional tensor
        calculate all statistics
    """
    if len(tensor.shape) != 1:
        tensor = torch.flatten(tensor)
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    range_val = max_val - min_val
    mean_val = torch.mean(tensor)
    stdev = torch.std(tensor)
    tensor_stats = [max_val, min_val, range_val, mean_val, stdev]
    return tensor_stats


def basic_block_statistics(block):
    """
        Given a residual basic block
        calculate its weight statistics
    """
    conv1 = block.conv1
    conv2 = block.conv2

    weight = torch.cat(
        [torch.flatten(conv1.weight),
         torch.flatten(conv2.weight)])
    weight_statistics = tensor_statistics(weight)

    if conv1.weight.grad is not None:
        grad = torch.cat([
            torch.flatten(conv1.weight.grad),
            torch.flatten(conv2.weight.grad)
        ])
        grad_statistics = tensor_statistics(grad)
    else:
        grad_statistics = None
    residual_block_stats = [weight_statistics, grad_statistics]
    return residual_block_stats


def bottleneck_statistics(block, grad=True):
    """
        Given a residual basic block
        calculate its weight statistics
    """
    conv1 = block.conv1
    conv2 = block.conv2
    conv3 = block.conv3

    weight = torch.cat([
        torch.flatten(conv1.weight),
        torch.flatten(conv2.weight),
        torch.flatten(conv3.weight)
    ])
    weight_statistics = tensor_statistics(weight)

    if grad:
        grad = torch.cat([
            torch.flatten(conv1.weight.grad),
            torch.flatten(conv2.weight.grad),
            torch.flatten(conv3.weight.grad)
        ])
        grad_statistics = tensor_statistics(grad)
    else:
        grad_statistics = None

    residual_block_stats = [weight_statistics, grad_statistics]

    return residual_block_stats


def conv_statistics(conv, name, logger=None):
    """
        Given a convolutional layer
        Calculate its statistics
    """
    weight_statistics = tensor_statistics(torch.flatten(conv.weight))
    logger.log_tensor_statistics(weight_statistics, "weight")
    bias_statistics = None
    grad_statistics = None
    if conv.bias is not None:
        bias_statistics = tensor_statistics(conv.bias)
        logger.log_tensor_statistics(bias_statistics, "bias  ")
    if conv.weight.grad is not None:
        grad_statistics = tensor_statistics(torch.flatten(conv.weight.grad))
        logger.log_tensor_statistics(grad_statistics, "grad  ")
    conv_stats = [weight_statistics, bias_statistics, grad_statistics]
    return conv_stats
