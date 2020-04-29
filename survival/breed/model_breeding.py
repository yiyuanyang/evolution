"""
    Content: This file stores the logic that breed to unit size structures
    Author: Yiyuan Yang
    Date: April. 26th 2020
"""

import torch
from torch import nn

def breed_conv(
    left_conv, 
    right_conv,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    bias = False,
    policy = "average"
):
    left_weight = left_conv.weight.data
    right_weight = right_conv.weight.data
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        bias=bias
    )

    with torch.no_grad():
        new_conv.weight = nn.Parameter(breed_tensor(left_weight, right_weight))
        if bias:
            new_conv.bias = nn.Parameter(breed_tensor(left_conv.bias.data, right_conv.bias.data))

    return new_conv
        

    def breed_tensor(left_tensor, right_tensor, policy = "average"):
        if policy == "average":
            # A simple average of two tensors
            return (left_tensor + right_tensor)/2
        elif policy == "random" or policy == "random_mutate":
            # For every weight value, randomly select either from left or right
            left_factor = torch.randint_like(left_tensor, low=0, high=2)
            right_factor = torch.ones_like(right_tensor, low=0, high=2)
            right_factor = right_factor - left_factor
            left_factor = left_factor.type(torch.float32)
            right_factor = right_factor.type(torch.float32)
            linear_combined_weight = left_tensor * left_factor + right_tensor * right_factor

            if policy == "random_mutate":
            # Also multiply every weight by a modifier
                modifier = (torch.randint_like(linear_combined_weight, low=0, high=2) * torch.randn_like(linear_combined_weight) / 6) + 1
                linear_combined_weight  *= modifier

            return linear_combined_weight
                
