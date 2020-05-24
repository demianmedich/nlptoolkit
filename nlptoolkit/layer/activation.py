# coding=utf-8
import torch.nn as nn


def get_activation_layer(activation: str):
    activation = activation.lower()
    if activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    else:
        return nn.Identity()
