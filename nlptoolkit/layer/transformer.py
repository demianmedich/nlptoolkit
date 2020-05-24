# coding=utf-8
from typing import Optional

import torch
from torch.nn import Module
from torch.nn import ModuleList


class TransformerEncoder(Module):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 num_layers: int,
                 intermediate_size: int,
                 layer_norm_eps: float = 1e-5,
                 dropout_prob: float = 0.1,
                 activation: str = "gelu",
                 use_positional_encoding: bool = True):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.layers = ModuleList(
            [TransformerEncoderSubLayer(
                d_model,
                num_heads,
                intermediate_size,
                layer_norm_eps=layer_norm_eps,
                dropout_prob=dropout_prob,
                activation=activation
            ) for _ in range(num_layers)])

    def forward(self,
                src_embedding: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        if self.use_positional_encoding:
            ...
        for layer in self.layers:
            src_embedding = layer(src_embedding, attention_mask)


class TransformerEncoderSubLayer(Module):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 intermediate_size: int,
                 layer_norm_eps: float,
                 dropout_prob: float,
                 activation: str):
        super().__init__()
        ...


class MultiHeadAttention(Module):

    def __init__(self):
        super().__init__()
