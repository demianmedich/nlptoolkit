# coding=utf-8

import torch
import torch.nn as nn
from abc import ABCMeta
from abc import abstractmethod


class BaseModel(nn.Module, metaclass=ABCMeta):

    def calculate_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def inference(self, *args, **kwargs):
        raise NotImplementedError()
