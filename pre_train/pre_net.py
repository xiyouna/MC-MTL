import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet

# from resnet12 import resnet12
# from pytorch.models.resnet_mtl import ResNetMtl
from models.SC import SCR, SelfCorrelationComputation
# from pytorch.models.cas import CasBlock

import sys


class PreNet(nn.Module):
    def __init__(self, args):
        super(PreNet, self).__init__()
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        planes = [640, 64, 64, 64, 640]
        # self.casblock = CasBlock(out_channels)
        self.resnet = ResNet(args)
        # self.resnet = ResNetMtl(mtl=mtl)
        self.scr_module = self._make_scr_layer(planes=[640, 64, 64, 64, 640])

    def _make_scr_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()

        corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
        self_block = SCR(planes=planes, stride=stride)
        layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)

    def forward(self, sample_image):
        embeddings_r = self.resnet(sample_image)
        identity = embeddings_r
        x = self.scr_module(embeddings_r)
        # x = self.corr_block(embeddings_r)
        # x = self.scr_block(x)

        embeddings = x + identity
        embeddings = F.relu(embeddings, inplace=False)
        # embeddings, ca_weights, sa_weights = self.casblock(embeddings_r)
        # labels = self.classifier(embeddings)

        return embeddings
