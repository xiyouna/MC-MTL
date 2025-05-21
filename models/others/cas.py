import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from resnet12 import resnet12

import sys

class CABlock(nn.Module):
    def __init__(self, in_channels=640, resize_factor=4):
        super(CABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hid_channels = in_channels // resize_factor
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, image_features):
        avg_pool_weights = self.fc(self.avg_pool(image_features))
        max_pool_weights = self.fc(self.max_pool(image_features))
        weights = torch.sigmoid(avg_pool_weights + max_pool_weights)

        return image_features * weights, weights

class SABlock(nn.Module):
    def __init__(self):
        super(SABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, image_features):
        transpose_features = image_features.view(*image_features.shape[:2], -1).transpose(1, 2)
        avg_pooled_features = self.avg_pool(transpose_features)
        max_pooled_features = self.max_pool(transpose_features)
        pooled_features = torch.cat((avg_pooled_features, max_pooled_features), 2)
        pooled_features = pooled_features.transpose(1, 2).view(-1, 2, *image_features.shape[2:])
        weights = torch.sigmoid(self.conv(pooled_features))

        return image_features * weights, weights

class CasBlock(nn.Module):
    def __init__(self, out_channels):
        super(CasBlock, self).__init__()
        self.ca_block = CABlock(out_channels)
        self.sa_block = SABlock()

    def forward(self, image_features):
        embeddings = image_features
        ca_embeddings, ca_weights = self.ca_block(embeddings)
        embeddings = ca_embeddings
        sa_embeddings, sa_weights = self.sa_block(embeddings)
        embeddings = sa_embeddings

        # return embeddings, ca_weights, sa_weights
        return embeddings

