from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils

from math import sqrt
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.resmap = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = self.resmap(x)

        return x + residual

class convlayer(nn.Module):
    def __init__(self, nIn, nOut, k = 3, p = 1, s = 1, d = 1):
        super(convlayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=k, padding=p, stride=s, dilation=d),
            nn.BatchNorm2d(nOut),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class deconvlayer(nn.Module):
    def __init__(self, nIn, nOut, k = 4, p = 1, s = 2):
        super(deconvlayer, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(nIn, nOut, kernel_size=k, padding=p, stride=s),
            nn.BatchNorm2d(nOut),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


