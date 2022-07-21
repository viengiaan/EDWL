from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils

from TOOLS.ulti import Extract_SubImages_for_BAYER

from math import sqrt
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, d = 1):
        super(ResidualBlock, self).__init__()
        self.resmap = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size = 3, padding = d, stride = 1, dilation=d),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size = 3, padding = d, stride = 1, dilation=d),
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

##############################################################################
class FilterNet(nn.Module):
    def __init__(self, channels, kernel = 3):
        super(FilterNet, self).__init__()
        filters = 32
        self.kernel = kernel

        self.init = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.PReLU()
        )

        self.conv01 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.PReLU()
        )
        self.conv02 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.PReLU()
        )
        self.conv03 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.PReLU()
        )
        self.conv04 = nn.Sequential(
            nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.PReLU()
        )
        self.conv05 = nn.Sequential(
            nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.PReLU()
        )

        self.out_k = nn.Conv2d(filters * 2, channels * kernel * kernel, kernel_size = 3, stride = 1, padding = 1)

        self.soft_max = nn.Softmax(dim=1)

    def RESIZE_BLOCK(self, x):
        B, C, H, W = x.size()
        C_ = C // (self.kernel * self.kernel)

        out = torch.reshape(x, (B, self.kernel * self.kernel * C_, H, W))

        return out

    def forward(self, x):
        init = self.init(x)

        conv01 = self.conv01(init)
        conv02 = self.conv02(conv01)
        conv03 = self.conv03(conv02)
        conv04 = self.conv04(torch.cat((conv02, conv03), dim = 1))
        conv05 = self.conv05(torch.cat((conv01, conv04), dim = 1))
        out_k = self.out_k(torch.cat((init, conv05), dim = 1))

        # out_k = self.RESIZE_BLOCK(out_k)
        out_k = self.soft_max(out_k)

        return out_k

class Local_Filter_Operation(nn.Module):
    def __init__(self, channels, kernel, padding):
        super(Local_Filter_Operation, self).__init__()
        self.kernel = kernel
        self.padding = padding
        self.FilterNet = FilterNet(channels=channels, kernel=kernel)
        self.pad = torch.nn.ZeroPad2d(padding)

    def Reshape(self, x):
        device = x.device
        B, C, H, W = x.size()
        k = self.kernel
        p = self.padding

        x_pad = self.pad(x)

        x_pad = torch.reshape(x_pad, (B * C, 1, H + p * 2, W + p * 2))
        out = torch.zeros((B * C, k * k, H, W)).to(device)
        count = 0
        for i in range(k):
            for j in range(k):
                out[:,count,:,:] = x_pad[:, 0, 0 + i : H + i, 0 + j : W + j]
                count = count + 1

        out = torch.reshape(out, (B, C * k * k, H, W))

        return out

    def forward(self, inputs):

        inputs_resphape = self.Reshape(inputs)

        local_kernel = self.FilterNet(inputs)

        # B, C, H, W = inputs.size()

        results = local_kernel * inputs_resphape
        results = torch.sum(results, 1).unsqueeze(dim = 1)
        # results = torch.reshape(results, (B, C, H, W))
        # results = torch.sum(results, 1).unsqueeze(dim = 1)

        return results

class Feature_Selection(nn.Module):
    def __init__(self):
        super(Feature_Selection, self).__init__()

        self.FeatureSelection_R = Local_Filter_Operation(channels=4, kernel=3, padding=1)
        self.FeatureSelection_G1 = Local_Filter_Operation(channels=4, kernel=3, padding=1)
        self.FeatureSelection_G2 = Local_Filter_Operation(channels=4, kernel=3, padding=1)
        self.FeatureSelection_B = Local_Filter_Operation(channels=4, kernel=3, padding=1)

    def forward(self, x):

        #### BAYER to R-G-G-B subimages 32x32x1 -> 16x16x4
        E_sub = Extract_SubImages_for_BAYER(x)

        #### FEATURES selection
        R = self.FeatureSelection_R(E_sub)
        G1 = self.FeatureSelection_G1(E_sub)
        G2 = self.FeatureSelection_G1(E_sub)
        B = self.FeatureSelection_B(E_sub)

        out = torch.cat((R, G1, G2, B), dim = 1)

        return out

class Fusion_Net(nn.Module):
    def __init__(self):
        super(Fusion_Net, self).__init__()

        self.Fusion = Local_Filter_Operation(channels=2, kernel=3, padding=1)

    def forward(self, x1, x2):

        inputs = torch.cat((x1, x2), dim = 1)

        #### FUSION VIA LOCAL FILTERS
        out = self.Fusion(inputs)

        return out
        G = self.DM_G(inputs)
        B = self.DM_B(inputs)

        out = torch.cat((R, G, B), dim = 1)

        return out
