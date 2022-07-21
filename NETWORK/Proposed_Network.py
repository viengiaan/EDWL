from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils

from NETWORK.Network import convlayer, deconvlayer

from math import sqrt
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

########################################################################################################################
########################################################################################################################
########################################################################################################################
# asymmetric gaussian shaped activation function g_A
class GaussActivation(nn.Module):
    def __init__(self, a, mu, sigma1, sigma2):
        super(GaussActivation, self).__init__()

        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma1 = Parameter(torch.tensor(sigma1, dtype=torch.float32))
        self.sigma2 = Parameter(torch.tensor(sigma2, dtype=torch.float32))

    def forward(self, inputFeatures):

        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.sigma1.data = torch.clamp(self.sigma1.data, 0.5, 2.0)
        self.sigma2.data = torch.clamp(self.sigma2.data, 0.5, 2.0)

        lowerThanMu = inputFeatures < self.mu
        largerThanMu = inputFeatures >= self.mu

        leftValuesActiv = self.a * torch.exp(- self.sigma1 * ((inputFeatures - self.mu) ** 2))
        leftValuesActiv.masked_fill_(largerThanMu, 0.0)

        rightValueActiv = 1 + (self.a - 1) * torch.exp(- self.sigma2 * ((inputFeatures - self.mu) ** 2))
        rightValueActiv.masked_fill_(lowerThanMu, 0.0)

        output = leftValuesActiv + rightValueActiv

        return output

class GaussActivationv2(nn.Module):
    def __init__(self, a, mu, sigma1, sigma2):
        super(GaussActivationv2, self).__init__()

        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma1 = Parameter(torch.tensor(sigma1, dtype=torch.float32))
        self.sigma2 = Parameter(torch.tensor(sigma2, dtype=torch.float32))

    def forward(self, inputFeatures):
        self.a.data = self.a.data.clamp(min=1.01)
        self.mu.data = self.mu.data.clamp(min=0.1)
        self.sigma1.data = self.sigma1.data.clamp(min=0.1)
        self.sigma2.data = self.sigma2.data.clamp(min=0.1)

        lowerThanMu = inputFeatures < self.mu
        largerThanMu = inputFeatures >= self.mu

        leftValuesActiv = self.a * torch.exp(- self.sigma1 * ((inputFeatures - self.mu) ** 2))
        leftValuesActiv.masked_fill_(largerThanMu, 0.0)

        rightValueActiv = 1 + (self.a - 1) * torch.exp(- self.sigma2 * ((inputFeatures - self.mu) ** 2))
        rightValueActiv.masked_fill_(lowerThanMu, 0.0)

        output = leftValuesActiv + rightValueActiv

        return output

# mask updating functions, we recommand using alpha that is larger than 0 and lower than 1.0
class MaskUpdate(nn.Module):
    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()

        self.updateFunc = nn.ReLU(True)
        #self.alpha = Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.alpha = alpha
    def forward(self, inputMaskMap):
        """ self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        print(self.alpha) """

        return torch.pow(self.updateFunc(inputMaskMap) + 1e-6, self.alpha)

############################################################################
class convlayer_maskv2(nn.Module):
    def __init__(self, nIn, nOut, k = 3, p = 1, s = 1, d = 1):
        super(convlayer_maskv2, self).__init__()

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=k, stride=s, padding=p, dilation=d)

        self.attent_func = GaussActivation(1.1, 2.0, 1.0, 1.0)

        self.update = MaskUpdate(0.8)

    def forward(self, mask):
        features = self.conv(mask)

        maskActiv = self.attent_func(features)
        mask_update = self.update(features)

        return mask_update, maskActiv

class convlayer_maskv3(nn.Module):
    def __init__(self, nIn, nOut, k = 3, p = 1, s = 1, d = 1):
        super(convlayer_maskv3, self).__init__()

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=k, stride=s, padding=p, dilation=d)

        self.attent_func = GaussActivationv2(1.1, 2.0, 1.0, 1.0)

        self.update = MaskUpdate(0.8)

    def forward(self, mask):
        features = self.conv(mask)

        maskActiv = self.attent_func(features)
        mask_update = self.update(features)

        return mask_update, maskActiv

########################################################################################################################
########################################################################################################################
########################################################################################################################
class FusionModule_v4(nn.Module):
    def __init__(self, channels):
        super(FusionModule_v4, self).__init__()
        self.channels = channels

        self.alpha_C = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.alpha_H = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.alpha_W = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, F_ch, F_sp):
        batch_size = F_ch.size(0)
        H = F_ch.size(2)
        W = F_ch.size(3)

        Fc = F_ch
        Fs = F_sp

        # Learn
        alpha_C = self.alpha_C(torch.cat((F_ch, F_sp), dim=1))
        alpha_H = self.alpha_H(torch.cat((F_ch, F_sp), dim=1))
        alpha_W = self.alpha_W(torch.cat((F_ch, F_sp), dim=1))

        # Channel-Similarity Map
        Fc_a = Fc.view(batch_size, self.channels, -1)  # B, C, HW
        Fs_a = Fs.view(batch_size, self.channels, -1)
        Fs_a = Fs_a.permute(0, 2, 1)  # B, HW, C

        Fc_a = F.normalize(Fc_a, dim=2)
        Fs_a = F.normalize(Fs_a, dim=1)
        Sc = torch.bmm(Fc_a, Fs_a)  # B, C, C
        Sc_sum = torch.sum(Sc, dim=2)
        Sc_sum = torch.sum(Sc_sum, dim=1)
        Sc_sum = Sc_sum.unsqueeze(dim=1)
        Sc_sum = Sc_sum.unsqueeze(dim=2)  # B, 1, 1
        Sc_norm = Sc / (Sc_sum + 1e-5)  # B, C, C
        Sc_norm = torch.sum(Sc, dim=2).unsqueeze(dim=2)

        # Height-Similarity Map
        Fc_b = Fc.permute(0, 2, 1, 3).contiguous()  # B (0), H(2), C(1), W(3)
        Fs_b = Fs.permute(0, 2, 1, 3).contiguous()

        Fc_b = Fc_b.view(batch_size, H, -1)  # B, H, CW
        Fs_b = Fs_b.view(batch_size, H, -1)  # B, H, CW
        Fs_b = Fs_b.permute(0, 2, 1)  # B, CW, H

        Fc_b = F.normalize(Fc_b, dim=2)
        Fs_b = F.normalize(Fs_b, dim=1)
        Sh = torch.bmm(Fc_b, Fs_b)  # B, H, H
        Sh_sum = torch.sum(Sh, dim=2)
        Sh_sum = torch.sum(Sh_sum, dim=1)
        Sh_sum = Sh_sum.unsqueeze(dim=1)
        Sh_sum = Sh_sum.unsqueeze(dim=2)  # B, 1, 1
        Sh_norm = Sh / (Sh_sum + 1e-5)  # B, H, H
        Sh_norm = torch.sum(Sh, dim=2).unsqueeze(dim=2)

        # Column-Similarity Map
        Fc_c = Fc.permute(0, 3, 1, 2).contiguous()  # B (0), W(3), C(1), H(2)
        Fs_c = Fs.permute(0, 3, 1, 2).contiguous()

        Fc_c = Fc_c.view(batch_size, W, -1)  # B, W, CH
        Fs_c = Fs_c.view(batch_size, W, -1)  # B, W, CH
        Fs_c = Fs_c.permute(0, 2, 1)  # B, CH, W

        Fc_c = F.normalize(Fc_c, dim=2)
        Fs_c = F.normalize(Fs_c, dim=1)
        Sw = torch.bmm(Fc_c, Fs_c)  # B, W, W
        Sw_sum = torch.sum(Sw, dim=2)
        Sw_sum = torch.sum(Sw_sum, dim=1)
        Sw_sum = Sw_sum.unsqueeze(dim=1)
        Sw_sum = Sw_sum.unsqueeze(dim=2)  # B, 1, 1
        Sw_norm = Sw / (Sw_sum + 1e-5)  # B, W, W
        Sw_norm = torch.sum(Sw, dim=2).unsqueeze(dim=2)

        # Final Weight Map for fusing Fch & Fsp
        Sc_full = Sc_norm.expand(-1, -1, H * W)  # B, C, HW
        Sc_full = Sc_full.view(batch_size, self.channels, H, W)  # B, C, H, W

        Sh_full = Sh_norm.expand(-1, -1, self.channels * W)  # B, H, CW
        Sh_full = Sh_full.view(batch_size, H, self.channels, W)  # B(0), H(1), C(2), W(3)
        Sh_full = Sh_full.permute(0, 2, 1, 3).contiguous()  # B, C, H, W

        Sw_full = Sw_norm.expand(-1, -1, self.channels * H)  # B, W, CH
        Sw_full = Sw_full.view(batch_size, W, self.channels, H)  # B(0), W(1), C(2), H(3)
        Sw_full = Sw_full.permute(0, 2, 3, 1).contiguous()

        Ac = (alpha_C * Sc_full + alpha_H * Sh_full + alpha_W * Sw_full) / (
                    alpha_C + alpha_H + alpha_W + 1e-5)  # B, C, H, W
        Ac = F.sigmoid(Ac)
        Ac_sum = torch.sum(Ac, dim=3)
        Ac_sum = torch.sum(Ac_sum, dim=2)
        Ac_sum = Ac_sum.unsqueeze(dim=2)
        Ac_sum = Ac_sum.unsqueeze(dim=3)
        Ac_norm = Ac / (Ac_sum + 1e-5)  # B, C, H, W

        return Ac_norm

class PE_Encoder_with_FusionModule_v4(nn.Module):
    def __init__(self):
        super(PE_Encoder_with_FusionModule_v4, self).__init__()
        filters = [128, 256, 512]

        ### Feature Encoder
        self.inc = nn.Sequential(
            convlayer(1, filters[0], k=3, p=1, s=1),
            convlayer(filters[0], filters[0], k=3, p=1, s=1),
        )

        self.down1 = nn.Sequential(
            convlayer(filters[0], filters[0], k=2, p=0, s=2),
            convlayer(filters[0], filters[1], k=3, p=1, s=1),
            convlayer(filters[1], filters[1], k=3, p=1, s=1),
        )

        self.down2 = nn.Sequential(
            convlayer(filters[1], filters[1], k=2, p=0, s=2),
            convlayer(filters[1], filters[2], k=3, p=1, s=1),
            convlayer(filters[2], filters[2], k=3, p=1, s=1),
        )

        ### Attention maps Generator
        self.inc_mask_a = convlayer_maskv3(1, filters[0], k=3, p=1, s=1)
        self.inc_mask_b = convlayer_maskv3(filters[0], filters[0], k=3, p=1, s=1)

        self.down1a_mask = convlayer_maskv3(filters[0], filters[0], k=2, p=0, s=2)
        self.down1b_mask = convlayer_maskv3(filters[0], filters[1], k=3, p=1, s=1)
        self.down1c_mask = convlayer_maskv3(filters[1], filters[1], k=3, p=1, s=1)

        self.down2a_mask = convlayer_maskv3(filters[1], filters[1], k=2, p=0, s=2)
        self.down2b_mask = convlayer_maskv3(filters[1], filters[2], k=3, p=1, s=1)
        self.down2c_mask = convlayer_maskv3(filters[2], filters[2], k=3, p=1, s=1)

        ### Learnable Fusion Weights

        # scale 1
        self.sc1_fusion = FusionModule_v4(128)

        self.sc1_ws = nn.Parameter(torch.randn(128, 1, 1)).float()
        self.sc1_wl = nn.Parameter(torch.randn(128, 1, 1)).float()

        # scale 2
        self.sc2_fusion = FusionModule_v4(256)

        self.sc2_ws = nn.Parameter(torch.randn(256, 1, 1)).float()
        self.sc2_wl = nn.Parameter(torch.randn(256, 1, 1)).float()

        # scale 3
        self.sc3_fusion = FusionModule_v4(512)

        self.ws = nn.Parameter(torch.randn(512, 1, 1)).float()
        self.wl = nn.Parameter(torch.randn(512, 1, 1)).float()

    def forward(self, x_s, x_l, M_s, M_l):
        eps = 1e-5

        ### Features Extraction
        x1_s = self.inc(x_s)
        x1_l = self.inc(x_l)

        x2_s = self.down1(x1_s)
        x2_l = self.down1(x1_l)

        x3_s = self.down2(x2_s)
        x3_l = self.down2(x2_l)

        ### Construct Attention Maps
        mask_update1_s, _ = self.inc_mask_a(M_s)
        mask_update1_s, maskActiv1_s = self.inc_mask_b(mask_update1_s)

        mask_update1_l, _ = self.inc_mask_a(M_l)
        mask_update1_l, maskActiv1_l = self.inc_mask_b(mask_update1_l)

        mask_update2_s, _ = self.down1a_mask(mask_update1_s)
        mask_update2_s, _ = self.down1b_mask(mask_update2_s)
        mask_update2_s, maskActiv2_s = self.down1c_mask(mask_update2_s)

        mask_update2_l, _ = self.down1a_mask(mask_update1_l)
        mask_update2_l, _ = self.down1b_mask(mask_update2_l)
        mask_update2_l, maskActiv2_l = self.down1c_mask(mask_update2_l)

        mask_update3_s, _ = self.down2a_mask(mask_update2_s)
        mask_update3_s, _ = self.down2b_mask(mask_update3_s)
        _, maskActiv3_s = self.down2c_mask(mask_update3_s)

        mask_update3_l, _ = self.down2a_mask(mask_update2_l)
        mask_update3_l, _ = self.down2b_mask(mask_update3_l)
        _, maskActiv3_l = self.down2c_mask(mask_update3_l)

        ### Multi-exposed features fusion
        x1_spatial = (maskActiv1_s * x1_s + maskActiv1_l * x1_l) \
                     / (maskActiv1_s + maskActiv1_l + eps)
        x1_global = (self.sc1_ws * x1_s + self.sc1_wl * x1_l) / (self.sc1_ws + self.sc1_wl + eps)

        x1_A_global = self.sc1_fusion(x1_global, x1_spatial)
        x1_fused = x1_A_global * x1_global + (1. - x1_A_global) * x1_spatial

        x2_spatial = (maskActiv2_s * x2_s + maskActiv2_l * x2_l) \
                     / (maskActiv2_s + maskActiv2_l + eps)
        x2_global = (self.sc2_ws * x2_s + self.sc2_wl * x2_l) / (self.sc2_ws + self.sc2_wl + eps)

        x2_A_global = self.sc2_fusion(x2_global, x2_spatial)
        x2_fused = x2_A_global * x2_global + (1. - x2_A_global) * x2_spatial

        x3_spatial = (maskActiv3_s * x3_s + maskActiv3_l * x3_l) \
                     / (maskActiv3_s + maskActiv3_l + eps)
        x3_global = (self.ws * x3_s + self.wl * x3_l) / (self.ws + self.wl + eps)

        x3_A_global = self.sc3_fusion(x3_global, x3_spatial)
        x3_fused = x3_A_global * x3_global + (1. - x3_A_global) * x3_spatial

        return x3_fused, x2_fused, x1_fused


class ExRNet_FusionModule_v4(nn.Module):
    def __init__(self):
        super(ExRNet_FusionModule_v4, self).__init__()
        self.G1 = PE_Encoder_with_FusionModule_v4()

        ### Decoder
        filters = [128, 256, 512]
        self.up1 = deconvlayer(filters[2], filters[1])
        self.conv1 = nn.Sequential(
            convlayer(filters[1] * 2, filters[1], k=3, p=1, s=1),
            convlayer(filters[1], filters[1], k=3, p=1, s=1),
        )

        self.up2 = deconvlayer(filters[1], filters[0])
        self.conv2 = nn.Sequential(
            convlayer(filters[0] * 2, filters[0], k=3, p=1, s=1),
            convlayer(filters[0], filters[0], k=3, p=1, s=1),
        )

        self.outc = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.sig = nn.Sigmoid()

        self.bicubic = nn.Upsample(scale_factor=(2, 1), mode='bicubic')

    def Exposure_Decompose(self, x):
        device = x.device
        BATCH, CHANNEL, HEIGHT, WIDTH = x.size()

        Es = torch.zeros((BATCH, CHANNEL, HEIGHT // 2, WIDTH)).to(device)
        Es[:, :, 0: HEIGHT // 2: 2, :] = x[:, :, 0: HEIGHT: 4, :]
        Es[:, :, 1: HEIGHT // 2: 2, :] = x[:, :, 1: HEIGHT: 4, :]

        El = torch.zeros((BATCH, CHANNEL, HEIGHT // 2, WIDTH)).to(device)
        El[:, :, 0: HEIGHT // 2: 2, :] = x[:, :, 2: HEIGHT: 4, :]
        El[:, :, 1: HEIGHT // 2: 2, :] = x[:, :, 3: HEIGHT: 4, :]

        return Es, El

    def forward(self, x, mask):
        # MASK: 0-well exposed regions; 1-poor exposed regions
        new_mask = 1. - mask

        B = new_mask.size(0)
        num_of_values = torch.sum(new_mask, dim=2)
        num_of_values = torch.sum(num_of_values, dim=2)
        for i in range(B):
            if num_of_values[i, :] < 102:  # 10% of 32x32
                A = new_mask[i, :, :, :]
                A[A < 1e-5] = 0.1
                new_mask[i, :, :, :] = A

        #### Decompose Es, El
        Es, El = self.Exposure_Decompose(x)
        Ms, Ml = self.Exposure_Decompose(new_mask)

        #### Bicubic interpolation
        Es = self.bicubic(Es)
        El = self.bicubic(El)

        Ms = self.bicubic(Ms).clamp(min=0, max=1)
        Ml = self.bicubic(Ml).clamp(min=0, max=1)

        #### Encode features
        F_merged_3, F_merged_2, F_merged_1 = self.G1(Es, El, Ms, Ml)

        #### Decode features
        up1 = self.up1(F_merged_3)
        up1 = self.conv1(torch.cat((F_merged_2, up1), dim=1))

        up2 = self.up2(up1)
        up2 = self.conv2(torch.cat((F_merged_1, up2), dim=1))

        out = self.outc(up2)
        out = self.sig(out)

        return out

########################################################################
########################################################################
########################################################################
class FusionModule_for_MEF_v2(nn.Module):
    def __init__(self, channels):
        super(FusionModule_for_MEF_v2, self).__init__()
        self.channels = channels

        self.alpha_C = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.alpha_H = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.alpha_W = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # self.alpha_C = nn.Parameter(torch.randn(self.channels, self.H, self.W)).float()
        # self.alpha_H = nn.Parameter(torch.randn(self.channels, self.H, self.W)).float()
        # self.alpha_W = nn.Parameter(torch.randn(self.channels, self.H, self.W)).float()

    def forward(self, F_ch, F_sp):
        # FS: F_ch
        # FL: F_sp
        batch_size = F_ch.size(0)
        H = F_ch.size(2)
        W = F_ch.size(3)

        Fc = F_ch
        Fs = F_sp

        # Learn
        alpha_C = self.alpha_C(torch.cat((F_ch, F_sp), dim=1))
        alpha_H = self.alpha_H(torch.cat((F_ch, F_sp), dim=1))
        alpha_W = self.alpha_W(torch.cat((F_ch, F_sp), dim=1))

        # Channel-Similarity Map
        Fc_a = Fc.view(batch_size, self.channels, -1)  # B, C, HW
        Fs_a = Fs.view(batch_size, self.channels, -1)
        Fs_a = Fs_a.permute(0, 2, 1)  # B, HW, C

        Fc_a = F.normalize(Fc_a, dim=2)
        Fs_a = F.normalize(Fs_a, dim=1)
        Sc = torch.bmm(Fc_a, Fs_a)  # B, C, C
        # Sc = Normalize_Tensor(Sc)
        Sc_sum = torch.sum(Sc, dim=2)
        Sc_sum = torch.sum(Sc_sum, dim=1)
        Sc_sum = Sc_sum.unsqueeze(dim=1)
        Sc_sum = Sc_sum.unsqueeze(dim=2)  # B, 1, 1
        Sc_norm = Sc / (Sc_sum + 1e-5)  # B, C, C
        Sc_norm = torch.sum(Sc, dim=2).unsqueeze(dim=2)

        # Height-Similarity Map
        Fc_b = Fc.permute(0, 2, 1, 3).contiguous()  # B (0), H(2), C(1), W(3)
        Fs_b = Fs.permute(0, 2, 1, 3).contiguous()

        Fc_b = Fc_b.view(batch_size, H, -1)  # B, H, CW
        Fs_b = Fs_b.view(batch_size, H, -1)  # B, H, CW
        Fs_b = Fs_b.permute(0, 2, 1)  # B, CW, H

        Fc_b = F.normalize(Fc_b, dim=2)
        Fs_b = F.normalize(Fs_b, dim=1)
        Sh = torch.bmm(Fc_b, Fs_b)  # B, H, H
        # Sh = Normalize_Tensor(Sh)
        Sh_sum = torch.sum(Sh, dim=2)
        Sh_sum = torch.sum(Sh_sum, dim=1)
        Sh_sum = Sh_sum.unsqueeze(dim=1)
        Sh_sum = Sh_sum.unsqueeze(dim=2)  # B, 1, 1
        Sh_norm = Sh / (Sh_sum + 1e-5)  # B, H, H
        Sh_norm = torch.sum(Sh, dim=2).unsqueeze(dim=2)

        # Column-Similarity Map
        Fc_c = Fc.permute(0, 3, 1, 2).contiguous()  # B (0), W(3), C(1), H(2)
        Fs_c = Fs.permute(0, 3, 1, 2).contiguous()

        Fc_c = Fc_c.view(batch_size, W, -1)  # B, W, CH
        Fs_c = Fs_c.view(batch_size, W, -1)  # B, W, CH
        Fs_c = Fs_c.permute(0, 2, 1)  # B, CH, W

        Fc_c = F.normalize(Fc_c, dim=2)
        Fs_c = F.normalize(Fs_c, dim=1)
        Sw = torch.bmm(Fc_c, Fs_c)  # B, W, W
        # Sw = Normalize_Tensor(Sw)
        Sw_sum = torch.sum(Sw, dim=2)
        Sw_sum = torch.sum(Sw_sum, dim=1)
        Sw_sum = Sw_sum.unsqueeze(dim=1)
        Sw_sum = Sw_sum.unsqueeze(dim=2)  # B, 1, 1
        Sw_norm = Sw / (Sw_sum + 1e-5)  # B, W, W
        Sw_norm = torch.sum(Sw, dim=2).unsqueeze(dim=2)

        # Final Weight Map for fusing Fch & Fsp
        Sc_full = Sc_norm.expand(-1, -1, H * W)  # B, C, HW
        Sc_full = Sc_full.view(batch_size, self.channels, H, W)  # B, C, H, W

        Sh_full = Sh_norm.expand(-1, -1, self.channels * W)  # B, H, CW
        Sh_full = Sh_full.view(batch_size, H, self.channels, W)  # B(0), H(1), C(2), W(3)
        Sh_full = Sh_full.permute(0, 2, 1, 3).contiguous()  # B, C, H, W

        Sw_full = Sw_norm.expand(-1, -1, self.channels * H)  # B, W, CH
        Sw_full = Sw_full.view(batch_size, W, self.channels, H)  # B(0), W(1), C(2), H(3)
        Sw_full = Sw_full.permute(0, 2, 3, 1).contiguous()

        Ac = (alpha_C * Sc_full + alpha_H * Sh_full + alpha_W * Sw_full) / (
                    alpha_C + alpha_H + alpha_W + 1e-5)  # B, C, H, W
        # Ac = Normalize_Tensor(Ac, flag=1)
        Ac = F.sigmoid(Ac)
        Ac_sum = torch.sum(Ac, dim=3)
        Ac_sum = torch.sum(Ac_sum, dim=2)
        Ac_sum = Ac_sum.unsqueeze(dim=2)
        Ac_sum = Ac_sum.unsqueeze(dim=3)
        Ac_norm = Ac / (Ac_sum + 1e-5)  # B, C, H, W

        return Ac_norm

class PE_Encoder_with_FusionModule_for_MEF_v2(nn.Module):
    def __init__(self):
        super(PE_Encoder_with_FusionModule_for_MEF_v2, self).__init__()
        filters = [128, 256, 512]

        ### Feature Encoder
        self.inc = nn.Sequential(
            convlayer(1, filters[0], k=3, p=1, s=1),
            convlayer(filters[0], filters[0], k=3, p=1, s=1),
        )

        self.down1 = nn.Sequential(
            convlayer(filters[0], filters[0], k=2, p=0, s=2),
            convlayer(filters[0], filters[1], k=3, p=1, s=1),
            convlayer(filters[1], filters[1], k=3, p=1, s=1),
        )

        self.down2 = nn.Sequential(
            convlayer(filters[1], filters[1], k=2, p=0, s=2),
            convlayer(filters[1], filters[2], k=3, p=1, s=1),
            convlayer(filters[2], filters[2], k=3, p=1, s=1),
        )

        ### Attention maps Generator
        self.inc_mask_a = convlayer_maskv3(1, filters[0], k=3, p=1, s=1)
        self.inc_mask_b = convlayer_maskv3(filters[0], filters[0], k=3, p=1, s=1)

        self.down1a_mask = convlayer_maskv3(filters[0], filters[0], k=2, p=0, s=2)
        self.down1b_mask = convlayer_maskv3(filters[0], filters[1], k=3, p=1, s=1)
        self.down1c_mask = convlayer_maskv3(filters[1], filters[1], k=3, p=1, s=1)

        self.down2a_mask = convlayer_maskv3(filters[1], filters[1], k=2, p=0, s=2)
        self.down2b_mask = convlayer_maskv3(filters[1], filters[2], k=3, p=1, s=1)
        self.down2c_mask = convlayer_maskv3(filters[2], filters[2], k=3, p=1, s=1)

        ### Learnable Fusion Weights

        # scale 1
        self.sc1_fusion = FusionModule_for_MEF_v2(128)

        # scale 2
        self.sc2_fusion = FusionModule_for_MEF_v2(256)

        # scale 3
        self.sc3_fusion = FusionModule_for_MEF_v2(512)

    def forward(self, x_s, x_l, M_s, M_l):
        eps = 1e-5

        ### Features Extraction
        x1_s = self.inc(x_s)
        x1_l = self.inc(x_l)

        x2_s = self.down1(x1_s)
        x2_l = self.down1(x1_l)

        x3_s = self.down2(x2_s)
        x3_l = self.down2(x2_l)

        ### Construct Attention Maps
        mask_update1_s, _ = self.inc_mask_a(M_s)
        mask_update1_s, maskActiv1_s = self.inc_mask_b(mask_update1_s)

        mask_update1_l, _ = self.inc_mask_a(M_l)
        mask_update1_l, maskActiv1_l = self.inc_mask_b(mask_update1_l)

        mask_update2_s, _ = self.down1a_mask(mask_update1_s)
        mask_update2_s, _ = self.down1b_mask(mask_update2_s)
        mask_update2_s, maskActiv2_s = self.down1c_mask(mask_update2_s)

        mask_update2_l, _ = self.down1a_mask(mask_update1_l)
        mask_update2_l, _ = self.down1b_mask(mask_update2_l)
        mask_update2_l, maskActiv2_l = self.down1c_mask(mask_update2_l)

        mask_update3_s, _ = self.down2a_mask(mask_update2_s)
        mask_update3_s, _ = self.down2b_mask(mask_update3_s)
        _, maskActiv3_s = self.down2c_mask(mask_update3_s)

        mask_update3_l, _ = self.down2a_mask(mask_update2_l)
        mask_update3_l, _ = self.down2b_mask(mask_update3_l)
        _, maskActiv3_l = self.down2c_mask(mask_update3_l)

        ### Multi-exposed features fusion
        A_map_x1_s = self.sc1_fusion(x1_s, x1_l)
        att_x1_s = A_map_x1_s * maskActiv1_s
        att_x1_l = (1. - A_map_x1_s) * maskActiv1_l

        x1_fused = (att_x1_s * x1_s + att_x1_l * x1_l) \
                   / (att_x1_s + att_x1_l + eps)

        A_map_x2_s = self.sc2_fusion(x2_s, x2_l)
        att_x2_s = A_map_x2_s * maskActiv2_s
        att_x2_l = (1. - A_map_x2_s) * maskActiv2_l

        x2_fused = (att_x2_s * x2_s + att_x2_l * x2_l) \
                   / (att_x2_s + att_x2_l + eps)

        A_map_x3_s = self.sc3_fusion(x3_s, x3_l)
        att_x3_s = A_map_x3_s * maskActiv3_s
        att_x3_l = (1. - A_map_x3_s) * maskActiv3_l

        x3_fused = (att_x3_s * x3_s + att_x3_l * x3_l) \
                   / (att_x3_s + att_x3_l + eps)

        return x3_fused, x2_fused, x1_fused

class ExRNet_FusionModule_for_MEF_v2(nn.Module):
    def __init__(self):
        super(ExRNet_FusionModule_for_MEF_v2, self).__init__()
        self.G1 = PE_Encoder_with_FusionModule_for_MEF_v2()

        ### Decoder
        filters = [128, 256, 512]
        self.up1 = deconvlayer(filters[2], filters[1])
        self.conv1 = nn.Sequential(
            convlayer(filters[1] * 2, filters[1], k=3, p=1, s=1),
            convlayer(filters[1], filters[1], k=3, p=1, s=1),
        )

        self.up2 = deconvlayer(filters[1], filters[0])
        self.conv2 = nn.Sequential(
            convlayer(filters[0] * 2, filters[0], k=3, p=1, s=1),
            convlayer(filters[0], filters[0], k=3, p=1, s=1),
        )

        self.outc = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.sig = nn.Sigmoid()

        self.bicubic = nn.Upsample(scale_factor=(2, 1), mode='bicubic')

    def Exposure_Decompose(self, x):
        device = x.device
        BATCH, CHANNEL, HEIGHT, WIDTH = x.size()

        Es = torch.zeros((BATCH, CHANNEL, HEIGHT // 2, WIDTH)).to(device)
        Es[:, :, 0: HEIGHT // 2: 2, :] = x[:, :, 0: HEIGHT: 4, :]
        Es[:, :, 1: HEIGHT // 2: 2, :] = x[:, :, 1: HEIGHT: 4, :]

        El = torch.zeros((BATCH, CHANNEL, HEIGHT // 2, WIDTH)).to(device)
        El[:, :, 0: HEIGHT // 2: 2, :] = x[:, :, 2: HEIGHT: 4, :]
        El[:, :, 1: HEIGHT // 2: 2, :] = x[:, :, 3: HEIGHT: 4, :]

        return Es, El

    def forward(self, x, mask):
        # MASK: 0-well exposed regions; 1-poor exposed regions
        new_mask = 1. - mask

        B = new_mask.size(0)
        num_of_values = torch.sum(new_mask, dim=2)
        num_of_values = torch.sum(num_of_values, dim=2)
        for i in range(B):
            if num_of_values[i, :] < 102:  # 10% of 32x32
                A = new_mask[i, :, :, :]
                A[A < 1e-5] = 0.1
                new_mask[i, :, :, :] = A

        #### Decompose Es, El
        Es, El = self.Exposure_Decompose(x)
        Ms, Ml = self.Exposure_Decompose(new_mask)

        #### Bicubic interpolation
        Es = self.bicubic(Es)
        El = self.bicubic(El)

        Ms = self.bicubic(Ms).clamp(min=0, max=1)
        Ml = self.bicubic(Ml).clamp(min=0, max=1)

        #### Encode features
        F_merged_3, F_merged_2, F_merged_1 = self.G1(Es, El, Ms, Ml)

        #### Decode features
        up1 = self.up1(F_merged_3)
        up1 = self.conv1(torch.cat((F_merged_2, up1), dim=1))

        up2 = self.up2(up1)
        up2 = self.conv2(torch.cat((F_merged_1, up2), dim=1))

        out = self.outc(up2)
        out = self.sig(out)

        return out
        
########################################################################
########################################################################
########################################################################
class FusionModule_v5(nn.Module):
    def __init__(self, channels):
        super(FusionModule_v5, self).__init__()
        self.channels = channels

        self.alpha_C = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.alpha_H = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.alpha_W = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # self.alpha_C = nn.Parameter(torch.randn(self.channels, self.H, self.W)).float()
        # self.alpha_H = nn.Parameter(torch.randn(self.channels, self.H, self.W)).float()
        # self.alpha_W = nn.Parameter(torch.randn(self.channels, self.H, self.W)).float()

    def forward(self, F_ch, F_sp):
        batch_size = F_ch.size(0)
        H = F_ch.size(2)
        W = F_ch.size(3)

        Fc = F_ch
        Fs = F_sp

        # Learn
        alpha_C = self.alpha_C(torch.cat((F_ch, F_sp), dim=1))
        alpha_H = self.alpha_H(torch.cat((F_ch, F_sp), dim=1))
        alpha_W = self.alpha_W(torch.cat((F_ch, F_sp), dim=1))

        # Channel-Similarity Map
        Fc_a = Fc.view(batch_size, self.channels, -1)  # B, C, HW
        Fs_a = Fs.view(batch_size, self.channels, -1)
        Fs_a = Fs_a.permute(0, 2, 1)  # B, HW, C

        Fc_a = F.normalize(Fc_a, dim = 2)
        Fs_a = F.normalize(Fs_a, dim = 1)
        Sc = torch.bmm(Fc_a, Fs_a) # B, C, C
        Sc_sum = torch.sum(Sc, dim = 2)
        Sc_sum = torch.sum(Sc_sum, dim = 1)
        Sc_sum = Sc_sum.unsqueeze(dim = 1)
        Sc_sum = Sc_sum.unsqueeze(dim = 2) # B, 1, 1
        Sc_norm = Sc / (Sc_sum + 1e-5) # B, C, C
        Sc_norm = torch.sum(Sc, dim = 2).unsqueeze(dim = 2)

        # Height-Similarity Map
        Fc_b = Fc.permute(0,2,1,3).contiguous() # B (0), H(2), C(1), W(3)
        Fs_b = Fs.permute(0,2,1,3).contiguous()

        Fc_b = Fc_b.view(batch_size, H, -1) # B, H, CW
        Fs_b = Fs_b.view(batch_size, H, -1)  # B, H, CW
        Fs_b = Fs_b.permute(0, 2, 1) # B, CW, H

        Fc_b = F.normalize(Fc_b, dim = 2)
        Fs_b = F.normalize(Fs_b, dim = 1)
        Sh = torch.bmm(Fc_b, Fs_b) # B, H, H
        Sh_sum = torch.sum(Sh, dim = 2)
        Sh_sum = torch.sum(Sh_sum, dim = 1)
        Sh_sum = Sh_sum.unsqueeze(dim = 1)
        Sh_sum = Sh_sum.unsqueeze(dim = 2) # B, 1, 1
        Sh_norm = Sh / (Sh_sum + 1e-5) # B, H, H
        Sh_norm = torch.sum(Sh, dim = 2).unsqueeze(dim = 2)

        # Column-Similarity Map
        Fc_c = Fc.permute(0, 3, 1, 2).contiguous()  # B (0), W(3), C(1), H(2)
        Fs_c = Fs.permute(0, 3, 1, 2).contiguous()

        Fc_c = Fc_c.view(batch_size, W, -1)  # B, W, CH
        Fs_c = Fs_c.view(batch_size, W, -1)  # B, W, CH
        Fs_c = Fs_c.permute(0, 2, 1)  # B, CH, W

        Fc_c = F.normalize(Fc_c, dim = 2)
        Fs_c = F.normalize(Fs_c, dim = 1)
        Sw = torch.bmm(Fc_c, Fs_c) # B, W, W
        Sw_sum = torch.sum(Sw, dim = 2)
        Sw_sum = torch.sum(Sw_sum, dim = 1)
        Sw_sum = Sw_sum.unsqueeze(dim = 1)
        Sw_sum = Sw_sum.unsqueeze(dim = 2) # B, 1, 1
        Sw_norm = Sw / (Sw_sum + 1e-5) # B, W, W
        Sw_norm = torch.sum(Sw, dim=2).unsqueeze(dim=2)

        # Final Weight Map for fusing Fch & Fsp
        Sc_full = Sc_norm.expand(-1, -1, H * W) # B, C, HW
        Sc_full = Sc_full.view(batch_size, self.channels, H, W) # B, C, H, W

        Sh_full = Sh_norm.expand(-1, -1, self.channels * W)  # B, H, CW
        Sh_full = Sh_full.view(batch_size, H, self.channels, W)# B(0), H(1), C(2), W(3)
        Sh_full = Sh_full.permute(0, 2, 1, 3).contiguous() # B, C, H, W

        Sw_full = Sw_norm.expand(-1, -1, self.channels * H) # B, W, CH
        Sw_full = Sw_full.view(batch_size, W, self.channels, H) # B(0), W(1), C(2), H(3)
        Sw_full = Sw_full.permute(0, 2, 3, 1).contiguous()

        Ac = (alpha_C * Sc_full + alpha_H * Sh_full + alpha_W * Sw_full) / (alpha_C + alpha_H + alpha_W + 1e-5) # B, C, H, W
        Ac = Normalize_Tensor(Ac, flag=1) # to [0, 1]
        # Ac = F.sigmoid(Ac)
        Ac_sum = torch.sum(Ac, dim = 3)
        Ac_sum = torch.sum(Ac_sum, dim = 2)
        Ac_sum = Ac_sum.unsqueeze(dim = 2)
        Ac_sum = Ac_sum.unsqueeze(dim = 3)
        Ac_norm = Ac / (Ac_sum + 1e-5) # B, C, H, W

        return Ac_norm
        
class PE_Encoder_with_FusionModule_v5(nn.Module):
    def __init__(self):
        super(PE_Encoder_with_FusionModule_v5, self).__init__()
        filters = [128, 256, 512]

        ### Feature Encoder
        self.inc = nn.Sequential(
            convlayer(1, filters[0], k=3, p=1, s=1),
            convlayer(filters[0], filters[0], k=3, p=1, s=1),
        )

        self.down1 = nn.Sequential(
            convlayer(filters[0], filters[0], k=2, p=0, s=2),
            convlayer(filters[0], filters[1], k=3, p=1, s=1),
            convlayer(filters[1], filters[1], k=3, p=1, s=1),
        )

        self.down2 = nn.Sequential(
            convlayer(filters[1], filters[1], k=2, p=0, s=2),
            convlayer(filters[1], filters[2], k=3, p=1, s=1),
            convlayer(filters[2], filters[2], k=3, p=1, s=1),
        )

        ### Attention maps Generator
        self.inc_mask_a = convlayer_maskv3(1, filters[0], k=3, p=1, s=1)
        self.inc_mask_b = convlayer_maskv3(filters[0], filters[0], k=3, p=1, s=1)

        self.down1a_mask = convlayer_maskv3(filters[0], filters[0], k=2, p=0, s=2)
        self.down1b_mask = convlayer_maskv3(filters[0], filters[1], k=3, p=1, s=1)
        self.down1c_mask = convlayer_maskv3(filters[1], filters[1], k=3, p=1, s=1)

        self.down2a_mask = convlayer_maskv3(filters[1], filters[1], k=2, p=0, s=2)
        self.down2b_mask = convlayer_maskv3(filters[1], filters[2], k=3, p=1, s=1)
        self.down2c_mask = convlayer_maskv3(filters[2], filters[2], k=3, p=1, s=1)

        ### Learnable Fusion Weights

        # scale 1
        self.sc1_fusion = FusionModule_v5(128)

        self.sc1_ws = nn.Parameter(torch.randn(128, 1, 1)).float()
        self.sc1_wl = nn.Parameter(torch.randn(128, 1, 1)).float()

        # scale 2
        self.sc2_fusion = FusionModule_v5(256)

        self.sc2_ws = nn.Parameter(torch.randn(256, 1, 1)).float()
        self.sc2_wl = nn.Parameter(torch.randn(256, 1, 1)).float()

        # scale 3
        self.sc3_fusion = FusionModule_v5(512)

        self.ws = nn.Parameter(torch.randn(512, 1, 1)).float()
        self.wl = nn.Parameter(torch.randn(512, 1, 1)).float()

    def forward(self, x_s, x_l, M_s, M_l):
        eps = 1e-5

        ### Features Extraction
        x1_s = self.inc(x_s)
        x1_l = self.inc(x_l)

        x2_s = self.down1(x1_s)
        x2_l = self.down1(x1_l)

        x3_s = self.down2(x2_s)
        x3_l = self.down2(x2_l)

        ### Construct Attention Maps
        mask_update1_s, _ = self.inc_mask_a(M_s)
        mask_update1_s, maskActiv1_s = self.inc_mask_b(mask_update1_s)

        mask_update1_l, _ = self.inc_mask_a(M_l)
        mask_update1_l, maskActiv1_l = self.inc_mask_b(mask_update1_l)

        mask_update2_s, _ = self.down1a_mask(mask_update1_s)
        mask_update2_s, _ = self.down1b_mask(mask_update2_s)
        mask_update2_s, maskActiv2_s = self.down1c_mask(mask_update2_s)

        mask_update2_l, _ = self.down1a_mask(mask_update1_l)
        mask_update2_l, _ = self.down1b_mask(mask_update2_l)
        mask_update2_l, maskActiv2_l = self.down1c_mask(mask_update2_l)

        mask_update3_s, _ = self.down2a_mask(mask_update2_s)
        mask_update3_s, _ = self.down2b_mask(mask_update3_s)
        _, maskActiv3_s = self.down2c_mask(mask_update3_s)

        mask_update3_l, _ = self.down2a_mask(mask_update2_l)
        mask_update3_l, _ = self.down2b_mask(mask_update3_l)
        _, maskActiv3_l = self.down2c_mask(mask_update3_l)

        ### Multi-exposed features fusion
        x1_spatial = (maskActiv1_s * x1_s + maskActiv1_l * x1_l) \
                     / (maskActiv1_s + maskActiv1_l + eps)
        x1_global = (self.sc1_ws * x1_s + self.sc1_wl * x1_l) / (self.sc1_ws + self.sc1_wl + eps)

        x1_A_global = self.sc1_fusion(x1_global, x1_spatial)
        x1_fused = x1_A_global * x1_global + (1. - x1_A_global) * x1_spatial

        x2_spatial = (maskActiv2_s * x2_s + maskActiv2_l * x2_l) \
                     / (maskActiv2_s + maskActiv2_l + eps)
        x2_global = (self.sc2_ws * x2_s + self.sc2_wl * x2_l) / (self.sc2_ws + self.sc2_wl + eps)

        x2_A_global = self.sc2_fusion(x2_global, x2_spatial)
        x2_fused = x2_A_global * x2_global + (1. - x2_A_global) * x2_spatial

        x3_spatial = (maskActiv3_s * x3_s + maskActiv3_l * x3_l) \
                     / (maskActiv3_s + maskActiv3_l + eps)
        x3_global = (self.ws * x3_s + self.wl * x3_l) / (self.ws + self.wl + eps)

        x3_A_global = self.sc3_fusion(x3_global, x3_spatial)
        x3_fused = x3_A_global * x3_global + (1. - x3_A_global) * x3_spatial

        return x3_fused, x2_fused, x1_fused

class ExRNet_FusionModule_v5(nn.Module):
    def __init__(self):
        super(ExRNet_FusionModule_v5, self).__init__()
        self.G1 = PE_Encoder_with_FusionModule_v5()

        ### Decoder
        filters = [128, 256, 512]
        self.up1 = deconvlayer(filters[2], filters[1])
        self.conv1 = nn.Sequential(
            convlayer(filters[1] * 2, filters[1], k=3, p=1, s=1),
            convlayer(filters[1], filters[1], k=3, p=1, s=1),
        )

        self.up2 = deconvlayer(filters[1], filters[0])
        self.conv2 = nn.Sequential(
            convlayer(filters[0] * 2, filters[0], k=3, p=1, s=1),
            convlayer(filters[0], filters[0], k=3, p=1, s=1),
        )

        self.outc = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.sig = nn.Sigmoid()

        self.bicubic = nn.Upsample(scale_factor=(2,1), mode = 'bicubic')

    def Exposure_Decompose(self, x):
        device = x.device
        BATCH, CHANNEL, HEIGHT, WIDTH = x.size()

        Es = torch.zeros((BATCH, CHANNEL, HEIGHT // 2, WIDTH)).to(device)
        Es[:, :, 0: HEIGHT // 2 : 2, :] = x[:, :, 0 : HEIGHT : 4, :]
        Es[:, :, 1: HEIGHT // 2 : 2, :] = x[:, :, 1 : HEIGHT : 4, :]

        El = torch.zeros((BATCH, CHANNEL, HEIGHT // 2, WIDTH)).to(device)
        El[:, :, 0 : HEIGHT // 2 : 2, :] = x[:, :, 2 : HEIGHT : 4, :]
        El[:, :, 1 : HEIGHT // 2 : 2, :] = x[:, :, 3 : HEIGHT : 4, :]

        return Es, El

    def forward(self, x, mask):
        # MASK: 0-well exposed regions; 1-poor exposed regions
        new_mask = 1. - mask

        B = new_mask.size(0)
        num_of_values = torch.sum(new_mask,dim = 2)
        num_of_values = torch.sum(num_of_values,dim = 2)
        for i in range(B):
            if num_of_values[i,:] < 102: # 10% of 32x32
                A = new_mask[i,:,:,:]
                A[A < 1e-5] = 0.1
                new_mask[i,:,:,:] = A

        #### Decompose Es, El
        Es, El = self.Exposure_Decompose(x)
        Ms, Ml = self.Exposure_Decompose(new_mask)

        #### Bicubic interpolation
        Es = self.bicubic(Es)
        El = self.bicubic(El)

        Ms = self.bicubic(Ms).clamp(min=0,max=1)
        Ml = self.bicubic(Ml).clamp(min=0,max=1)

        #### Encode features
        F_merged_3, F_merged_2, F_merged_1 = self.G1(Es, El, Ms, Ml)

        #### Decode features
        up1 = self.up1(F_merged_3)
        up1 = self.conv1(torch.cat((F_merged_2, up1), dim=1))

        up2 = self.up2(up1)
        up2 = self.conv2(torch.cat((F_merged_1, up2), dim=1))

        out = self.outc(up2)
        out = self.sig(out)

        return out
        
################################################################################## REBUTAL ECCV
################################################################################## REBUTAL ECCV
################################################################################## REBUTAL ECCV
################################################################################## REBUTAL ECCV
class PE_Encoder_with_FusionModule_v4_SingleMEF(nn.Module):
    def __init__(self):
        super(PE_Encoder_with_FusionModule_v4_SingleMEF, self).__init__()
        filters = [128, 256, 512]

        ### Feature Encoder
        self.inc = nn.Sequential(
            convlayer(1, filters[0], k=3, p=1, s=1),
            convlayer(filters[0], filters[0], k=3, p=1, s=1),
        )

        self.down1 = nn.Sequential(
            convlayer(filters[0], filters[0], k=2, p=0, s=2),
            convlayer(filters[0], filters[1], k=3, p=1, s=1),
            convlayer(filters[1], filters[1], k=3, p=1, s=1),
        )

        self.down2 = nn.Sequential(
            convlayer(filters[1], filters[1], k=2, p=0, s=2),
            convlayer(filters[1], filters[2], k=3, p=1, s=1),
            convlayer(filters[2], filters[2], k=3, p=1, s=1),
        )

        ### Attention maps Generator
        self.inc_mask_a = convlayer_maskv3(1, filters[0], k=3, p=1, s=1)
        self.inc_mask_b = convlayer_maskv3(filters[0], filters[0], k=3, p=1, s=1)

        ### Learnable Fusion Weights

        # scale 1
        self.sc1_fusion = FusionModule_v4(128)

        self.sc1_ws = nn.Parameter(torch.randn(128, 1, 1)).float()
        self.sc1_wl = nn.Parameter(torch.randn(128, 1, 1)).float()

        # scale 2
        self.sc2_fusion = nn.Conv2d(256 * 2, 256, kernel_size = 1, stride = 1, padding = 0)

        # scale 3
        self.sc3_fusion = nn.Conv2d(512 * 2, 512, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x_s, x_l, M_s, M_l):
        eps = 1e-5

        ### Features Extraction
        x1_s = self.inc(x_s)
        x1_l = self.inc(x_l)

        x2_s = self.down1(x1_s)
        x2_l = self.down1(x1_l)

        x3_s = self.down2(x2_s)
        x3_l = self.down2(x2_l)

        ### Construct Attention Maps
        mask_update1_s, _ = self.inc_mask_a(M_s)
        mask_update1_s, maskActiv1_s = self.inc_mask_b(mask_update1_s)

        mask_update1_l, _ = self.inc_mask_a(M_l)
        mask_update1_l, maskActiv1_l = self.inc_mask_b(mask_update1_l)

        ### Multi-exposed features fusion
        x1_spatial = (maskActiv1_s * x1_s + maskActiv1_l * x1_l) \
                     / (maskActiv1_s + maskActiv1_l + eps)
        x1_global = (self.sc1_ws * x1_s + self.sc1_wl * x1_l) / (self.sc1_ws + self.sc1_wl + eps)

        x1_A_global = self.sc1_fusion(x1_global, x1_spatial)
        x1_fused = x1_A_global * x1_global + (1. - x1_A_global) * x1_spatial

        x2_fused = self.sc2_fusion(torch.cat((x2_s, x2_l), dim = 1))

        x3_fused = self.sc3_fusion(torch.cat((x3_s, x3_l), dim = 1))

        return x3_fused, x2_fused, x1_fused

class PE_Encoder_with_FusionModule_v4_SingleMEFv2(nn.Module):
    def __init__(self):
        super(PE_Encoder_with_FusionModule_v4_SingleMEFv2, self).__init__()
        filters = [128, 256, 512]

        ### Feature Encoder
        self.inc = nn.Sequential(
            convlayer(1, filters[0], k=3, p=1, s=1),
            convlayer(filters[0], filters[0], k=3, p=1, s=1),
        )

        self.down1 = nn.Sequential(
            convlayer(filters[0], filters[0], k=2, p=0, s=2),
            convlayer(filters[0], filters[1], k=3, p=1, s=1),
            convlayer(filters[1], filters[1], k=3, p=1, s=1),
        )

        self.down2 = nn.Sequential(
            convlayer(filters[1], filters[1], k=2, p=0, s=2),
            convlayer(filters[1], filters[2], k=3, p=1, s=1),
            convlayer(filters[2], filters[2], k=3, p=1, s=1),
        )

        ### Attention maps Generator
        self.inc_mask_a = convlayer_maskv3(1, filters[0], k=3, p=1, s=1)
        self.inc_mask_b = convlayer_maskv3(filters[0], filters[0], k=3, p=1, s=1)

        ### Learnable Fusion Weights
        # scale 1
        self.sc1_fusion = FusionModule_v4(128)

        self.sc1_ws = nn.Parameter(torch.randn(128, 1, 1)).float()
        self.sc1_wl = nn.Parameter(torch.randn(128, 1, 1)).float()


    def forward(self, x_s, x_l, M_s, M_l):
        eps = 1e-5

        ### Features Extraction
        x1_s = self.inc(x_s)
        x1_l = self.inc(x_l)

        ### Construct Attention Maps
        mask_update1_s, _ = self.inc_mask_a(M_s)
        mask_update1_s, maskActiv1_s = self.inc_mask_b(mask_update1_s)

        mask_update1_l, _ = self.inc_mask_a(M_l)
        mask_update1_l, maskActiv1_l = self.inc_mask_b(mask_update1_l)

        ### Multi-exposed features fusion
        x1_spatial = (maskActiv1_s * x1_s + maskActiv1_l * x1_l) \
                     / (maskActiv1_s + maskActiv1_l + eps)
        x1_global = (self.sc1_ws * x1_s + self.sc1_wl * x1_l) / (self.sc1_ws + self.sc1_wl + eps)

        x1_A_global = self.sc1_fusion(x1_global, x1_spatial)
        x1_fused = x1_A_global * x1_global + (1. - x1_A_global) * x1_spatial

        ### Features Extraction
        x2_fused = self.down1(x1_fused)

        x3_fused = self.down2(x2_fused)


        return x3_fused, x2_fused, x1_fused

class ExRNet_FusionModule_v4_SingleMEF(nn.Module):
    def __init__(self):
        super(ExRNet_FusionModule_v4_SingleMEF, self).__init__()
        self.G1 = PE_Encoder_with_FusionModule_v4_SingleMEFv2()

        ### Decoder
        filters = [128, 256, 512]
        self.up1 = deconvlayer(filters[2], filters[1])
        self.conv1 = nn.Sequential(
            convlayer(filters[1] * 2, filters[1], k=3, p=1, s=1),
            convlayer(filters[1], filters[1], k=3, p=1, s=1),
        )

        self.up2 = deconvlayer(filters[1], filters[0])
        self.conv2 = nn.Sequential(
            convlayer(filters[0] * 2, filters[0], k=3, p=1, s=1),
            convlayer(filters[0], filters[0], k=3, p=1, s=1),
        )

        self.outc = nn.Conv2d(filters[0], 1, kernel_size=1)
        self.sig = nn.Sigmoid()

        self.bicubic = nn.Upsample(scale_factor=(2,1), mode = 'bicubic')

    def Exposure_Decompose(self, x):
        device = x.device
        BATCH, CHANNEL, HEIGHT, WIDTH = x.size()

        Es = torch.zeros((BATCH, CHANNEL, HEIGHT // 2, WIDTH)).to(device)
        Es[:, :, 0: HEIGHT // 2 : 2, :] = x[:, :, 0 : HEIGHT : 4, :]
        Es[:, :, 1: HEIGHT // 2 : 2, :] = x[:, :, 1 : HEIGHT : 4, :]

        El = torch.zeros((BATCH, CHANNEL, HEIGHT // 2, WIDTH)).to(device)
        El[:, :, 0 : HEIGHT // 2 : 2, :] = x[:, :, 2 : HEIGHT : 4, :]
        El[:, :, 1 : HEIGHT // 2 : 2, :] = x[:, :, 3 : HEIGHT : 4, :]

        return Es, El

    def forward(self, x, mask):
        # MASK: 0-well exposed regions; 1-poor exposed regions
        new_mask = 1. - mask

        B = new_mask.size(0)
        num_of_values = torch.sum(new_mask,dim = 2)
        num_of_values = torch.sum(num_of_values,dim = 2)
        for i in range(B):
            if num_of_values[i,:] < 102: # 10% of 32x32
                A = new_mask[i,:,:,:]
                A[A < 1e-5] = 0.1
                new_mask[i,:,:,:] = A

        #### Decompose Es, El
        Es, El = self.Exposure_Decompose(x)
        Ms, Ml = self.Exposure_Decompose(new_mask)

        #### Bicubic interpolation
        Es = self.bicubic(Es)
        El = self.bicubic(El)

        Ms = self.bicubic(Ms).clamp(min=0,max=1)
        Ml = self.bicubic(Ml).clamp(min=0,max=1)

        #### Encode features
        F_merged_3, F_merged_2, F_merged_1 = self.G1(Es, El, Ms, Ml)

        #### Decode features
        up1 = self.up1(F_merged_3)
        up1 = self.conv1(torch.cat((F_merged_2, up1), dim=1))

        up2 = self.up2(up1)
        up2 = self.conv2(torch.cat((F_merged_1, up2), dim=1))

        out = self.outc(up2)
        out = self.sig(out)

        return out

