import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from math import log10, exp
import mat73

def mat2tensor(path, matname, channel):
    img = scipy.io.loadmat(path)
    output = img.get(matname)

    if channel > 1:
        output = output.swapaxes(0, 2).swapaxes(1, 2)

    out = torch.from_numpy(output)

    if channel > 1:
        height = out.shape[1]
        width = out.shape[2]
    else:
        height = out.shape[0]
        width = out.shape[1]

    out = torch.reshape(out, (1, channel, height, width))

    return out

def mat73_2tensor(path, matname, channel):
    img = mat73.loadmat(path)
    output = img.get(matname)

    if channel > 1:
        output = output.swapaxes(0, 2).swapaxes(1, 2)

    out = torch.from_numpy(output)

    if channel > 1:
        height = out.shape[1]
        width = out.shape[2]
    else:
        height = out.shape[0]
        width = out.shape[1]

    out = torch.reshape(out, (1, channel, height, width))

    return out

def Extract_raw_image(input):
    (batch, _, h, w) = input.size()
    DEVICE = input.device
    output = torch.zeros((batch, 1, h, w)).to(DEVICE)

    # RGGB bayer
    # Red channel
    output[:, 0, 0 : h : 2, 0 : w : 2] = input[:, 0, 0 : h : 2, 0 : w : 2]
    # 2 Green channels
    output[:, 0, 0 : h : 2, 1 : w : 2] = input[:, 1, 0 : h : 2, 1 : w : 2]
    output[:, 0, 1 : h : 2, 0 : w : 2] = input[:, 1, 1 : h : 2, 0 : w : 2]
    # Blue channels
    output[:, 0, 1 : h : 2, 1 : w : 2] = input[:, 2, 1 : h : 2, 1 : w : 2]

    return output

def reshape_numpy_swapaxes(inputs, h, w):
    outputs = torch.abs(inputs.cpu())
    outputs = torch.reshape(outputs, (1, h, w))
    outputs = outputs.detach().numpy()
    outputs = outputs.swapaxes(0, 2).swapaxes(0, 1)

    return outputs

##############################################################################
##############################################################################
def Convert2Luma(inputs):
    a = 17.554
    b = 826.81
    c = 0.10013
    d = -884.17
    e = 209.16
    f = -731.28
    yl = 5.6046
    yh = 10469
    
    outputs = torch.zeros_like(inputs)
    outputs[inputs < yl] = a * inputs[inputs < yl]
    outputs[(inputs >= yl) & (inputs < yh)] = b * torch.pow(inputs[(inputs >= yl) & (inputs < yh)], c) + d
    outputs[inputs >= yh] = e * torch.log(inputs[inputs >= yh]) + f
    
    outputs = outputs / 4095.
      
    return outputs

def Convert2Rad(L):
    a = 0.0569
    b = 7.3014e-30
    c = 9.9872
    d = 884.17
    e = 32.994
    f = 0.0047811
    ll = 98.381
    lh = 1204.7

    L = L * 4095.

    y = torch.zeros_like(L)
    y[L < ll] = a * L[L < ll]
    y[(L >= ll) & (L < lh)] = b * torch.pow(L[(L >= ll) & (L < lh)] + d, c)
    y[L >= lh] = e * torch.exp(f * L[L >= lh])

    return y

##############################################################################
##############################################################################
def Convert2LogDomain(inputs):
    outputs = torch.log(1 + 5000 * inputs) / np.log(1 + 5000)
    outputs = outputs

    return outputs

def Convert2LogDomain_norm(inputs):
    outputs = torch.log(1 + 5000 * inputs) / np.log(1 + 5000)
    outputs = outputs / 3.5

    return outputs

def Convert2Rad_from_LogDomain(inputs):
    inputs = inputs * 3.5
    outputs = torch.exp(inputs * np.log(1 + 5000)) - 1
    outputs = outputs / 5000

    return outputs

def Extract_SubImages_for_BAYER(input):
    (batch, _, h, w) = input.size()
    DEVICE = input.device
    output = torch.zeros((batch, 4, h // 2, w // 2)).to(DEVICE).float()

    # RGGB bayer
    # Red channel
    output[:, 0, :, :] = input[:, 0, 0 : h : 2, 0 : w : 2]
    # 2 Green channels
    output[:, 1, :, :] = input[:, 0, 0 : h : 2, 1 : w : 2]
    output[:, 2, :, :] = input[:, 0, 1 : h : 2, 0 : w : 2]
    # Blue channels
    output[:, 3, :, :] = input[:, 0, 1 : h : 2, 1 : w : 2]

    return output
