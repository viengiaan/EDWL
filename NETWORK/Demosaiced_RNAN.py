import NETWORK.Demosaiced_RNAN_common as common

import torch.nn as nn

import torch

def make_model(args, parent=False):
    return RNAN(args)


### RNAN
### residual attention + downscale upscale + denoising
class _ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_ResGroup, self).__init__()
        modules_body = []

        modules_body.append(
            common.ResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True),
                                          res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res


### nonlocal residual attention + downscale upscale + denoising
class _NLResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_NLResGroup, self).__init__()
        modules_body = []
        modules_body.append(
            common.NLResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True),
                                            res_scale=1))

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res


class RNAN(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(RNAN, self).__init__()

        n_resgroup = 10
        # n_resblock = args.n_resblocks
        n_feats = 64
        kernel_size = 3
        # reduction = args.reduction
        # scale = args.scale[0]
        act = nn.ReLU(True)

        # define head module
        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module

        modules_body_nl_low = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=1)]
        modules_body = [
            _ResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=1) \
            for _ in range(n_resgroup - 2)]
        modules_body_nl_high = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=1)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            conv(n_feats, 3, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body_nl_low = nn.Sequential(*modules_body_nl_low)
        self.body = nn.Sequential(*modules_body)
        self.body_nl_high = nn.Sequential(*modules_body_nl_high)
        self.tail = nn.Sequential(*modules_tail)

    def Bayer_to_RGB(self, img):
        B, C, H, W = img.size()
        device = img.device
        out = torch.zeros((B, 3, H, W)).to(device)
        out[:,0,0 : H : 2, 0 : W : 2] = img[:,0,0 : H : 2, 0 : W : 2]
        out[:,1,0 : H : 2, 1 : W : 2] = img[:,0,0 : H : 2, 1 : W : 2]
        out[:,1,1 : H : 2, 0 : W : 2] = img[:,0,1 : H : 2, 0 : W : 2]
        out[:,2,1 : H : 2, 1 : W : 2] = img[:,0,1 : H : 2, 1 : W : 2]

        return out

    def forward(self, img):
        img_RGB = self.Bayer_to_RGB(img)

        feats_shallow = self.head(img_RGB)

        res = self.body_nl_low(feats_shallow)
        res = self.body(res)
        res = self.body_nl_high(res)

        res_main = self.tail(res)

        res_clean = img_RGB + res_main

        return res_clean

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
