import time, math, glob
import torch
import scipy.io
import numpy as np

from NETWORK.Demosaiced_RNAN import RNAN
from NETWORK.Proposed_Network import ExRNet_FusionModule_v4
from NETWORK.Network_subimages import Feature_Selection, Fusion_Net

from TOOLS.ulti import Convert2LogDomain_norm, Convert2Luma, Convert2Rad_from_LogDomain, mat2tensor
from TOOLS.ulti_v3 import Generate_HDR_image_with_GaussianWeightv3, Torchtensor2Array

#################################################### KALANTARI DATASET
num_of_images = 12
image_list = sorted(glob.glob("/Kalantari/input" + "/*.*"))

image_savename = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


softmask_list = sorted(glob.glob("/Kalantari/softmask" + "/*.*"))

###################################################
device = "cuda:0"

########################################################## Proposed
############### 32x32
net_path_ExRNet = 'WEIGHTS_ECCV2022/Proposed/ExRNet/net_84.pth'
net_path_DINet = 'WEIGHTS_ECCV2022/Proposed/DINet/net_84.pth'
net_path_FusionNet = 'WEIGHTS_ECCV2022/Proposed/FusionNet/net_84.pth'
net_path = 'WEIGHTS_ECCV2022/Proposed/Color/net_84.pth'

############################################# 
save_path = 'test_results/Kalantari/Proposed/'

with torch.no_grad():
    ######## Proposed
    ExRNet = ExRNet_FusionModule_v4()
    state_dict = torch.load(net_path_ExRNet, map_location = lambda s, l: s)
    ExRNet.load_state_dict(state_dict)
    ExRNet.eval()
    ExRNet.to(device)

    DINet = Feature_Selection()
    state_dict = torch.load(net_path_DINet, map_location = lambda s, l: s)
    DINet.load_state_dict(state_dict)
    DINet.eval()
    DINet.to(device)

    FusionNet = Fusion_Net()
    state_dict = torch.load(net_path_FusionNet, map_location = lambda s, l: s)
    FusionNet.load_state_dict(state_dict)
    FusionNet.eval()
    FusionNet.to(device)

    HDR_net = RNAN() # Zhang 2020
    state_dict = torch.load(net_path, map_location = lambda s, l: s)
    HDR_net.load_state_dict(state_dict)
    HDR_net.eval()
    HDR_net.to(device)

    count = 0
    for image_name in image_list:
        print('Image: %d' %(count + 1))

        # image_name = image_list[i]
        input = mat2tensor(image_name, 'E_hat', channel=1)
        input = input.float()

        softmask_name = softmask_list[count]
        softmask = mat2tensor(softmask_name, 'SoftMask', channel = 1)
        softmask = softmask.float()

        ### Proposed
        input_log = Convert2LogDomain_norm(input)
        # 32 x 32
        HDR = Generate_HDR_image_with_GaussianWeightv3(input_log, softmask, ExRNet, DINet, FusionNet, HDR_net, device, size_patch=32, stride=16)

        HDR = Torchtensor2Array(HDR) # {Proposed}

        name = str(count + 1).zfill(6)
        # name = image_savename[count]
        scipy.io.savemat(save_path + name + '.mat', mdict={'HDR': HDR}) # {Proposed}

        count = count + 1
#
# net_path = '/media/vgan/00a91feb-14ee-4d63-89c0-1bb1b7e57b8a/LOCAL/CVPR_2022_Networks/Proposed/BJDD/64/Color/net_89.pth'
# save_path = 'test_data/HDR-Eye/MergeNet_v2/BJDD_64/'
