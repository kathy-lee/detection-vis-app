import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence

# import sys
# sys.path.insert(0, '/home/kangle/projects/detection-vis-app')

from detection_vis_backend.networks.fftradnet import FPN_BackBone, RangeAngle_Decoder, Detection_Header, BasicBlock
from detection_vis_backend.networks.rodnet import RadarVanilla, RadarStackedHourglass_HG, RadarStackedHourglass_HGwI, DeformConvPack3D, MNet 


class NetworkFactory:
    _instances = {}
    _singleton_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._singleton_instance:
            cls._singleton_instance = super().__new__(cls)
        return cls._singleton_instance

    def get_instance(self, class_name, params=None):
        if class_name not in self._instances:
            # Fetch the class from globals, create a singleton instance
            cls = globals()[class_name]
            if params:
                self._instances[class_name] = cls(**params)  # Pass the dict as arguments if it's provided
            else:
                self._instances[class_name] = cls()
        return self._instances[class_name]
    


class FFTRadNet(nn.Module):
    def __init__(self,mimo_layer,channels,blocks,regression_layer = 2, detection_head=True,segmentation_head=True):
        super(FFTRadNet, self).__init__()
    
        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.FPN = FPN_BackBone(num_block=blocks,channels=channels,block_expansion=4, mimo_layer = mimo_layer,use_bn = True)
        self.RA_decoder = RangeAngle_Decoder()
        
        if(self.detection_head):
            self.detection_header = Detection_Header(input_angle_size=channels[3]*4,reg_layer=regression_layer)

        if(self.segmentation_head):
            self.freespace = nn.Sequential(BasicBlock(256,128),BasicBlock(128,64),nn.Conv2d(64, 1, kernel_size=1))

    def forward(self,x):
                       
        out = {'Detection':[],'Segmentation':[]}
        
        features= self.FPN(x)
        RA = self.RA_decoder(features)

        if(self.detection_head):
            out['Detection'] = self.detection_header(RA)

        if(self.segmentation_head):
            Y =  F.interpolate(RA, (256, 224))
            out['Segmentation'] = self.freespace(Y)
        
        return out
    

class RODNet(nn.Module):
    def __init__(self, type, in_channels, n_class, stacked_num=1, mnet_cfg=None, dcn=True):
        super(RODNet, self).__init__()
        self.nettype = type
        mnet_cfg = tuple(mnet_cfg)

        if self.nettype == 'CDC':
            self.cdc = RadarVanilla(in_channels=in_channels, n_class=n_class)
        elif self.nettype == 'HG':
            self.stacked_hourglass = RadarStackedHourglass_HG(in_channels=in_channels, n_class=n_class, stacked_num=stacked_num) 
        elif self.nettype == 'HGwI':
            self.stacked_hourglass = RadarStackedHourglass_HGwI(in_channels=in_channels, n_class=n_class, stacked_num=stacked_num)
        elif self.nettype == 'CDCv2':
            self.dcn = dcn
            if dcn:
                self.conv_op = DeformConvPack3D
            else:
                self.conv_op = nn.Conv3d
            if mnet_cfg is not None:
                in_chirps_mnet, out_channels_mnet = mnet_cfg
                assert in_channels == in_chirps_mnet
                self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
                self.with_mnet = True
                self.cdc = RadarVanilla(out_channels_mnet, n_class, use_mse_loss=False)
            else:
                self.with_mnet = False
                self.cdc = RadarVanilla(in_channels, n_class, use_mse_loss=False)
        elif self.nettype == 'HGv2':
            self.dcn = dcn
            if dcn:
                self.conv_op = DeformConvPack3D
            else:
                self.conv_op = nn.Conv3d
            if mnet_cfg is not None:
                in_chirps_mnet, out_channels_mnet = mnet_cfg
                assert in_channels == in_chirps_mnet
                self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
                self.with_mnet = True
                self.stacked_hourglass = RadarStackedHourglass_HG(out_channels_mnet, n_class, stacked_num=stacked_num,
                                                            conv_op=self.conv_op)
            else:
                self.with_mnet = False
                self.stacked_hourglass = RadarStackedHourglass_HG(in_channels, n_class, stacked_num=stacked_num,
                                                            conv_op=self.conv_op)
        elif self.nettype == 'HGwIv2':
            self.dcn = dcn
            if dcn:
                self.conv_op = DeformConvPack3D
            else:
                self.conv_op = nn.Conv3d
            if mnet_cfg is not None:
                in_chirps_mnet, out_channels_mnet = mnet_cfg
                self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
                self.with_mnet = True
                self.stacked_hourglass = RadarStackedHourglass_HGwI(out_channels_mnet, n_class, stacked_num=stacked_num,
                                                            conv_op=self.conv_op)
            else:
                self.with_mnet = False
                self.stacked_hourglass = RadarStackedHourglass_HGwI(in_channels, n_class, stacked_num=stacked_num,
                                                            conv_op=self.conv_op)
        else:
            raise ValueError("Model type not supported")

    def forward(self, x):
        if self.nettype == 'CDC':
            out = self.cdc(x)
        elif self.nettype == 'HG':
            out = self.stacked_hourglass(x)
        elif self.nettype == 'HGwI':
            out = self.stacked_hourglass(x)
        elif self.nettype == 'CDCv2':
            if self.with_mnet:
                x = self.mnet(x)
            out = self.cdc(x)
        elif self.nettype == 'HGv2':
            if self.with_mnet:
                x = self.mnet(x)
            out = self.stacked_hourglass(x)
        else:  # nettype: HGwIv2
            if self.with_mnet: 
                x = self.mnet(x)
            out = self.stacked_hourglass(x)
        return out