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
from detection_vis_backend.networks.record import RecordEncoder, RecordDecoder, RecordEncoderNoLstm
from detection_vis_backend.networks.raddet import RadarResNet3D, YoloHead
from detection_vis_backend.networks.darod import RoIBBox, RoIPooling, RadarFeatures, Decoder, DARODBlock2D


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


# class RODNet(nn.Module):
#     def __init__(self, type, in_channels, n_class, stacked_num=1, mnet_cfg=None, dcn=True): 
#                     # win_size = 16, patch_size = 8, norm_layer = 'batch', hidden_size = 516, channels_features = (1,2,3,4),
#                     # receptive_field = [3,3,3,3], mlp_dim = 3072, num_layers = 12, num_heads = 12, out_head = 1):
#         super(RODNet, self).__init__()
#         self.nettype = type
#         mnet_cfg = tuple(mnet_cfg)

#         if self.nettype == 'CDC':
#             self.cdc = RadarVanilla(in_channels=in_channels, n_class=n_class)
#         elif self.nettype == 'HG':
#             self.stacked_hourglass = RadarStackedHourglass_HG(in_channels=in_channels, n_class=n_class, stacked_num=stacked_num) 
#         elif self.nettype == 'HGwI':
#             self.stacked_hourglass = RadarStackedHourglass_HGwI(in_channels=in_channels, n_class=n_class, stacked_num=stacked_num)
#         elif self.nettype == 'CDCv2':
#             self.dcn = dcn
#             if dcn:
#                 self.conv_op = DeformConvPack3D
#             else:
#                 self.conv_op = nn.Conv3d
#             if mnet_cfg is not None:
#                 in_chirps_mnet, out_channels_mnet = mnet_cfg
#                 assert in_channels == in_chirps_mnet
#                 self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
#                 self.with_mnet = True
#                 self.cdc = RadarVanilla(out_channels_mnet, n_class, use_mse_loss=False)
#             else:
#                 self.with_mnet = False
#                 self.cdc = RadarVanilla(in_channels, n_class, use_mse_loss=False)
#         elif self.nettype in ('HGv2', 'HGwIv2', 'HGwIv2_2d', 'hrformer2d', 'unetr_2d', 'unetr_2d_res_final', 'maxvit2'):
#             self.dcn = dcn
#             if dcn:
#                 self.conv_op = DeformConvPack3D
#             else:
#                 self.conv_op = nn.Conv3d
#             if mnet_cfg is not None:
#                 in_chirps_mnet, out_channels_mnet = mnet_cfg
#                 assert in_channels == in_chirps_mnet
#                 self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
#                 self.with_mnet = True   
#             else:
#                 self.with_mnet = False
                
#             if self.nettype == 'HGv2':
#                 if mnet_cfg is not None:
#                     self.stacked_hourglass = RadarStackedHourglass_HG(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
#                 else:
#                     self.stacked_hourglass = RadarStackedHourglass_HG(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
#             elif self.nettype == 'HGwIv2':
#                 if mnet_cfg is not None:
#                     self.stacked_hourglass = RadarStackedHourglass_HGwI(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op) 
#                 else:
#                     self.stacked_hourglass = RadarStackedHourglass_HGwI(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
#             # elif self.nettype == 'HGwIv2_2d':
#             #     if mnet_cfg is not None:
#             #         self.stacked_hourglass = RadarStackedHourglass_HGwI2d(out_channels_mnet, win_size, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
#             #     else:
#             #         self.stacked_hourglass = RadarStackedHourglass_HGwI2d(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
#             # elif self.nettype == 'hrformer2d':
#             #     if mnet_cfg is not None:
#             #         self.stacked_hourglass = RadarStackedHourglass_HRFormer2d(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op,
#             #                                             win_size = win_size, patch_size = patch_size, hidden_size = hidden_size, mlp_dim = mlp_dim,
#             #                                             num_layers = num_layers, receptive_field = receptive_field, norm_layer = norm_layer, 
#             #                                             num_heads = num_heads, channels_features = channels_features)
#             #     else:
#             #         self.stacked_hourglass = RadarStackedHourglass_HRFormer2d(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
#             # elif self.nettype == 'unetr_2d':
#             #     if mnet_cfg is not None:
#             #         self.stacked_hourglass = RadarStackedHourglass_UNETR2d(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op,
#             #                                             win_size = win_size, patch_size = patch_size, hidden_size = hidden_size, mlp_dim = mlp_dim,
#             #                                             num_layers = num_layers, receptive_field = receptive_field, norm_layer = norm_layer, num_heads = num_heads)
#             #     else:
#             #         self.stacked_hourglass = RadarStackedHourglass_UNETR2d(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)       
#             # elif self.nettype == 'unetr_2d_res_final':
#             #     if mnet_cfg is not None:
#             #         self.stacked_hourglass = RadarStackedHourglass_UNETR2dRes(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op,
#             #                                             win_size = win_size, patch_size = patch_size, hidden_size = hidden_size, mlp_dim = mlp_dim,
#             #                                             num_layers = num_layers, receptive_field = receptive_field, norm_layer = norm_layer,
#             #                                             out_head = out_head, num_heads = num_heads)
#             #     else:
#             #         self.stacked_hourglass = RadarStackedHourglass_UNETR2dRes(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)    
#             # elif self.nettype == 'maxvit2':
#             #     if mnet_cfg is not None:
#             #         self.stacked_hourglass = RadarStackedHourglass_MAXVIT2(out_channels_mnet, n_class, stacked_num=stacked_num,
#             #                                             win_size = win_size, patch_size = patch_size, hidden_size = hidden_size,
#             #                                             num_layers = num_layers, receptive_field = receptive_field,
#             #                                             out_head = out_head)
#             #     else:
#             #         self.stacked_hourglass = RadarStackedHourglass_MAXVIT2(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
#         else:
#             raise ValueError("Model type not supported")

#     def forward(self, x):
#         if self.nettype == 'CDC':
#             out = self.cdc(x)
#         elif self.nettype == 'HG':
#             out = self.stacked_hourglass(x)
#         elif self.nettype == 'HGwI':
#             out = self.stacked_hourglass(x)
#         elif self.nettype == 'CDCv2':
#             if self.with_mnet:
#                 x = self.mnet(x)
#             out = self.cdc(x)
#         elif self.nettype in ('HGv2', 'HGwIv2'):
#             if self.with_mnet:
#                 x = self.mnet(x)
#             out = self.stacked_hourglass(x)
#         return out
    

class RODNet_CDC(nn.Module):
    def __init__(self, in_channels, n_class):
        super(RODNet_CDC, self).__init__()
        self.cdc = RadarVanilla(in_channels, n_class, use_mse_loss=False)

    def forward(self, x):
        x = self.cdc(x)
        return x


class RODNet_CDCv2(nn.Module):
    def __init__(self, in_channels, n_class, mnet_cfg=None, dcn=True):
        super(RODNet_CDCv2, self).__init__()
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

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        x = self.cdc(x)
        return x

class RODNet_HG(nn.Module):
    def __init__(self, in_channels, n_class, stacked_num=2):
        super(RODNet_HG, self).__init__()
        self.stacked_hourglass = RadarStackedHourglass_HG(in_channels, n_class, stacked_num=stacked_num)

    def forward(self, x):
        out = self.stacked_hourglass(x)
        return out

class RODNet_HGwI(nn.Module):
    def __init__(self, in_channels, n_class, stacked_num=1):
        super(RODNet_HGwI, self).__init__()
        self.stacked_hourglass = RadarStackedHourglass_HGwI(in_channels, n_class, stacked_num=stacked_num)

    def forward(self, x):
        out = self.stacked_hourglass(x)
        return out

## If only consider all subtypes of RODNet paper/project, the following declaration way is more compact. 
## If also consider to include all subtypes in RadarFormer paper/project, choose the declaration way below. 
# class RODNet_v2Base(nn.Module):
#     def __init__(self, in_channels, n_class, stacked_num=2, mnet_cfg=None, dcn=True, hourglass_type=None):
#         super(RODNet_v2Base, self).__init__()
#         self.dcn = dcn
#         if dcn:
#             self.conv_op = DeformConvPack3D
#         else:
#             self.conv_op = nn.Conv3d
#         if mnet_cfg is not None:
#             in_chirps_mnet, out_channels_mnet = mnet_cfg
#             assert in_channels == in_chirps_mnet
#             self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
#             self.with_mnet = True
#             self.stacked_hourglass = hourglass_type(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
#         else:
#             self.with_mnet = False
#             self.stacked_hourglass = hourglass_type(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)

#     def forward(self, x):
#         if self.with_mnet:
#             x = self.mnet(x)
#         out = self.stacked_hourglass(x)
#         return out

# class RODNet_HGv2(RODNet_v2Base):
#     def __init__(self, in_channels, n_class, stacked_num=2, mnet_cfg=None, dcn=True, hourglass_type=RadarStackedHourglass_HG):
#         super().__init__(in_channels, n_class, stacked_num, mnet_cfg, dcn, hourglass_type)
    
# class RODNet_HGwIv2(RODNet_v2Base):
#     def __init__(self, in_channels, n_class, stacked_num=2, mnet_cfg=None, dcn=True, hourglass_type=RadarStackedHourglass_HGwI):
#         super().__init__(in_channels, n_class, stacked_num, mnet_cfg, dcn, hourglass_type)


class RODNet_v2Base(nn.Module):
    def __init__(self, in_channels, n_class, stacked_num=2, mnet_cfg=None, dcn=True, hourglass_type=None):
        super(RODNet_v2Base, self).__init__()
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
        else:
            self.with_mnet = False

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        out = self.stacked_hourglass(x)
        return out
    
class RODNet_HGv2(RODNet_v2Base):
    def __init__(self, in_channels, n_class, stacked_num=2, mnet_cfg=None, dcn=True, hourglass_type=None):
        super().__init__(in_channels, n_class, stacked_num, mnet_cfg, dcn)
        self.stacked_hourglass = RadarStackedHourglass_HG(in_channels if mnet_cfg is None else mnet_cfg[1], n_class, stacked_num=stacked_num, conv_op=self.conv_op)

class RODNet_HGwIv2(RODNet_v2Base):
    def __init__(self, in_channels, n_class, stacked_num=2, mnet_cfg=None, dcn=True, hourglass_type=None):
        super().__init__(in_channels, n_class, stacked_num, mnet_cfg, dcn)
        self.stacked_hourglass = RadarStackedHourglass_HGwI(in_channels if mnet_cfg is None else mnet_cfg[1], n_class, stacked_num=stacked_num, conv_op=self.conv_op)

# #Subtypes from RadarFormer: 'hrformer2d', 'unetr_2d', 'unetr_2d_res_final', 'maxvit2'
# class RadarFormer_hrformer2d(RODNet_v2Base):
#     def __init__(self, in_channels, n_class, stacked_num=1, mnet_cfg=None, dcn=True, win_size=16, patch_size = 8, 
#                     norm_layer = 'batch', hidden_size = 516, channels_features = (1,2,3,4), receptive_field = [3,3,3,3],
#                     mlp_dim = 3072, num_layers = 12, num_heads = 12):
#         super().__init__(in_channels, n_class, stacked_num, mnet_cfg, dcn)
#         self.stacked_hourglass = RadarStackedHourglass_HRFormer2d(in_channels if mnet_cfg is None else mnet_cfg[1], n_class, stacked_num=stacked_num, conv_op=self.conv_op,
#                                                                 win_size=win_size, patch_size=patch_size, hidden_size=hidden_size, mlp_dim=mlp_dim,
#                                                                 num_layers=num_layers, receptive_field=receptive_field, norm_layer=norm_layer, 
#                                                                 num_heads=num_heads, channels_features=channels_features)

# class RadarFormer_unetr2d(RODNet_v2Base):
#     def __init__(self, in_channels, n_class, stacked_num=1, mnet_cfg=None, dcn=True, win_size=16, patch_size=8, hidden_size=516, mlp_dim=3072,
#                     num_layers=12, receptive_field=[[3,3,3,3],[3,3,3,3]], norm_layer=12, num_heads=12):
#         super().__init__(in_channels, n_class, stacked_num, mnet_cfg, dcn)
#         self.stacked_hourglass = RadarStackedHourglass_unetr2d(in_channels if mnet_cfg is None else mnet_cfg[1], n_class, stacked_num=stacked_num, conv_op=self.conv_op, 
#                                                                 win_size = win_size, patch_size = patch_size, hidden_size = hidden_size, mlp_dim = mlp_dim,
#                                                                 num_layers = num_layers, receptive_field = receptive_field, norm_layer = norm_layer, 
#                                                                 num_heads = num_heads)

# class RadarFormer_maxvit2(RODNet_v2Base):
#     def __init__(self, in_channels, n_class, stacked_num=1, mnet_cfg=None, dcn=True, out_head = 1, win_size = 16, patch_size = 8, 
#                     hidden_size = 516, receptive_field = [[3,3,3,3],[3,3,3,3]], num_layers = 12):
#         super().__init__(in_channels, n_class, stacked_num, mnet_cfg, dcn)
#         self.stacked_hourglass = RadarStackedHourglass_maxvit2(in_channels if mnet_cfg is None else mnet_cfg[1], n_class, stacked_num=stacked_num,
#                                                                win_size = win_size, patch_size = patch_size, hidden_size = hidden_size,
#                                                                 num_layers = num_layers, receptive_field = receptive_field, out_head = out_head)


class RECORD(nn.Module):
    def __init__(self, encoder_config, decoder_config, in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class
        @param config: configuration file of the model
        @param alpha: expansion factor to modify the size of the model (default: 1.0)
        @param in_channels: number of input channels (default: 8)
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        @param n_class: number of classes (default: 3)
        @param shallow: load a shallow version of RECORD (fewer channels in the decoder)
        """
        super(RECORD, self).__init__()
        self.encoder = RecordEncoder(in_channels=in_channels, config=encoder_config, norm=norm)
        self.decoder = RecordDecoder(decoder_config, n_class=n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass RECORD model
        @param x: input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: ConfMap prediction of the last time step with shape (B, n_classes, H, W)
        """
        time_steps = x.shape[2]
        assert len(x.shape) == 5
        for t in range(time_steps):
            if t == 0:
                # Init hidden states if first time step of sliding window
                self.encoder.__init_hidden__()
            st_features_lstm1, st_features_lstm2, st_features_backbone = self.encoder(x[:, :, t])

        confmap_pred = self.decoder(st_features_lstm1, st_features_lstm2, st_features_backbone)
        return self.sigmoid(confmap_pred)
    

class RECORDNoLstm(nn.Module):
    def __init__(self, encoder_config, decoder_config,  in_channels=8, norm='layer', n_class=3):
        """
        RECurrent Online object detectOR (RECORD) model class for online inference
        @param config: configuration file of the model
        @param in_channels: number of input channels (default: 8)
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        @param n_class: number of classes (default: 3)
        """
        super(RECORDNoLstm, self).__init__()
        self.encoder = RecordEncoderNoLstm(config=encoder_config, in_channels=in_channels, norm=norm)
        self.decoder = RecordDecoder(config=decoder_config, n_class=n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass RECORD-OI model
        @param x: input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: ConfMap prediction of the last time step with shape (B, n_classes, H, W)
        """
        x3, x2, x1 = self.encoder(x)
        confmap_pred = self.decoder(x3, x2, x1)
        return self.sigmoid(confmap_pred)


class RECORDNoLstmMulti(RECORDNoLstm):
    def __init__(self, encoder_config, decoder_config, in_channels=96, norm='layer', n_class=3):
        super().__init__(encoder_config, decoder_config, in_channels=in_channels, norm=norm, n_class=n_class)


class MVRECORD(nn.Module):
    def __init__(self, encoder_ra_config, encoder_rd_config, encoder_ad_config, decoder_ra_config, decoder_rd_config, in_channels=1, n_class=4, norm='layer'):
        """
        Multi view RECurrent Online object detectOR (MV-RECORD) model class
        @param config: config dict to build the model
        @param n_frames: number of input frames (i.e. timesteps)
        @param in_channels: number of input channels (default: 1)
        @param n_classes: number of classes (default: 4)
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        """
        super(MVRECORD, self).__init__()
        self.n_class = n_class
        self.in_channels = in_channels

        # Backbone (encoder)
        self.rd_encoder = RecordEncoder(in_channels=self.in_channels, config=encoder_rd_config, norm=norm)
        self.ra_encoder = RecordEncoder(in_channels=self.in_channels, config=encoder_ra_config, norm=norm)
        self.ad_encoder = RecordEncoder(in_channels=self.in_channels, config=encoder_ad_config, norm=norm)

        # Temporal Multi View Skip Connections
        in_channels_skip_connection_lstm1 = encoder_rd_config['bottleneck_lstm1']['in_channels'] + \
                                            encoder_ad_config['bottleneck_lstm1']['in_channels'] + \
                                            encoder_ra_config['bottleneck_lstm1']['in_channels']
        # We project the concatenation of features to the initial #channels of each view (kernel_size = 1)
        self.skip_connection_lstm1_conv = nn.Conv2d(in_channels=in_channels_skip_connection_lstm1,
                                                    out_channels=encoder_rd_config['bottleneck_lstm1']['out_channels'],
                                                    kernel_size=1)

        in_channels_skip_connection_lstm2 = encoder_rd_config['bottleneck_lstm2']['in_channels'] + \
                                            encoder_ad_config['bottleneck_lstm2']['in_channels'] + \
                                            encoder_ra_config['bottleneck_lstm2']['in_channels']
        # We project the concatenation of features to the initial #channels of each view (kernel_size = 1)
        self.skip_connection_lstm2_conv = nn.Conv2d(in_channels=in_channels_skip_connection_lstm2,
                                                    out_channels=encoder_rd_config['bottleneck_lstm2']['out_channels'],
                                                    kernel_size=1)

        # We downsample the RA view on the azimuth dimension to match the size of AD and RD view
        self.down_sample_ra_view_skip_connection1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.up_sample_rd_ad_views_skip_connection1 = nn.Upsample(scale_factor=(1, 2))

        # Decoding
        self.rd_decoder = RecordDecoder(config=decoder_rd_config, n_class=self.n_class)
        self.ra_decoder = RecordDecoder(config=decoder_ra_config, n_class=self.n_class)

    def forward(self, inputs):
        """
        Forward pass MV-RECORD model
        @param x_rd: RD input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @param x_ra: RA input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @param x_ad: AD input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: RD and RA segmentation masks of the last time step with shape (B, n_class, H, W)
        """
        x_rd, x_ra, x_ad = inputs
        win_size = x_rd.shape[2]
        # Backbone
        for t in range(win_size):
            if t == 0:
                self.ra_encoder.__init_hidden__()
                self.rd_encoder.__init_hidden__()
                self.ad_encoder.__init_hidden__()

            st_features_backbone_rd, st_features_lstm2_rd, st_features_lstm1_rd = self.rd_encoder(x_rd[:, :, t])
            st_features_backbone_ra, st_features_lstm2_ra, st_features_lstm1_ra = self.ra_encoder(x_ra[:, :, t])
            st_features_backbone_ad, st_features_lstm2_ad, st_features_lstm1_ad = self.ad_encoder(x_ad[:, :, t])

        # Concat latent spaces of each view
        rd_ad_ra_latent_space = torch.cat((st_features_backbone_rd,
                                           st_features_backbone_ra,
                                           st_features_backbone_ad), dim=1)


        # Latent space for skip connection 2 - Range Doppler and Range Angle view (h_kskip_1 in the paper)
        # Concatenate
        latent_rd_ad_ra_skip_connection_2 = torch.cat((st_features_lstm2_rd,
                                                       st_features_lstm2_ra,
                                                       st_features_lstm2_ad), dim=1)
        # Reduce # channels
        latent_rd_ad_ra_skip_connection_2 = self.skip_connection_lstm2_conv(latent_rd_ad_ra_skip_connection_2)

        # Latent space for skip connection 1 (h_kskip_0 in the paper)
        # Skip connection for RD decoder - Down sample features map from RA view to match sizes of AD and RD views
        latent_skip_connection1_rd = torch.cat((st_features_lstm1_rd,
                                                self.down_sample_ra_view_skip_connection1(st_features_lstm1_ra),
                                                st_features_lstm1_ad), dim=1)
        latent_skip_connection1_rd = self.skip_connection_lstm1_conv(latent_skip_connection1_rd)

        # Skip connection for RA decoder - Up sample features maps from RD and AD view to match sizes of RA view
        latent_skip_connection1_ra = torch.cat((self.up_sample_rd_ad_views_skip_connection1(st_features_lstm1_rd),
                                  st_features_lstm1_ra,
                                  self.up_sample_rd_ad_views_skip_connection1(st_features_lstm1_ad)), dim=1)
        latent_skip_connection1_ra = self.skip_connection_lstm1_conv(latent_skip_connection1_ra)

        # Decode
        pred_rd = self.rd_decoder(rd_ad_ra_latent_space, latent_rd_ad_ra_skip_connection_2, latent_skip_connection1_rd)
        pred_ra = self.ra_decoder(rd_ad_ra_latent_space, latent_rd_ad_ra_skip_connection_2, latent_skip_connection1_ra)

        return {"rd": pred_rd, "ra": pred_ra}
    

class RADDet(nn.Module):
    def __init__(self, input_channels, n_class, num_anchors_layer):
        super(RADDet, self).__init__()

        self.backbone_stage = RadarResNet3D(input_channels)
        self.yolohead = YoloHead(num_anchors_layer, n_class, 256, 4) 
        
    def forward(self, x):
        features = self.backbone_stage(x)
        #print(f"backstone_stage passed. output feature shape: {features.shape}")        
        yolo_raw = self.yolohead(features)
        yolo_raw = yolo_raw.permute(0, 3, 4, 1, 2)
        #print(f"YoloHead passed. output shape: {yolo_raw.shape}")
        return yolo_raw
    

class DAROD(nn.Module):
    def __init__(self, input_size, rpn, fastrcnn, n_class, use_dropout, dropout_rate, use_bn, layout, dilation_rate, feature_map_shape, range_res, doppler_res):
        super(DAROD, self).__init__()
        self.input_size = input_size
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.total_labels = n_class
        self.fastrcnn_cfg = fastrcnn
        self.rpn_cfg = rpn
        self.feature_map_shape = feature_map_shape
        self.layout = layout
        self.dilation_rate = dilation_rate
        self.range_res = range_res
        self.doppler_res = doppler_res
        self.anchors = self.anchors_generation()

        if self.use_bn:
            block_norm = "group_norm"  # By default for this model
        else:
            block_norm = None

        # Backbone
        self.block1 = DARODBlock2D(in_channels=1, filters=64, padding="same", kernel_size=(3, 3),
                                   num_conv=2, dilation_rate=self.dilation_rate, activation="leaky_relu",
                                   block_norm=block_norm, pooling_size=(2, 2),
                                   pooling_strides=(2, 2), name="darod_block1")

        self.block2 = DARODBlock2D(in_channels=64, filters=128, padding="same", kernel_size=(3, 3),
                                   num_conv=2, dilation_rate=self.dilation_rate, activation="leaky_relu",
                                   block_norm=block_norm, pooling_size=(2, 1),
                                   pooling_strides=(2, 1), name="darod_block2")

        self.block3 = DARODBlock2D(in_channels=128, filters=256, padding="same", kernel_size=(3, 3),
                                   num_conv=3, dilation_rate=(1, 1), activation="leaky_relu",
                                   block_norm=block_norm, pooling_size=(2, 1),
                                   pooling_strides=(2, 1), name="darod_block3")
        
        # RPN
        self.rpn_conv = nn.Conv2d(in_channels=256, 
                          out_channels=self.rpn_cfg["rpn_channels"], 
                          kernel_size=self.rpn_cfg["rpn_window"], 
                          padding=(self.rpn_cfg["rpn_window"][0]//2, self.rpn_cfg["rpn_window"][1]//2))
        self.rpn_cls_output = nn.Conv2d(in_channels=self.rpn_cfg["rpn_channels"], 
                                out_channels=self.rpn_cfg["anchor_count"], 
                                kernel_size=1)
        self.rpn_reg_output = nn.Conv2d(in_channels=self.rpn_cfg["rpn_channels"], 
                                out_channels=4 * self.rpn_cfg["anchor_count"], 
                                kernel_size=1)

        # Fast RCNN
        self.roi_bbox = RoIBBox(self.anchors, self.fastrcnn_cfg["pre_nms_topn_train"], self.fastrcnn_cfg["pre_nms_topn_test"], 
                                self.fastrcnn_cfg["post_nms_topn_train"], self.fastrcnn_cfg["post_nms_topn_test"], self.rpn_cfg["rpn_nms_iou"], self.rpn_cfg["variances"])
        self.radar_features = RadarFeatures(self.input_size, self.range_res, self.doppler_res )
        self.roi_pooled = RoIPooling(self.fastrcnn_cfg["pooling_size"])

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(4098, self.fastrcnn_cfg["in_channels_1"]) 
        self.fc2 = nn.Linear(self.fastrcnn_cfg["in_channels_1"], self.fastrcnn_cfg["in_channels_2"])
        self.frcnn_cls = nn.Linear(self.fastrcnn_cfg["in_channels_2"], self.total_labels)
        self.frcnn_reg = nn.Linear(self.fastrcnn_cfg["in_channels_2"], self.total_labels * 4)

        self.decoder = Decoder(self.fastrcnn_cfg["variances_boxes"], self.total_labels, self.fastrcnn_cfg["frcnn_num_pred"], self.fastrcnn_cfg["box_nms_score"], self.fastrcnn_cfg["box_nms_iou"])

        if self.use_dropout:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        return 

    def forward(self, input):
        # print("---------------network------------------")
        # print(f"input: {input.shape}")
        x = self.block1(input)
        #print(f"after block1: {x.shape}")
        x = self.block2(x)
        #print(f"after block2: {x.shape}")
        x = self.block3(x)
        #print(f"after block3: {x.shape}")

        rpn_out = F.relu(self.rpn_conv(x))
        rpn_cls_pred = self.rpn_cls_output(rpn_out)
        rpn_delta_pred = self.rpn_reg_output(rpn_out)
        #print(f"rpn_out: {rpn_out.shape}")
        #print(f"rpn_cls_pred: {rpn_cls_pred.shape}")
        #print(f"rpn_delta_pred: {rpn_delta_pred.shape}")

        # if step == "rpn":
        #     return rpn_cls_pred, rpn_delta_pred

        roi_bboxes_out, roi_bboxes_scores = self.roi_bbox(rpn_delta_pred, rpn_cls_pred)
        #print(f"roi_bboxes_out: {roi_bboxes_out.shape}")
        #print(f"roi_bboxes_scores: {roi_bboxes_scores.shape}")
        roi_pooled_out = self.roi_pooled(x, roi_bboxes_out)
        #print(f"roi_pooled_out: {roi_pooled_out.shape}")

        output = roi_pooled_out.view(roi_pooled_out.size(0), roi_pooled_out.size(1), -1) # need to adjust
        features = self.radar_features(roi_bboxes_out, roi_bboxes_scores)
        output = torch.cat([output, features], dim=-1)

        # Reshape for applying layers:  (batch_size * time_steps, input_dim)         
        output_reshaped = output.view(-1, output.shape[2]) 
        output_reshaped = F.relu(self.fc1(output_reshaped))
        if self.use_dropout:
            output_reshaped = self.dropout(output_reshaped)
        output_reshaped = F.relu(self.fc2(output_reshaped))
        if self.use_dropout:
            output_reshaped = self.dropout(output_reshaped)
        frcnn_cls_pred_reshaped = self.frcnn_cls(output_reshaped)
        frcnn_reg_pred_reshaped = self.frcnn_reg(output_reshaped)
        # Reshape predictions back to original form: (batch_size, time_steps, output_dim)
        frcnn_cls_pred = frcnn_cls_pred_reshaped.view(output.shape[0], output.shape[1], -1)
        frcnn_reg_pred = frcnn_reg_pred_reshaped.view(output.shape[0], output.shape[1], -1)
        decoder_output = self.decoder([roi_bboxes_out, frcnn_reg_pred, F.softmax(frcnn_cls_pred, dim=-1)])

        rpn_cls_pred = rpn_cls_pred.permute(0, 2, 3, 1)
        rpn_delta_pred = rpn_delta_pred.permute(0, 2, 3, 1)
        # print(f"Network output:")
        # print(f"rpn_cls_pred: {rpn_cls_pred.shape}")
        # print(f"rpn_delta_pred: {rpn_delta_pred.shape}")
        # print(f"frcnn_cls_pred: {frcnn_cls_pred.shape}")
        # print(f"frcnn_reg_pred: {frcnn_reg_pred.shape}")
        # print(f"roi_bboxes_out: {roi_bboxes_out.shape}")
        # print("---------------network------------------")
        return {"rpn_cls_pred": rpn_cls_pred, "rpn_delta_pred": rpn_delta_pred, "roi_bboxes_out": roi_bboxes_out,
                "frcnn_cls_pred": frcnn_cls_pred, "frcnn_reg_pred": frcnn_reg_pred, "decoder_output": decoder_output}

    
    def anchors_generation(self,):
        """
        Generating anchors boxes with different shapes centered on
        each pixel.
        :param config: config file with input data, anchor scales/ratios
        :param train: anchors for training or inference
        :return: base_anchors = (anchor_count * fm_h * fm_w, [y1, x1, y2, x2]
        """
        in_height, in_width = self.feature_map_shape
        scales, ratios = self.rpn_cfg["anchor_scales"], self.rpn_cfg["anchor_ratios"]
        num_scales, num_ratios = len(scales), len(ratios)
        scale_tensor = torch.tensor(scales, dtype=torch.float32)
        ratio_tensor = torch.tensor(ratios, dtype=torch.float32)
        boxes_per_pixel = (num_scales + num_ratios - 1)
        #
        offset_h, offset_w = 0.5, 0.5
        steps_h = 1.0 / in_height
        steps_w = 1.0 / in_width

        # Generate all center points for the anchor boxes
        center_h = (torch.arange(in_height, dtype=torch.float32) + offset_h) * steps_h
        center_w = (torch.arange(in_width, dtype=torch.float32) + offset_w) * steps_w
        shift_y, shift_x = torch.meshgrid(center_h, center_w)
        shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

        # Generate "boxes_per_pixel" number of heights and widths that are later
        # used to create anchor box corner coordinates xmin, xmax, ymin ymax
        w = torch.cat((scale_tensor * torch.sqrt(ratio_tensor[0]), scales[0] * torch.sqrt(ratio_tensor[1:])),
                    dim=-1) * in_height / in_width
        h = torch.cat((scale_tensor / torch.sqrt(ratio_tensor[0]), scales[0] / torch.sqrt(ratio_tensor[1:])),
                    dim=-1) * in_height / in_width

        # Divide by 2 to get the half height and half width
        #anchor_manipulation = torch.tile(torch.transpose(torch.stack([-w, -h, w, h]), 0), (in_height * in_width, 1)) / 2
        anchor_manipulation = torch.stack([-w, -h, w, h]).t().repeat(in_height * in_width, 1) / 2

        # Each center point will have `boxes_per_pixel` number of anchor boxes, so
        # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
        out_grid = torch.repeat_interleave(torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1), boxes_per_pixel, dim=0)
        output = out_grid + anchor_manipulation
        # if train:
        mask = ((output <= 0.0) | (output >= 1.0)).any(dim=-1, keepdim=True)
        mask_expanded = mask.expand_as(output)
        output[mask_expanded] = 0.0
        # else:
        #     output = torch.clamp(output, min=0.0, max=1.0)
        return output

