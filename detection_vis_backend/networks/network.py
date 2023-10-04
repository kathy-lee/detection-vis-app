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
from detection_vis_backend.networks.rodnet import RadarVanilla, RadarStackedHourglass_HG, RadarStackedHourglass_HGwI, DeformConvPack3D, MNet, RadarStackedHourglass_HGwI2d
from detection_vis_backend.networks.record import RecordEncoder, RecordDecoder, RecordEncoderNoLstm


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
                    # win_size = 16, patch_size = 8, norm_layer = 'batch', hidden_size = 516, channels_features = (1,2,3,4),
                    # receptive_field = [3,3,3,3], mlp_dim = 3072, num_layers = 12, num_heads = 12, out_head = 1):
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
        elif self.nettype in ('HGv2', 'HGwIv2', 'HGwIv2_2d', 'hrformer2d', 'unetr_2d', 'unetr_2d_res_final', 'maxvit2'):
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
                
            if self.nettype == 'HGv2':
                if mnet_cfg is not None:
                    self.stacked_hourglass = RadarStackedHourglass_HG(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
                else:
                    self.stacked_hourglass = RadarStackedHourglass_HG(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
            elif self.nettype == 'HGwIv2':
                if mnet_cfg is not None:
                    self.stacked_hourglass = RadarStackedHourglass_HGwI(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op) 
                else:
                    self.stacked_hourglass = RadarStackedHourglass_HGwI(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
            # elif self.nettype == 'HGwIv2_2d':
            #     if mnet_cfg is not None:
            #         self.stacked_hourglass = RadarStackedHourglass_HGwI2d(out_channels_mnet, win_size, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
            #     else:
            #         self.stacked_hourglass = RadarStackedHourglass_HGwI2d(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
            # elif self.nettype == 'hrformer2d':
            #     if mnet_cfg is not None:
            #         self.stacked_hourglass = RadarStackedHourglass_HRFormer2d(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op,
            #                                             win_size = win_size, patch_size = patch_size, hidden_size = hidden_size, mlp_dim = mlp_dim,
            #                                             num_layers = num_layers, receptive_field = receptive_field, norm_layer = norm_layer, 
            #                                             num_heads = num_heads, channels_features = channels_features)
            #     else:
            #         self.stacked_hourglass = RadarStackedHourglass_HRFormer2d(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
            # elif self.nettype == 'unetr_2d':
            #     if mnet_cfg is not None:
            #         self.stacked_hourglass = RadarStackedHourglass_UNETR2d(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op,
            #                                             win_size = win_size, patch_size = patch_size, hidden_size = hidden_size, mlp_dim = mlp_dim,
            #                                             num_layers = num_layers, receptive_field = receptive_field, norm_layer = norm_layer, num_heads = num_heads)
            #     else:
            #         self.stacked_hourglass = RadarStackedHourglass_UNETR2d(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)       
            # elif self.nettype == 'unetr_2d_res_final':
            #     if mnet_cfg is not None:
            #         self.stacked_hourglass = RadarStackedHourglass_UNETR2dRes(out_channels_mnet, n_class, stacked_num=stacked_num, conv_op=self.conv_op,
            #                                             win_size = win_size, patch_size = patch_size, hidden_size = hidden_size, mlp_dim = mlp_dim,
            #                                             num_layers = num_layers, receptive_field = receptive_field, norm_layer = norm_layer,
            #                                             out_head = out_head, num_heads = num_heads)
            #     else:
            #         self.stacked_hourglass = RadarStackedHourglass_UNETR2dRes(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)    
            # elif self.nettype == 'maxvit2':
            #     if mnet_cfg is not None:
            #         self.stacked_hourglass = RadarStackedHourglass_MAXVIT2(out_channels_mnet, n_class, stacked_num=stacked_num,
            #                                             win_size = win_size, patch_size = patch_size, hidden_size = hidden_size,
            #                                             num_layers = num_layers, receptive_field = receptive_field,
            #                                             out_head = out_head)
            #     else:
            #         self.stacked_hourglass = RadarStackedHourglass_MAXVIT2(in_channels, n_class, stacked_num=stacked_num, conv_op=self.conv_op)
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
        elif self.nettype in ('HGv2', 'HGwIv2', 'HGwIv2_2d', 'hrformer2d', 'unetr_2d', 'unetr_2d_res_final', 'maxvit2'):
            if self.with_mnet:
                x = self.mnet(x)
            out = self.stacked_hourglass(x)
        return out
    

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


class MVRecord(nn.Module):
    def __init__(self, config, n_frames, in_channels=1, n_classes=4, norm='layer'):
        """
        Multi view RECurrent Online object detectOR (MV-RECORD) model class
        @param config: config dict to build the model
        @param n_frames: number of input frames (i.e. timesteps)
        @param in_channels: number of input channels (default: 1)
        @param n_classes: number of classes (default: 4)
        @param norm: type of normalisation (default: LayerNorm). Other normalisation are not supported yet.
        """
        super(MVRecord, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.n_frames = n_frames

        # Backbone (encoder)
        self.rd_encoder = RecordEncoder(in_channels=self.in_channels, config=config['encoder_rd_config'], norm=norm)
        self.ra_encoder = RecordEncoder(in_channels=self.in_channels, config=config['encoder_ra_config'], norm=norm)
        self.ad_encoder = RecordEncoder(in_channels=self.in_channels, config=config['encoder_ad_config'], norm=norm)

        # Temporal Multi View Skip Connections
        in_channels_skip_connection_lstm1 = config['encoder_rd_config']['bottleneck_lstm1']['in_channels'] + \
                                            config['encoder_ad_config']['bottleneck_lstm1']['in_channels'] + \
                                            config['encoder_ra_config']['bottleneck_lstm1']['in_channels']
        # We project the concatenation of features to the initial #channels of each view (kernel_size = 1)
        self.skip_connection_lstm1_conv = nn.Conv2d(in_channels=in_channels_skip_connection_lstm1,
                                                    out_channels=config['encoder_rd_config']['bottleneck_lstm1']['out_channels'],
                                                    kernel_size=1)

        in_channels_skip_connection_lstm2 = config['encoder_rd_config']['bottleneck_lstm2']['in_channels'] + \
                                            config['encoder_ad_config']['bottleneck_lstm2']['in_channels'] + \
                                            config['encoder_ra_config']['bottleneck_lstm2']['in_channels']
        # We project the concatenation of features to the initial #channels of each view (kernel_size = 1)
        self.skip_connection_lstm2_conv = nn.Conv2d(in_channels=in_channels_skip_connection_lstm2,
                                                    out_channels=config['encoder_rd_config']['bottleneck_lstm2']['out_channels'],
                                                    kernel_size=1)

        # We downsample the RA view on the azimuth dimension to match the size of AD and RD view
        self.down_sample_ra_view_skip_connection1 = nn.AvgPool2d(kernel_size=(1, 2))
        self.up_sample_rd_ad_views_skip_connection1 = nn.Upsample(scale_factor=(1, 2))

        # Decoding
        self.rd_decoder = RecordDecoder(config=config['decoder_rd_config'], n_class=self.n_classes)
        self.ra_decoder = RecordDecoder(config=config['decoder_ra_config'], n_class=self.n_classes)

    def forward(self, inputs):
        """
        Forward pass MV-RECORD model
        @param x_rd: RD input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @param x_ra: RA input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @param x_ad: AD input tensor with shape (B, C, T, H, W) where T is the number of timesteps
        @return: RD and RA segmentation masks of the last time step with shape (B, n_class, H, W)
        """
        x_rd, x_ra, x_ad = inputs
        # Backbone
        for t in range(self.n_frames):
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