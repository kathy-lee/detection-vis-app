import torch
import torch.nn as nn
import math

from detection_vis_backend.networks.ops.dcn import DeformConvPack3D


class MNet(nn.Module):
    def __init__(self, in_chirps, out_channels, conv_op=None):
        super(MNet, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        if conv_op is None:
            conv_op = nn.Conv3d
        self.conv_op = conv_op

        self.t_conv3d = conv_op(in_channels=2, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                                padding=(1, 0, 0))
        t_conv_out = math.floor((in_chirps + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        self.t_maxpool = nn.MaxPool3d(kernel_size=(t_conv_out, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_win = self.t_conv3d(x[:, :, win, :, :, :])
            x_win = self.t_maxpool(x_win)
            x_win = x_win.view(batch_size, self.out_channels, w, h)
            x_out[:, :, win, ] = x_win
        return x_out


class RadarVanilla(nn.Module):

    def __init__(self, in_channels, n_class, use_mse_loss=False):
        super(RadarVanilla, self).__init__()
        self.encoder = RODEncode_vanilla(in_channels=in_channels)
        self.decoder = RODDecode_vanilla(n_class=n_class)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if not self.use_mse_loss:
            x = self.sigmoid(x)
        return x


class RODEncode_vanilla(nn.Module):

    def __init__(self, in_channels=2):
        super(RODEncode_vanilla, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=in_channels, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)
        return x


class RODDecode_vanilla(nn.Module):

    def __init__(self, n_class):
        super(RODDecode_vanilla, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        # self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
        #                                   radar_configs['ramap_asize']), mode='nearest')

    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.convt3(x)  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        return x
    

class RadarStackedHourglass_HG(nn.Module):

    def __init__(self, in_channels, n_class, stacked_num=1, conv_op=None, use_mse_loss=False):
        super(RadarStackedHourglass_HG, self).__init__()
        self.stacked_num = stacked_num
        if conv_op is None:
            self.conv1a = nn.Conv3d(in_channels=in_channels, out_channels=32,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        else:
            self.conv1a = conv_op(in_channels=in_channels, out_channels=32,
                                  kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))

        self.conv1b = nn.Conv3d(in_channels=32, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))

        self.hourglass = []
        for i in range(stacked_num):
            self.hourglass.append(nn.ModuleList([RODEncode_HG(), RODDecode_HG(),
                                                 nn.Conv3d(in_channels=64, out_channels=n_class,
                                                           kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                                           padding=(4, 2, 2)),
                                                 nn.Conv3d(in_channels=n_class, out_channels=64,
                                                           kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                                           padding=(4, 2, 2))]))
        self.hourglass = nn.ModuleList(self.hourglass)
        self.relu = nn.ReLU()
        self.bn1a = nn.BatchNorm3d(num_features=32)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))

        out = []
        for i in range(self.stacked_num):
            x, x1, x2, x3 = self.hourglass[i][0](x)
            x = self.hourglass[i][1](x, x1, x2, x3)
            confmap = self.hourglass[i][2](x)
            if not self.use_mse_loss:
                confmap = self.sigmoid(confmap)
            out.append(confmap)
            if i < self.stacked_num - 1:
                confmap_ = self.hourglass[i][3](confmap)
                x = x + confmap_
        return out


class RODEncode_HG(nn.Module):

    def __init__(self):
        super(RODEncode_HG, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))

        self.skipconv1a = nn.Conv3d(in_channels=64, out_channels=64,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.skipconv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                    kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.skipconv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.skipconv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                    kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.skipconv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.skipconv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                    kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)

        self.skipbn1a = nn.BatchNorm3d(num_features=64)
        self.skipbn1b = nn.BatchNorm3d(num_features=64)
        self.skipbn2a = nn.BatchNorm3d(num_features=128)
        self.skipbn2b = nn.BatchNorm3d(num_features=128)
        self.skipbn3a = nn.BatchNorm3d(num_features=256)
        self.skipbn3b = nn.BatchNorm3d(num_features=256)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.skipbn1a(self.skipconv1a(x)))
        x1 = self.relu(self.skipbn1b(self.skipconv1b(x1)))
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)

        x2 = self.relu(self.skipbn2a(self.skipconv2a(x)))
        x2 = self.relu(self.skipbn2b(self.skipconv2b(x2)))
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)

        x3 = self.relu(self.skipbn3a(self.skipconv3a(x)))
        x3 = self.relu(self.skipbn3b(self.skipconv3b(x3)))
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)

        return x, x1, x2, x3


class RODDecode_HG(nn.Module):

    def __init__(self):
        super(RODDecode_HG, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        # self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
        #                                   radar_configs['ramap_asize']), mode='nearest')

    def forward(self, x, x1, x2, x3):
        x = self.prelu(self.convt1(x + x3))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.convt2(x + x2))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.convt3(x + x1)  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        return x
    

class RadarStackedHourglass_HGwI(nn.Module):

    def __init__(self, in_channels, n_class, stacked_num=1, conv_op=None, use_mse_loss=False):
        super(RadarStackedHourglass_HGwI, self).__init__()
        self.stacked_num = stacked_num
        if conv_op is None:
            self.conv1a = nn.Conv3d(in_channels=in_channels, out_channels=32,
                                    kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        else:
            self.conv1a = conv_op(in_channels=in_channels, out_channels=32,
                                  kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))

        self.conv1b = nn.Conv3d(in_channels=32, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1c = nn.Conv3d(in_channels=64, out_channels=160,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))

        self.hourglass = []
        for i in range(stacked_num):
            self.hourglass.append(nn.ModuleList([RODEncode_HGwI(), RODDecode_HGwI(),
                                                 nn.Conv3d(in_channels=160, out_channels=n_class,
                                                           kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                                           padding=(4, 2, 2)),
                                                 nn.Conv3d(in_channels=n_class, out_channels=160,
                                                           kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                                           padding=(4, 2, 2))]))
        self.hourglass = nn.ModuleList(self.hourglass)
        self.relu = nn.ReLU()
        self.bn1a = nn.BatchNorm3d(num_features=32)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn1c = nn.BatchNorm3d(num_features=160)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.relu(self.bn1c(self.conv1c(x)))

        out = []
        for i in range(self.stacked_num):
            x, x1, x2, x3 = self.hourglass[i][0](x)
            x = self.hourglass[i][1](x, x1, x2, x3)
            confmap = self.hourglass[i][2](x)
            if not self.use_mse_loss:
                confmap = self.sigmoid(confmap)
            out.append(confmap)
            if i < self.stacked_num - 1:
                confmap_ = self.hourglass[i][3](confmap)
                x = x + confmap_
        return out


class InceptionLayerConcat(nn.Module):
    """
    Kernal size: for 2d kernal size, since the kernal size in temporal domain will be fixed
    """

    def __init__(self, kernal_size, in_channel, stride):
        super(InceptionLayerConcat, self).__init__()

        paddingX = kernal_size[0] // 2
        paddingY = kernal_size[1] // 2

        self.branch1 = nn.Conv3d(in_channels=in_channel, out_channels=32,
                                 kernel_size=(5, kernal_size[0], kernal_size[1]), stride=stride,
                                 padding=(2, paddingX, paddingY))
        self.branch2a = nn.Conv3d(in_channels=in_channel, out_channels=64,
                                  kernel_size=(5, kernal_size[0], kernal_size[1]), stride=(1, 1, 1),
                                  padding=(2, paddingX, paddingY))
        self.branch2b = nn.Conv3d(in_channels=64, out_channels=64,
                                  kernel_size=(9, kernal_size[0], kernal_size[1]), stride=stride,
                                  padding=(4, paddingX, paddingY))
        self.branch3a = nn.Conv3d(in_channels=in_channel, out_channels=64,
                                  kernel_size=(5, kernal_size[0], kernal_size[1]), stride=(1, 1, 1),
                                  padding=(2, paddingX, paddingY))
        self.branch3b = nn.Conv3d(in_channels=64, out_channels=64,
                                  kernel_size=(13, kernal_size[0], kernal_size[1]), stride=stride,
                                  padding=(6, paddingX, paddingY))

    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)

        branch3 = self.branch3a(x)
        branch3 = self.branch3b(branch3)

        return torch.cat((branch1, branch2, branch3), 1)


class RODEncode_HGwI(nn.Module):

    def __init__(self):
        super(RODEncode_HGwI, self).__init__()
        self.inception1 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.inception2 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.inception3 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))

        self.skip_inception1 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.skip_inception2 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.skip_inception3 = InceptionLayerConcat(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        # self.conv4a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv4b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        # self.conv5a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv5b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1 = nn.BatchNorm3d(num_features=160)
        self.bn2 = nn.BatchNorm3d(num_features=160)
        self.bn3 = nn.BatchNorm3d(num_features=160)

        self.skip_bn1 = nn.BatchNorm3d(num_features=160)
        self.skip_bn2 = nn.BatchNorm3d(num_features=160)
        self.skip_bn3 = nn.BatchNorm3d(num_features=160)
        # self.bn4a = nn.BatchNorm3d(num_features=64)
        # self.bn4b = nn.BatchNorm3d(num_features=64)
        # self.bn5a = nn.BatchNorm3d(num_features=64)
        # self.bn5b = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.skip_bn1(self.skip_inception1(x)))
        x = self.relu(self.bn1(self.inception1(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)

        x2 = self.relu(self.skip_bn2(self.skip_inception2(x)))
        x = self.relu(self.bn2(self.inception2(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)

        x3 = self.relu(self.skip_bn3(self.skip_inception3(x)))
        x = self.relu(self.bn3(self.inception3(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)

        return x, x1, x2, x3


class RODDecode_HGwI(nn.Module):

    def __init__(self):
        super(RODDecode_HGwI, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=160, out_channels=160,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=160, out_channels=160,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=160, out_channels=160,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.conv1 = nn.Conv3d(in_channels=160, out_channels=160,
                               kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=160, out_channels=160,
                               kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3 = nn.Conv3d(in_channels=160, out_channels=160,
                               kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        # self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
        #                                   radar_configs['ramap_asize']), mode='nearest')

    def forward(self, x, x1, x2, x3):
        x = self.prelu(self.convt1(x + x3))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.convt2(x + x2))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.prelu(self.conv2(x))
        x = self.prelu(self.convt3(x + x1))  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        x = self.prelu(self.conv3(x))
        return x



class RODNet(nn.Module):
    def __init__(self, nettype, in_channels, n_class, stacked_num=1, mnet_cfg=None, dcn=True):
        super(RODNet, self).__init__()
        self.nettype = nettype

        if nettype == 'RODNetCDC':
            self.cdc = RadarVanilla(in_channels, n_class, use_mse_loss=False)
        elif nettype == 'RODNetHG':
            self.stacked_hourglass = RadarStackedHourglass_HG(in_channels, n_class, stacked_num=stacked_num) 
        elif nettype == 'RODNetHGwI':
            self.stacked_hourglass = RadarStackedHourglass_HGwI(in_channels, n_class, stacked_num=stacked_num)
        elif nettype == 'RODNetCDCDCN':
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
        elif nettype == 'RODNetHGDCN':
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
        elif nettype == 'RODNetHGwIDCN':
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
        if self.nettype == 'RODNetCDC':
            out = self.cdc(x)
        elif self.nettype == 'RODNetHG':
            out = self.stacked_hourglass(x)
        elif self.nettype == 'RODNetHGwI':
            out = self.stacked_hourglass(x)
        elif self.nettype == 'RODNetCDCDCN':
            if self.with_mnet:
                x = self.mnet(x)
            out = self.cdc(x)
        elif self.nettype == 'RODNetHGDCN':
            if self.with_mnet:
                x = self.mnet(x)
            out = self.stacked_hourglass(x)
        else:  # nettype: RODNetHGwIDCN
            if self.with_mnet: 
                x = self.mnet(x)
            out = self.stacked_hourglass(x)
        return out