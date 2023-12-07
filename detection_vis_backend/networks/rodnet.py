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


class InceptionLayerConcat_HGwI(nn.Module):
    """
    Kernal size: for 2d kernal size, since the kernal size in temporal domain will be fixed
    """

    def __init__(self, kernal_size, in_channel, stride):
        super(InceptionLayerConcat_HGwI, self).__init__()

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
        self.inception1 = InceptionLayerConcat_HGwI(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.inception2 = InceptionLayerConcat_HGwI(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.inception3 = InceptionLayerConcat_HGwI(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))

        self.skip_inception1 = InceptionLayerConcat_HGwI(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.skip_inception2 = InceptionLayerConcat_HGwI(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
        self.skip_inception3 = InceptionLayerConcat_HGwI(kernal_size=(5, 5), in_channel=160, stride=(1, 2, 2))
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



class RadarStackedHourglass_HGwI2d(nn.Module):
    def __init__(self, in_channels, win_size, n_class, stacked_num=1, conv_op=None, use_mse_loss=False):
        super(RadarStackedHourglass_HGwI2d, self).__init__()
        self.stacked_num = stacked_num
        self.win_size = win_size
        self.k_3d = 1+win_size//7 
        if conv_op is None:
            self.conv1a = nn.Conv3d(in_channels=in_channels, out_channels=16,
                                    kernel_size=(self.k_3d, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2))
        else:
            self.conv1a = conv_op(in_channels=in_channels, out_channels=16,
                                  kernel_size=(self.k_3d, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))

        self.conv1b = nn.Conv3d(in_channels=16, out_channels=32,
                                kernel_size=(self.k_3d, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2))
        self.conv1c = nn.Conv3d(in_channels=32, out_channels=80,
                                kernel_size=(self.k_3d, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))

        self.hourglass = []
        for i in range(stacked_num):
            self.hourglass.append(nn.ModuleList([RODEncode_HGwI2d(), RODDecode_HGwI2d(win_size=self.win_size),
                                                 nn.Conv3d(in_channels=32, out_channels=n_class,
                                                           kernel_size=(5, 3, 3), stride=(1, 1, 1),
                                                           padding=(2, 1, 1)),
                                                 nn.Conv3d(in_channels=n_class, out_channels=80,
                                                           kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                                           padding=(4, 2, 2))]))        

        self.hourglass = nn.ModuleList(self.hourglass)
        self.relu = nn.ReLU()
        self.bn1a = nn.BatchNorm3d(num_features=16)
        self.bn1b = nn.BatchNorm3d(num_features=32)
        self.bn1c = nn.BatchNorm3d(num_features=80)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss

    def forward(self, x):
        x_in1 = self.relu(self.bn1a(self.conv1a(x)))
        x_in2 = self.relu(self.bn1b(self.conv1b(x_in1)))
        x_in3 = self.relu(self.bn1c(self.conv1c(x_in2)))

        out = []
        for i in range(self.stacked_num):
            x, x1, x2, x3 = self.hourglass[i][0](x_in3)
            x = self.hourglass[i][1](x, x1, x2, x3, x_in1, x_in2, x_in3)
            confmap = self.hourglass[i][2](x)
            if not self.use_mse_loss:
                confmap = self.sigmoid(confmap)
            out.append(confmap)
            if i < self.stacked_num - 1:
                confmap_ = self.hourglass[i][3](confmap)
                x = x + confmap_
        return confmap


class InceptionLayerConcat_HGwI2d(nn.Module):
    """
    Kernal size: for 2d kernal size, since the kernal size in temporal domain will be fixed
    """

    def __init__(self, kernal_size, in_channel, stride):
        super(InceptionLayerConcat_HGwI2d, self).__init__()

        paddingX = kernal_size[0] // 2
        paddingY = kernal_size[1] // 2
        self.branch1 = nn.Conv2d(in_channels=in_channel, out_channels=16,
                                 kernel_size=(kernal_size[0],kernal_size[1]),stride=stride,
                                 padding=(paddingX,paddingY))
        self.branch2a = nn.Conv2d(in_channels=in_channel, out_channels=32,
                                 kernel_size=(kernal_size[0],kernal_size[1]),stride=(1,1),
                                 padding=(paddingX,paddingY))
        self.branch2b = nn.Conv2d(in_channels=32, out_channels=32,
                                 kernel_size=(kernal_size[0],kernal_size[1]),stride=stride,
                                 padding=(paddingX,paddingY))
        self.branch3a = nn.Conv2d(in_channels=in_channel, out_channels=32,
                                 kernel_size=(kernal_size[0],kernal_size[1]),stride=(1,1),
                                 padding=(paddingX,paddingY))
        self.branch3b = nn.Conv2d(in_channels=32, out_channels=32,
                                 kernel_size=(kernal_size[0],kernal_size[1]),stride=stride,
                                 padding=(paddingX,paddingY))

    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)

        branch3 = self.branch3a(x)
        branch3 = self.branch3b(branch3)

        return torch.cat((branch1, branch2, branch3), 1)


class RODEncode_HGwI2d(nn.Module):
    def __init__(self):
        super(RODEncode_HGwI2d, self).__init__()
        self.inception1 = InceptionLayerConcat_HGwI2d(kernal_size=(5, 5), in_channel=80, stride=(2, 2))
        self.inception2 = InceptionLayerConcat_HGwI2d(kernal_size=(5, 5), in_channel=80, stride=(2, 2))
        self.inception3 = InceptionLayerConcat_HGwI2d(kernal_size=(5, 5), in_channel=80, stride=(2, 2))

        self.skip_inception1 = InceptionLayerConcat_HGwI2d(kernal_size=(5, 5), in_channel=80, stride=(2, 2))
        self.skip_inception2 = InceptionLayerConcat_HGwI2d(kernal_size=(5, 5), in_channel=80, stride=(2, 2))
        self.skip_inception3 = InceptionLayerConcat_HGwI2d(kernal_size=(5, 5), in_channel=80, stride=(2, 2))

        self.bn1 = nn.BatchNorm2d(num_features=80)
        self.bn2 = nn.BatchNorm2d(num_features=80)
        self.bn3 = nn.BatchNorm2d(num_features=80)

        self.skip_bn1 = nn.BatchNorm2d(num_features=80)
        self.skip_bn2 = nn.BatchNorm2d(num_features=80)
        self.skip_bn3 = nn.BatchNorm2d(num_features=80)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x[:,:,0,:,:]
        x1 = self.relu(self.skip_bn1(self.skip_inception1(x)))
        x = self.relu(self.bn1(self.inception1(x)))  # (B, 2, W, 128, 128) -> (B, 32, W, 128, 128)

        x2 = self.relu(self.skip_bn2(self.skip_inception2(x)))
        x = self.relu(self.bn2(self.inception2(x)))  # (B, 2, W, 128, 128) -> (B, 32, W, 128, 128)

        x3 = self.relu(self.skip_bn3(self.skip_inception3(x)))
        x = self.relu(self.bn3(self.inception3(x)))  # (B, 2, W, 128, 128) -> (B, 32, W, 128, 128)

        return x, x1, x2, x3


class RODDecode_HGwI2d(nn.Module):
    def __init__(self,win_size):
        super(RODDecode_HGwI2d, self).__init__()
        k_3d = 1+win_size//7
        self.t2d1 = nn.ConvTranspose2d(in_channels=80, out_channels=80,
                                        kernel_size=(6,6), stride=(2,2), padding=2)
        self.t2d2 = nn.ConvTranspose2d(in_channels=80, out_channels=80,
                                        kernel_size=(6,6), stride=(2,2), padding=2)
        self.t2d3 = nn.ConvTranspose2d(in_channels=80, out_channels=80,
                                        kernel_size=(6,6), stride=(2,2), padding=2)
        self.c2d1 = nn.Conv2d(in_channels=80, out_channels=80,
                               kernel_size=(3,3), stride=( 1, 1), padding=(1,1))
        self.c2d2 = nn.Conv2d(in_channels=80, out_channels=80,
                               kernel_size=(3,3), stride=( 1, 1), padding=(1,1))
        self.c2d3 = nn.Conv2d(in_channels=80, out_channels=80,
                               kernel_size=(3,3), stride=(1, 1), padding=(1,1))

        self.convt1 = nn.ConvTranspose3d(in_channels=80, out_channels=80,
                                         kernel_size=(k_3d, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.convt2 = nn.ConvTranspose3d(in_channels=32, out_channels=32,
                                         kernel_size=((k_3d if (win_size) == 16 else (k_3d+1)) , 1, 1), stride=(2, 1, 1), padding=(0, 0,0))
        self.convt3 = nn.ConvTranspose3d(in_channels=16, out_channels=16,
                                         kernel_size=(k_3d+1, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        #self.convt4 = nn.ConvTranspose3d(in_channels=80, out_channels=80,
        #                                 kernel_size=(9, 1, 1))
        self.conv1 = nn.Conv3d(in_channels=80, out_channels=32,
                               kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=16,
                               kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0))
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32,
                               kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        # self.upsample = nn.Upsample(size=(rodnet_configs['win_size'], radar_configs['ramap_rsize'],
        #                                   radar_configs['ramap_asize']), mode='nearest')

    def forward(self, x, x1, x2, x3, x_in1, x_in2, x_in3):
        x = self.prelu(self.t2d1(x+x3))
        x = self.prelu(self.c2d1(x))
        x = self.prelu(self.t2d2(x+x2))
        x = self.prelu(self.c2d2(x))
        x = self.prelu(self.t2d3(x+x1))
        x = self.prelu(self.c2d3(x))
        x = torch.unsqueeze(x, dim=2)
        x = self.prelu(self.convt1(x + x_in3)) 
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.convt2(x + x_in2)) 
        x = self.prelu(self.conv2(x))
        x = self.prelu(self.convt3(x + x_in1))
        x = self.prelu(self.conv3(x))
        return x
    

class SmoothCELoss(nn.Module):
    """
    Smooth cross entropy loss
    SCE = SmoothL1Loss() + BCELoss()
    By default reduction is mean. 
    """
    def __init__(self, alpha):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.bce = nn.BCELoss()
        self.alpha = alpha
    
    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + (1-self.alpha) * self.smooth_l1(input, target)