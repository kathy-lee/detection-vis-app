import torch
import torch.nn as nn
import torch.nn.functional as F



class RODEncode_RA(nn.Module):
    def __init__(self):
        super(RODEncode_RA, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=2, out_channels=64,
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
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        # self.conv4a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv4b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        # self.conv5a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv5b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        # self.bn4a = nn.BatchNorm3d(num_features=64)
        # self.bn4b = nn.BatchNorm3d(num_features=64)
        # self.bn5a = nn.BatchNorm3d(num_features=64)
        # self.bn5b = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128) Note: W~2W in this case
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/8, 16, 16)
        # x = self.relu(self.bn4a(self.conv4a(x)))
        # x = self.relu(self.bn4b(self.conv4b(x)))
        # x = self.relu(self.bn5a(self.conv5a(x)))
        # x = self.relu(self.bn5b(self.conv5b(x)))
        return x


class RODDecode_RA(nn.Module):
    def __init__(self, win_size, ramap_rsize, ramap_asize):
        super(RODDecode_RA, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=32,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        # self.convt4 = nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
        #                                  kernel_size=(3, 6, 6), stride=(1, 4, 4), padding=(1, 1, 1))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(size=(win_size, ramap_rsize, ramap_asize), mode='nearest')

    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, 256, W/8, 16, 16) -> (B, 128, W/4, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/4, 32, 32) -> (B, 64, W/2, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, 64, W/2, 64, 64) -> (B, 32, W/2, 128, 128)
        # x = self.convt4(x)
        # x = self.upsample(x)
        # x = self.sigmoid(x)
        return x


class RODEncode_RV(nn.Module):
    def __init__(self):
        super(RODEncode_RV, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=1, out_channels=64,
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
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        # self.conv4a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv4b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        # self.conv5a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv5b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        # self.bn4a = nn.BatchNorm3d(num_features=64)
        # self.bn4b = nn.BatchNorm3d(num_features=64)
        # self.bn5a = nn.BatchNorm3d(num_features=64)
        # self.bn5b = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/8, 16, 16)
        # x = self.relu(self.bn4a(self.conv4a(x)))
        # x = self.relu(self.bn4b(self.conv4b(x)))
        # x = self.relu(self.bn5a(self.conv5a(x)))
        # x = self.relu(self.bn5b(self.conv5b(x)))
        return x


class RODDecode_RV(nn.Module):
    def __init__(self, win_size, ramap_rsize, ramap_asize):
        super(RODDecode_RV, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=32,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        # self.convt4 = nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
        #                                  kernel_size=(3, 6, 6), stride=(1, 4, 4), padding=(1, 1, 1))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(size=(win_size, ramap_rsize, ramap_asize), mode='nearest')

    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, 256, W/8, 16, 16) -> (B, 128, W/4, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/4, 32, 32) -> (B, 64, W/2, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, 64, W/2, 64, 64) -> (B, 32, W/2, 128, 128)
        # x = self.convt4(x)
        # x = self.upsample(x)
        # x = self.sigmoid(x)
        return x


class RODEncode_VA(nn.Module):
    def __init__(self):
        super(RODEncode_VA, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=1, out_channels=64,
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
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        # self.conv4a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv4b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        # self.conv5a = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        # self.conv5b = nn.Conv3d(in_channels=64, out_channels=64,
        #                         kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        # self.bn4a = nn.BatchNorm3d(num_features=64)
        # self.bn4b = nn.BatchNorm3d(num_features=64)
        # self.bn5a = nn.BatchNorm3d(num_features=64)
        # self.bn5b = nn.BatchNorm3d(num_features=64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/8, 16, 16)
        # x = self.relu(self.bn4a(self.conv4a(x)))
        # x = self.relu(self.bn4b(self.conv4b(x)))
        # x = self.relu(self.bn5a(self.conv5a(x)))
        # x = self.relu(self.bn5b(self.conv5b(x)))

        return x


class RODDecode_VA(nn.Module):
    def __init__(self, win_size, ramap_rsize, ramap_asize):
        super(RODDecode_VA, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=32,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        # self.convt4 = nn.ConvTranspose3d(in_channels=64, out_channels=n_class,
        #                                  kernel_size=(3, 6, 6), stride=(1, 4, 4), padding=(1, 1, 1))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(size=(win_size, ramap_rsize, ramap_asize), mode='nearest')

    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, 256, W/8, 16, 16) -> (B, 128, W/4, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/4, 32, 32) -> (B, 64, W/2, 64, 64)
        x = self.prelu(self.convt3(x))  # (B, 64, W/2, 64, 64) -> (B, 32, W/2, 128, 128)
        # x = self.convt4(x)
        # x = self.upsample(x)
        # x = self.sigmoid(x)
        return x
    

class Fuse_fea_new_rep(nn.Module):
    def __init__(self, n_class, n_range, n_angle):
        super(Fuse_fea_new_rep, self).__init__()
        self.n_range = n_range
        self.n_angle = n_angle
        self.convt1 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.convt2 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.convt3 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 1, 21), stride=(1, 1, 1), padding=(0, 0, 0),
                                dilation=(1, 1, 6)) # padding 60
        self.convt4 = nn.Conv3d(in_channels=48, out_channels=n_class,
                                kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_ra, feas_rv, feas_va):
        feas_rv = torch.sum(feas_rv, 4, keepdim=True) # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_ra1 = feas_rv.expand(-1, -1, -1, -1, self.n_angle)
        feas_va = torch.sum(feas_va, 4, keepdim=True)  # condense the the velocity dimension (B, 32, W/2, 128, 128) -> (B, 32, W/2, 128, 1)
        feas_va = torch.transpose(feas_va, 3, 4)    # (B, 32, W/2, 128, 1) -> (B, 32, W/2, 1, 128)
        feas_ra2 = feas_va.expand(-1, -1, -1, self.n_range, -1)  # (B, 32, W/2, 1, 128) -> (B, 32, W/2, 128, 128)

        fea_shap = feas_ra.shape # (B, 32, W/2, 128, 128)
        feas_ra = feas_ra.permute(0, 2, 1, 3, 4)    # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra = torch.unsqueeze(torch.reshape(feas_ra, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2) # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra1 = feas_ra1.permute(0, 2, 1, 3, 4)    # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra1 = torch.unsqueeze(torch.reshape(feas_ra1, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2) # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra2 = feas_ra2.permute(0, 2, 1, 3, 4)    # (B, 32, W/2, 128, 128) -> (B, W/2, 32, 128, 128)
        feas_ra2 = torch.unsqueeze(torch.reshape(feas_ra2, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2) # (B, W/2, 32, 128, 128) -> (B*W/2, 32, 128, 128) -> (B*W/2, 32, 1, 128, 128)
        feas_ra = torch.cat((feas_ra, feas_ra1, feas_ra2), 2) # 3*(B*W/2, 32, 1, 128, 128) -> (B*W/2, 32, 3, 128, 128)

        x1 = torch.squeeze(self.prelu(self.convt1(feas_ra)))  # (B*W/2, 32, 3, 128, 128) -> (B*W/2, 16, 1, 128, 128) -> (B*W/2, 16, 128, 128)
        x2 = torch.squeeze(self.prelu(self.convt2(feas_ra)))  # (B*W/2, 32, 3, 128, 128) -> (B*W/2, 16, 1, 128, 128) -> (B*W/2, 16, 128, 128)
        feas_ra = F.pad(feas_ra, (60, 60, 0, 0, 0, 0), "circular")
        x3 = torch.squeeze(self.prelu(self.convt3(feas_ra)))  # (B*W/2, 32, 3, 128, 128) -> (B*W/2, 16, 1, 128, 128) -> (B*W/2, 16, 128, 128)
        x1 = torch.cat((x1, x2, x3), 1) # (B*W/2, 16, 128, 128) -> (B*W/2, 48, 128, 128)

        x = torch.transpose(torch.reshape(x1, (fea_shap[0], fea_shap[2], 48, fea_shap[3], fea_shap[4])), 1, 2) # (B*W/2, 48, 128, 128) -> (B, W/2, 48, 128, 128) -> (B, 48, W/2, 128, 128)
        x = self.sigmoid(self.convt4(x))  # (B, 48, W/2, 128, 128) -> (B, 3, W/2, 128, 128)
        return x


class Cont_Loss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self, win_size):
        super(Cont_Loss, self).__init__()
        self.win_size = win_size
        self.MSE_loss = nn.MSELoss()

    def forward(self, out, target):
        loss = 0
        for i in range(out.shape[0]):
            batch_loss = 0
            for j in range(self.win_size):
                if j % 2 == 0:
                    batch_loss = batch_loss + self.MSE_loss(out[i, :, j, :, :], target[i, :, j + 1, :, :])
            loss = loss + batch_loss/(self.win_size/2)
            # loss = loss + batch_loss
        return loss


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    balance_cof = 4
    # focal_inds = pred.gt(1.4013e-45) * pred.lt(1-1.4013e-45)
    pred = torch.clamp(pred, 1.4013e-45, 1)
    fos = torch.sum(gt, 1)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    fos_inds = torch.unsqueeze(fos.gt(0).float(), 1)
    fos_inds = fos_inds.expand(-1, 3, -1, -1, -1)
    neg_inds = neg_inds + (balance_cof - 1) * fos_inds * gt.eq(0)

    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    # pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * focal_inds
    # neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * focal_inds
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * balance_cof
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class FocalLoss_Neg(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss_Neg, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)