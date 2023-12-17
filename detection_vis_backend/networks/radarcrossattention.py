import torch
import torch.nn as nn
import torch.nn.functional as F

class Encode(nn.Module):
    def __init__(self,in_channels):
        super(Encode,self).__init__()

        self.conv1a = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv1b = nn.Conv2d(in_channels =64, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(2,2))
        
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv2b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5,5), stride=(2,2), padding=(2,2))

        self.conv3a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv3b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=(2,2))

        self.bn1a = nn.BatchNorm2d(num_features=64)
        self.bn1b = nn.BatchNorm2d(num_features=64)
        self.bn2a = nn.BatchNorm2d(num_features=128)
        self.bn2b = nn.BatchNorm2d(num_features=128)
        self.bn3a = nn.BatchNorm2d(num_features=256)
        self.bn3b = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self,x):
        # print(x.dtype)
        # for name, param in self.named_parameters():
        #     print(name, param.dtype)
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 1, 256, 256) -> (B, 64,  256, 256)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, 256, 256) -> (B, 64, 128, 128)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, 128, 128) -> (B, 128, 128, 128)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, 128, 128) -> (B, 128, 64, 64)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, 64, 64) -> (B, 256, 64, 64 )
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, 64, 64) -> (B, 256, 32, 32)
        return x 
    
class Decode(nn.Module):
    def __init__(self, out_channel=32):
        super(Decode,self).__init__()

        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, stride=(2,2), padding=(2,2), kernel_size=(6,6))
        self.convt2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=(2,2), padding=(2,2), kernel_size=(6,6))
        self.convt3 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channel, stride=(2,2), padding=(2,2), kernel_size=(6,6))

        self.bn_cn = nn.BatchNorm2d(num_features=2)
        self.PRelu = nn.PReLU()

    def forward(self,x):
        x = self.PRelu(self.convt1(x)) # (B,256,32,32) -> (B,128,64,64)
        x = self.PRelu(self.convt2(x)) # (B,128,64,64) -> (B,64,128,128)   
        x = self.PRelu(self.convt3(x)) # (B,64,128,128) -> (B,32,256,256)
        return x
    

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.softmax = nn.Softmax(dim=2)
        self.norm = nn.LayerNorm([256,32,32])
    
    def forward(self, ra_e, rd_e, ad_e):
        x = torch.matmul(rd_e, ad_e) # B,256,32,32
        x = self.softmax(x.view(*x.size()[:2],-1)).view_as(x)
        x = ra_e*x
        x = self.norm(ra_e + x)
        return x
    

class FocalLoss_weight(nn.Module):
    def __init__(self,weights,alpha,beta):
        super(FocalLoss_weight,self).__init__()
        self.register_buffer('weights', weights)
        self.alpha = alpha
        self.beta = beta

    def forward(self,input,target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.weights * (1-pt)**self.alpha * BCE_loss
        return F_loss.sum()

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss,self).__init__()
    
    def forward(self,pred_map,target_map):
        target_map= target_map.flatten()
        idx = torch.nonzero(target_map)
        target_map = ((target_map/8)+0.5)
        pred_map = pred_map.flatten()
        c_loss = F.binary_cross_entropy_with_logits(pred_map[idx], target_map[idx], reduction='none')
        return c_loss.sum()

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss,self).__init__()
   
    def forward(self,pred_o,target_o):
        ep = 1e-10
        target_s= target_o[:,0,::].flatten() # sine 
        target_c= target_o[:,1,::].flatten() # cosine

        idx = torch.nonzero(target_c)

        
        pred_s= pred_o[:,0,::].flatten()
        pred_c= pred_o[:,1,::].flatten()

        loss = (F.mse_loss(pred_s[idx],target_s[idx],reduction='sum')
                + F.mse_loss(pred_c[idx],target_c[idx],reduction='sum'))
        return loss.sum()