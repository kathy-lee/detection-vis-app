"""TransRAD: Retentive Vision Transformer for radar object detection on RAD cubes.

Port of https://github.com/radar-lab/TransRAD (commit 8f9b9ad, GPL-3.0; Cheng & Cao,
IEEE TRS 2025, arXiv 2501.17977), itself built on qhfan/RMT and bubbliiiing/yolov8-pytorch.
A pristine checkout lives at /home/kangle/dataset/transrad_reference/TransRAD and every
class below names the file and lines it was copied from. Deviations from the reference
are commented where they happen; there are five: SyncBatchNorm -> BatchNorm2d (same
keys, same single-GPU behaviour), feature strides fixed from the architecture instead of
a dummy forward at construction, per-step prints removed, the two timm helpers vendored,
and the unused image-dataset code left out.

Layout: the RMT backbone, the YOLOv8 neck + decoupled heads (TransRADBody), the loss
(task-aligned assignment, DFL, CIoU, focal), the decoder and the two-stage NMS
(class NMS then Location-Aware NMS).
"""
import math
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from detection_vis_backend.train.utils import iou3d_np

REF_CHECKPOINT = "/home/kangle/dataset/transrad_reference/TransRAD/model_data/best_epoch_weights.pth"


# ---------------------------------------------------------------------------------------
# Vendored from timm (rwightman/pytorch-image-models, Apache-2.0): timm/layers/drop.py
# drop_path/DropPath and timm/layers/weight_init.py trunc_normal_. The reference imports
# them from timm; copying ~40 lines avoids a dependency on the whole package.
# ---------------------------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ---------------------------------------------------------------------------------------
# RMT backbone: REF/RMT_Det.py lines 34-610 (qhfan/RMT, as adapted by TransRAD).
# Copied verbatim except: SyncBatchNorm -> BatchNorm2d (see PatchMerging), the unused
# MemoryEfficientSwish, LayerNorm2d, RMT.init_weights and the two no_weight_decay
# helpers left out.
# ---------------------------------------------------------------------------------------
def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x


class RetNetRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None] #(n H*W H*W)
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] #(l l)
        mask = mask.abs() #(l l)
        mask = mask * self.decay[:, None, None] #(n l l)
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0]*slen[1] - 1))
            cos = torch.cos(self.angle * (slen[0]*slen[1] - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())

        elif chunkwise_recurrent:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)

            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(slen[0]*slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
            sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
            cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
            cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)
            mask = self.generate_2d_decay(slen[0], slen[1]) #(n l l)
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

class VisionRetentionChunk(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)


        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        (sin, cos), (mask_h, mask_w) = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''

        qr_w = qr.transpose(1, 2) #(b h n w d1)
        kr_w = kr.transpose(1, 2) #(b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)
        qk_mat_w = qk_mat_w + mask_w  #(b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1) #(b h n w w)
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = qr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)
        qk_mat_h = qk_mat_h + mask_h  #(b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output

class VisionRetentionAll(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos

        assert h*w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        qr = theta_shift(q, sin, cos) #(b n h w d1)
        kr = theta_shift(k, sin, cos) #(b n h w d1)

        qr = qr.flatten(2, 3) #(b n l d1)
        kr = kr.flatten(2, 3) #(b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d2)
        vr = vr.flatten(2, 3) #(b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2) #(b n l l)
        qk_mat = qk_mat + mask  #(b n l l)
        qk_mat = torch.softmax(qk_mat, -1) #(b n l l)
        output = torch.matmul(qk_mat, vr) #(b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=True
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        residual = x
        if self.dwconv is not None:
            x = self.dwconv(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = x + residual
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

class RetBlock(nn.Module):

    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = VisionRetentionChunk(embed_dim, num_heads)
        else:
            self.retention = VisionRetentionAll(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)

    def forward(
            self,
            x: torch.Tensor,
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x

# Deviation: the reference uses nn.SyncBatchNorm in PatchMerging and PatchEmbed. On one
# GPU with no process group it computes exactly BatchNorm2d, and both store the same
# state-dict keys, so the authors' checkpoint loads unchanged; BatchNorm2d also lets
# norm_eval (RMT.train) freeze these layers, which its isinstance check would miss for
# SyncBatchNorm.
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        '''
        x: B H W C
        '''
        x = x.permute(0, 3, 1, 2).contiguous()  #(b c h w)
        x = self.reduction(x) #(b oc oh ow)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) #(b oh ow oc)
        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, embed_dim, out_dim, depth, num_heads,
                 init_value: float, heads_range: float,
                 ffn_dim=96., drop_path=0., norm_layer=nn.LayerNorm, chunkwise_recurrent=False,
                 downsample: PatchMerging=None, use_checkpoint=False,
                 layerscale=False, layer_init_values=1e-5):

        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.chunkwise_recurrent = chunkwise_recurrent
        if chunkwise_recurrent:
            flag = 'chunk'
        else:
            flag = 'whole'
        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)

        # build blocks
        self.blocks = nn.ModuleList([
            RetBlock(flag, embed_dim, num_heads, ffn_dim,
                     drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        b, h, w, d = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x=x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
            else:
                x = blk(x, incremental_state=None, chunkwise_recurrent=self.chunkwise_recurrent, retention_rel_pos=rel_pos)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, 3, 1, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1) #(b h w c)
        return x

class RMT(nn.Module):

    def __init__(self, in_chans=3, out_indices=(0, 1, 2, 3),
                 embed_dims=[64, 128, 256, 512], depths=[2, 2, 8, 2], num_heads=[4, 4, 8, 16],
                 init_values=[2, 2, 2, 2], heads_ranges=[4, 4, 6, 6], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False, chunkwise_recurrents=[True, True, False, False], projection=1024,
                 layerscales=[False, False, False, False], layer_init_values=1e-6, norm_eval=True):
        super().__init__()
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.norm_eval = norm_eval

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
            norm_layer=norm_layer if self.patch_norm else None)


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                chunkwise_recurrent=chunkwise_recurrents[i_layer],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values
            )
            self.layers.append(layer)

        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(nn.LayerNorm(embed_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward(self, x):
        x = self.patch_embed(x) #[b,256,256,256]-->[b,64,64,32] #last_dim is embed_dim

        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x) #x_out->[b,64,64,32]; x->[b,32,32,64]
            if i in self.out_indices:
                x_out = self.extra_norms[i](x_out)#x_out->[b,64,64,32]
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out) #[b,32,64,64],[b,64,32,32],[b,128,16,16],[b,256,8,8]

        #return tuple(outs)
        return outs[1],outs[2],outs[3]


    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


# ---------------------------------------------------------------------------------------
# YOLOv8 neck blocks: REF/nets/backbone.py lines 197-248 (bubbliiiing/yolov8-pytorch).
# ---------------------------------------------------------------------------------------
def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class Conv(nn.Module):
    default_act = SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------------------------------------
# Anchors and box decoding: REF/utils/utils_bbox.py lines 16-28, 57-67. The torch<1.10
# meshgrid fallback is dropped (torch 2.1 here).
# ---------------------------------------------------------------------------------------
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w  = feats[i].shape
        sx          = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy          = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx      = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""

    lt, rb  = torch.split(distance, distance.shape[dim]//2, dim)
    x1y1    = anchor_points - lt
    x2y2    = anchor_points + rb
    if xywh:
        c_xy    = (x1y1 + x2y2) / 2
        wh      = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


# ---------------------------------------------------------------------------------------
# Weight init: REF/nets/yolo_training.py lines 704-722 (print removed).
# ---------------------------------------------------------------------------------------
def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


# ---------------------------------------------------------------------------------------
# DFL and the body: REF/nets/yolo.py lines 35-52 (DFL) and 57-185 (YoloBody).
# ---------------------------------------------------------------------------------------
class DFL(nn.Module):

    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16, box_attri=4):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1  #c1 = self.reg_max
        self.box_attri = box_attri

    def forward(self, x):

        b, c, a = x.shape

        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        return self.conv(x.view(b, self.box_attri, self.c1, a).transpose(2, 1).softmax(1)).view(b, self.box_attri, a)


class TransRADBody(nn.Module):
    """REF/nets/yolo.py YoloBody (lines 57-185): RMT backbone, two top-down C2f fusions,
    four decoupled heads per level (box distribution, class, objectness, Doppler end
    points). Attribute names are the reference's so model_data/best_epoch_weights.pth
    loads with strict=True."""

    def __init__(self, input_size, n_class, phi="n", embed_dims=(32, 64, 128, 256),
                 depths=(2, 2, 8, 2), num_heads=(4, 4, 8, 16), drop_path_rate=0.1, reg_max=16):
        super().__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00}
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        input_channels = input_size[0]          # Doppler bins after the adapter's resize (256)
        base_channels = int(wid_mul * 64)       # 16 for 'n'
        base_depth = max(round(dep_mul * 3), 1)  # 1 for 'n'

        self.backbone = RMT(in_chans=input_channels, out_indices=(0, 1, 2, 3),
                            embed_dims=list(embed_dims), depths=list(depths), num_heads=list(num_heads),
                            init_values=[2, 2, 2, 2], heads_ranges=[4, 4, 6, 6], mlp_ratios=[3, 3, 3, 3],
                            drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm,
                            patch_norm=True, use_checkpoint=False,
                            chunkwise_recurrents=[True, True, False, False], projection=1024,
                            layerscales=[False, False, False, False], layer_init_values=1e-6, norm_eval=True)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        # The bottom-up path is built but never used in forward (reference lines 84-91,
        # 134-156); kept so the checkpoint's down_sample*/conv3_for_downsample* keys load.
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)
        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)

        ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape = None
        self.nl = len(ch)
        # Deviation: the reference runs a dummy forward here to measure the strides
        # (line 97). The stem is stride 4 and each of the three merges stride 2, and the
        # body reads the last three stages, so the strides are 8, 16, 32 by construction.
        self.stride = torch.tensor([8., 16., 32.])
        self.obj_ness = 1
        self.reg_max = reg_max
        self.box_attri = 4
        self.no = n_class + self.reg_max * self.box_attri + self.obj_ness + 2
        self.num_classes = n_class
        self.input_shape = list(input_size)

        c2, c3 = max((16, ch[0] // self.box_attri, self.reg_max * self.box_attri)), max(ch[0], n_class)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.box_attri * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, n_class, 1), nn.Sigmoid()) for x in ch)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.obj_ness, 1), nn.Sigmoid()) for x in ch)
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, 2, 1), nn.Sigmoid()) for x in ch)
        # Train.py runs with pretrained=False, so the whole body -- backbone convs
        # included, after RMT's own init -- gets the normal(0, 0.02) init (lines 120-121).
        weights_init(self)
        self.dfl = DFL(self.reg_max, self.box_attri) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        P5 = feat3
        P5_upsample = self.upsample(feat3)
        P4 = torch.cat([P5_upsample, feat2], 1)
        P4 = self.conv3_for_upsample1(P4)
        P4_upsample = self.upsample(P4)
        P3 = torch.cat([P4_upsample, feat1], 1)
        P3 = self.conv3_for_upsample2(P3)

        shape = P3.shape
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cv4[i](x[i]), self.cv5[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls, obj, points = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * self.box_attri, self.num_classes, self.obj_ness, 2), 1)
        dbox = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)
