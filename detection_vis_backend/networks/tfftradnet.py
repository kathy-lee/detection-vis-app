"""T-FFTRadNet: a Swin vision transformer under FFTRadNet's detection (and, on RADIal,
freespace) heads, optionally fed raw ADC through a learnable Fourier transform.

Port of https://github.com/jgiroux8/T_FFTRadNet (commit 68bc041; Giroux, Bouchard and
Laganiere, ICCV 2023 BRAVO workshop, arXiv 2303.16940). A pristine clone is kept at
$DATASET_ROOTDIR/tfftradnet_reference/T_FFTRadNet. Module and attribute names follow the
files of the reference repo (jgiroux8/T_FFTRadNet 68bc041), so their state dicts load into
these modules unchanged.

Fourier-Net (the reference's model/fourier_net.py, one copy per sensor): complex-valued linear
layers initialized to the DFT matrices, a range transform over the ADC samples and then a
Doppler transform over the chirps. The reference sizes them 512 x 256 for RADIal and 256 x 64
for RADDet; here the sizes are arguments. The Doppler layer either centres the zero
Doppler bin through an offset in its weights (the reference's Doppler_Fourier_Net, the paper's
default ADC rows) or is the plain DFT (NoShift_Doppler_Fourier_Net, the paper's
"no shift" row). Note that the RADIal FFT_Net as committed in the reference repo builds the no-shift
layer; the shift is the default here because it is what the paper's RADIal ADC row uses.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from cplxmodule.nn import CplxAngle, CplxLinear, CplxModulus, CplxParameter, CplxToCplx
from cplxmodule.nn.modules.casting import TensorToCplx

# The reference code imports these from timm; the repo already vendors both (TransRAD needs them).
from detection_vis_backend.networks.transrad import DropPath, trunc_normal_


def to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def dft_matrix(n, shift=False):
    """W[h, j] = exp(-2 pi i j h' / n), h' = h + n/2 (mod n) when shift: row h then holds
    frequency bin h - n/2, which is fftshift applied to the transform's output."""
    h = np.arange(n)
    if shift:
        h = (h + n // 2) % n
    return np.exp(-2j * np.pi * np.outer(h, np.arange(n)) / n).astype(np.complex64)


def _dft_layer(n, shift=False):
    layer = CplxLinear(n, n, bias=False)
    layer.weight = CplxParameter(TensorToCplx()(torch.view_as_real(torch.from_numpy(dft_matrix(n, shift)))))
    return layer


class Range_Fourier_Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.range_nn = _dft_layer(n)

    def forward(self, x):
        return self.range_nn(x)


class Doppler_Fourier_Net(nn.Module):
    def __init__(self, n, shift=True):
        super().__init__()
        self.doppler_nn = _dft_layer(n, shift)

    def forward(self, x):
        return self.doppler_nn(x)


class ComplexAct(nn.Module):
    """The reference's complex activation: with use_phase, act(|z| + m) * exp(i arg z)."""

    def __init__(self, act, use_phase=False):
        super().__init__()
        self.act = act
        self.use_phase = use_phase
        self.mod = CplxModulus()
        self.angle = CplxAngle()
        self.m = nn.Parameter(torch.tensor(0.0).float())

    def forward(self, z):
        if self.use_phase:
            return self.act(self.mod(z) + self.m) * torch.exp(1.j * self.angle(z))
        return self.act(z.real) + 1.j * self.act(z.imag)


class FourierNet(nn.Module):
    """The reference's FFT_Net. Input complex (B, C, n_samples, n_chirps); output real
    (B, 2C, n_samples, n_chirps), i.e. range x Doppler as the stored RD spectra: real
    parts then imaginary parts, instance-normalized.

    activation "nl" is the non-linear variant of the paper (the reference's commented-out
    forward): ComplexAct(LeakyReLU, use_phase=True) after each of the two transforms."""

    ACTIVATIONS = ("linear", "nl")

    def __init__(self, n_samples, n_chirps, n_channels, doppler_shift=True, activation="linear"):
        super().__init__()
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"activation {activation!r} not in {self.ACTIVATIONS}")
        self.range_net = Range_Fourier_Net(n_samples)
        self.doppler_net = Doppler_Fourier_Net(n_chirps, doppler_shift)
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.norm = nn.InstanceNorm2d(2 * n_channels)
        if activation == "nl":
            self.activation = ComplexAct(act=nn.LeakyReLU(), use_phase=True)

    def transform(self, x):
        """Both transforms (and activations), complex (B, C, n_samples, n_chirps)."""
        x = x.permute(0, 1, 3, 2)
        x = self.range_net(x)
        if hasattr(self, "activation"):
            x = self.activation(x).unsqueeze(-1)
            x = TensorToCplx()(torch.concat([x.real, x.imag], axis=-1))
        x = self.cplx_transpose(2, 3)(x)
        x = self.doppler_net(x)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return torch.complex(x.real, x.imag)

    def forward(self, x):
        x = self.transform(x)
        return self.norm(torch.concat([x.real, x.imag], axis=1))


# ---------------------------------------------------------------------------------------
# Swin Transformer encoder, the reference's model/Swin.py (identical in RadIal/ and RADDet/),
# itself from Microsoft's Swin-Transformer-Object-Detection (MIT, Ze Liu et al. 2021).
# Bodies unchanged; the 2D encoder only, kept apart from T-RODNet's 3D, U-shaped Swin.

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """x: (num_windows*B, N, C); mask: (0/-inf) (num_windows, Wh*Ww, Wh*Ww) or None"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block."""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """x: (B, H*W, C); mask_matrix: attention mask for cyclic shift."""
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer"""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """x: (B, H*W, C)."""
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage."""

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """x: (B, H*W, C)."""

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone (https://arxiv.org/pdf/2103.14030), returning the
    feature map of every stage in out_indices."""

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights. T-FFTRadNet never calls this; loading a pretrained
        checkpoint here needed mmdet helpers that the reference code does not import either."""

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained checkpoints are loaded with load_state_dict, not here')

    def forward(self, x):
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
        return self


# ---------------------------------------------------------------------------------------
# Decoders and heads, the reference's model/FFTRadNet_ViT.py. RADIal and RADDet each have their
# own Detection_Header and Swin_RangeAngle_Decoder (different sizes; RADDet adds a 6-class
# head); they are renamed here per dataset. Attribute names are the reference's.

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out


class RadialDetectionHeader(nn.Module):
    """RadIal/model/FFTRadNet_ViT.py Detection_Header: [sigmoid class, range, angle]."""

    def __init__(self, use_bn=True, reg_layer=2, input_angle_size=0):
        super().__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn

        if(self.input_angle_size==224):
            self.conv1 = conv3x3(256, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==448):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==896):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias,stride=(1,2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle paraemter !')

        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, reg_layer, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return torch.cat([cls, reg], dim=1)


class RadialSwinDecoder(nn.Module):
    """RadIal/model/FFTRadNet_ViT.py Swin_RangeAngle_Decoder."""

    def __init__(self, ):
        super().__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))

        self.conv_block4 = BasicBlock(48,128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(192,256)
        self.L4  = nn.Conv2d(384*1,224,kernel_size=1,stride=1,padding=0)
        self.L3  = nn.Conv2d(192*1, 224, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(96*1, 224, kernel_size=1, stride=1,padding=0)

    def forward(self,features):

        T4 = self.L4(features[3]).transpose(1, 3)
        T3 = self.L3(features[2]).transpose(1, 3)
        T2 = self.L2(features[1]).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)
        return out


class RaddetDetectionHeader(nn.Module):
    """RADDet/model/FFTRadNet_ViT.py Detection_Header: ([sigmoid class, range, angle],
    6 class logits)."""

    def __init__(self, use_bn=True, reg_layer=2, input_angle_size=128):
        super().__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 128
        bias = not use_bn

        if(self.input_angle_size==128):
            self.conv1 = conv3x3(128, 64, bias=bias)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64, 32, bias=bias)
            self.bn2 = nn.BatchNorm2d(32)
        elif(self.input_angle_size==448):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==896):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias,stride=(1,2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle paraemter !')

        self.conv3 = conv3x3(32, 32, bias=bias)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = conv3x3(32, 32, bias=bias)
        self.bn4 = nn.BatchNorm2d(32)

        self.clshead = conv3x3(32, 1, bias=True)
        self.reghead = conv3x3(32, reg_layer, bias=True)
        self.objhead = conv3x3(32,6,bias=True)

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        obj = self.objhead(x)

        return torch.cat([cls, reg], dim=1),obj


class RaddetSwinDecoder(nn.Module):
    """RADDet/model/FFTRadNet_ViT.py Swin_RangeAngle_Decoder (RD and ADC inputs).
    deconv2 is built but unused, as in the reference; it is kept for the checkpoint keys."""

    def __init__(self, ):
        super().__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(4, 4, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))

        self.conv_block4 = BasicBlock(12,64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(80,128)
        self.deconv2 = nn.ConvTranspose2d(80,80,kernel_size=3,stride=(2,1),padding=1,output_padding=(1,0))
        self.L4  = nn.Conv2d(384*1,128,kernel_size=1,stride=1,padding=0)
        self.L3  = nn.Conv2d(192*1, 128, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(96*1, 128, kernel_size=1, stride=1,padding=0)
        self.drop = nn.Dropout2d(p=0.0)

    def forward(self,features):
        T4 = self.L4(self.drop(features[3])).transpose(1, 3)
        T3 = self.L3(self.drop(features[2])).transpose(1, 3)
        T2 = self.L2(self.drop(features[1])).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)

        return out


class RaddetSwinDecoderRAD(nn.Module):
    """RADDet/model/FFTRadNet_ViT.py Swin_RangeAngle_Decoder_RAD (RAD cube input, Doppler
    as channels). conv_block4 is built but unused, as in the reference; kept for the keys."""

    def __init__(self, ):
        super().__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))

        self.conv_block4 = BasicBlock(48,64)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1))
        self.conv_block3 = BasicBlock(128,128)
        self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=3,stride=(1,2),padding=1,output_padding=(0,1))
        self.L4  = nn.Conv2d(384*1,128,kernel_size=1,stride=1,padding=0)
        self.L3  = nn.Conv2d(192*1, 128, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(96*1, 128, kernel_size=1, stride=1,padding=0)
        self.drop = nn.Dropout2d(p=0.0)

    def forward(self,features):
        # Doppler, Range, Angle
        T4 = self.L4(self.drop(features[3]))
        T3 = self.L3(self.drop(features[2]))
        T2 = self.L2(self.drop(features[1]))
        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        S43 = self.deconv2(S43)
        out = self.conv_block3(S43)
        return out
