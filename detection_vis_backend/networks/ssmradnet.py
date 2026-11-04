"""SSMRadNet (Sen, Mohammad, Mukhopadhyay, WACV 2026, arXiv 2511.08769): sample-wise then
chirp-wise Mamba SSMs on raw RADIal ADC, with detection and freespace decoders.

Ported from github.com/AnuvabSen1/SSMRadNet at commit 39b982b, FFTRadNet/model/SSMRadNet.py
(classes MambaSSMBlock and RADFE), layer for layer so its checkpoints load unchanged.
Two departures, each marked where it happens: forward returns the detection output too,
and the pooled chirp features squeeze only their last axis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2


class JaccardLoss(nn.Module):
    """Soft IoU loss on the freespace logits, the paper's segmentation loss:
    1 - (sum p*t + 1) / (sum p + sum t - sum p*t + 1) per sample, p = sigmoid(logits),
    averaged over the batch."""

    def forward(self, logits, target):
        p = torch.sigmoid(logits).flatten(1)
        t = target.float().flatten(1)
        inter = (p * t).sum(1)
        union = p.sum(1) + t.sum(1) - inter
        return (1 - (inter + 1) / (union + 1)).mean()


class MambaSSMBlock(nn.Module):
    """Mamba SSM followed by a projection back to d_model. ssm="mamba" is the authors' layer;
    "mamba2" swaps in Mamba-2, a variant that is not in the paper."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, ssm="mamba"):
        super().__init__()
        if ssm == "mamba":
            self.ssm = Mamba(d_model, d_state, d_conv=d_conv, expand=expand)
        elif ssm == "mamba2":
            # 8 heads: Mamba2's fused causal conv needs its in-projection width
            # (2*inner + 2*d_state + heads) to be a multiple of 8, which at these narrow
            # widths (32 and 64) only 8 heads give.
            self.ssm = Mamba2(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                              headdim=expand * d_model // 8)
        else:
            raise ValueError(f"unknown ssm {ssm!r}: expected 'mamba' or 'mamba2'")
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return F.silu(self.out_proj(self.ssm(x)))


class RADFE(nn.Module):
    """Input (B, 512 samples, 256 chirps, 32 = 16 Rx real + imag); output the detection map
    (B, 3, 128, 224) as [cls (sigmoid), 2 reg] and the freespace logits (B, 1, 128, 128)."""

    def __init__(self, fast_time_len=512, fast_time_layers=1, fast_time_linear_dims=(32,),
                 slow_time_len=256, slow_time_layers=1, slow_time_linear_dims=(64,),
                 conv_channels=8, radar_channels=32, dropout_prob=0.3, ssm="mamba"):
        super().__init__()
        self.channels = radar_channels
        # Stored but never applied, as upstream.
        self.dropout_prob = dropout_prob

        self.inputNorm = nn.LayerNorm(self.channels)
        self.fast_linears = nn.ModuleList()
        self.fast_norms = nn.ModuleList()
        last_dim = self.channels
        for next_dim in fast_time_linear_dims:
            self.fast_linears.append(nn.Linear(last_dim, next_dim))
            self.fast_norms.append(nn.LayerNorm(next_dim))
            last_dim = next_dim
        self.fast_time_dim = last_dim
        self.fast_time_len = fast_time_len
        # Unused upstream too (the addition is commented out); kept so checkpoint keys match.
        self.fast_time_positional_encoding = nn.Parameter(torch.zeros(1, self.fast_time_len, self.fast_time_dim))
        self.fast_ssm_layers = nn.ModuleList([
            MambaSSMBlock(self.fast_time_dim, d_state=32, d_conv=4, expand=2, ssm=ssm)
            for _ in range(fast_time_layers)])

        self.slow_linears = nn.ModuleList()
        self.slow_norms = nn.ModuleList()
        last_dim = self.fast_time_dim
        for next_dim in slow_time_linear_dims:
            self.slow_linears.append(nn.Linear(last_dim, next_dim))
            self.slow_norms.append(nn.LayerNorm(next_dim))
            last_dim = next_dim
        self.chirp_feature_pooling = nn.AdaptiveAvgPool1d(1)
        self.slow_time_dim = last_dim
        self.slow_time_len = slow_time_len
        self.slow_time_positional_encoding = nn.Parameter(torch.zeros(1, self.slow_time_len, self.slow_time_dim))
        self.slow_ssm_layers = nn.ModuleList([
            MambaSSMBlock(self.slow_time_dim, d_state=32, d_conv=4, expand=2, ssm=ssm)
            for _ in range(slow_time_layers)])

        self.dim2D = [32, 32]
        self.project = nn.Conv1d(2 * self.slow_time_dim, self.dim2D[0] * self.dim2D[1], kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Segmentation decoder
        self.conv0 = nn.Conv2d(1, conv_channels, kernel_size=3, padding='same')
        self.upsample1 = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        self.conv1_1 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding='same')
        self.conv1_2 = nn.Conv2d(conv_channels, conv_channels // 2, kernel_size=3, padding='same')
        self.upsample2 = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        self.conv2_1 = nn.Conv2d(conv_channels // 2, conv_channels // 2, kernel_size=3, padding='same')
        self.conv2_2 = nn.Conv2d(conv_channels // 2, 1, kernel_size=3, padding='same')

        # Detection decoder
        self.pool_det = nn.AdaptiveAvgPool1d(32)
        self.conv1 = nn.Conv2d(self.slow_time_len, 32, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.up2 = nn.Upsample(size=(64, 112), mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.up3 = nn.Upsample(size=(128, 224), mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(8)
        self.clshead = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding='same', bias=True)
        self.reghead = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding='same', bias=True)

    def encode(self, x):
        B, sample, chirp, C = x.shape
        assert sample == self.fast_time_len and chirp == self.slow_time_len and C == self.channels

        x_t = self.inputNorm(x)
        for linear, norm in zip(self.fast_linears, self.fast_norms):
            x_t = norm(F.silu(linear(x_t)))

        # One sequence of samples per chirp: (B*chirps, samples, dim).
        x_fast = x_t.permute(0, 2, 1, 3).reshape(B * self.slow_time_len, self.fast_time_len, self.fast_time_dim)
        for layer in self.fast_ssm_layers:
            x_fast = layer(x_fast)

        # Expanded before pooling over the samples, as upstream (the paper's text has the reverse).
        x_fast_expand = x_fast
        for linear, norm in zip(self.slow_linears, self.slow_norms):
            x_fast_expand = norm(F.silu(linear(x_fast_expand)))

        # squeeze(-1), not upstream's squeeze(): that also dropped the batch axis at batch size 1.
        x_chirp = self.chirp_feature_pooling(x_fast_expand.transpose(1, 2)).squeeze(-1)
        x_chirp = x_chirp.reshape(B, self.slow_time_len, self.slow_time_dim)

        x_slow = x_chirp
        for layer in self.slow_ssm_layers:
            x_slow = layer(x_slow)

        encoder_features = {"x_fast": x_fast, "x_chirp": x_chirp, "x_slow": x_slow}
        return x_slow, x_chirp, encoder_features

    def decode_det(self, x, skip):
        B = x.shape[0]
        x = torch.cat([x, skip], dim=-1)  # (B, chirps, 2*slow_time_dim)
        x_spatial_proj = self.project(x.permute(0, 2, 1))  # (B, 32*32, chirps)
        # Each chirp becomes a channel of a 32x32 map.
        x_spatial = x_spatial_proj.transpose(1, 2).reshape(B, self.slow_time_len, self.dim2D[0], self.dim2D[1])

        x = self.bn1(F.silu(self.conv1(x_spatial)))
        x = self.up2(self.bn2(F.silu(self.conv2(x))))
        x = self.up3(self.bn3(F.silu(self.conv3(x))))
        x = self.bn4(self.conv4(x))

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        return torch.cat([cls, reg], dim=1)

    def decode_seg(self, x, skip):
        B = x.shape[0]
        x = torch.cat([x, skip], dim=-1)
        x_spatial_proj = self.project(x.permute(0, 2, 1))
        # Averaged over the chirps into one 32x32 map.
        x_spatial = self.pool(x_spatial_proj).transpose(1, 2).reshape(B, 1, self.dim2D[0], self.dim2D[1])

        x_out = F.silu(self.conv0(x_spatial))
        x_out = self.upsample1(x_out)
        x_out = F.silu(self.conv1_1(x_out))
        x_out = F.silu(self.conv1_2(x_out))
        x_out = self.upsample2(x_out)
        x_out = F.silu(self.conv2_1(x_out))
        return self.conv2_2(x_out)

    def forward(self, x):
        x_encoded, skip, _ = self.encode(x)
        # Upstream returns only the segmentation (detection commented out) while its
        # wrapper unpacks two outputs; both are returned here.
        return self.decode_det(x_encoded, skip), self.decode_seg(x_encoded, skip)
