"""TransRAD port: backbone, neck/heads, loss, decoder, NMS, adapter, trainer additions.
Everything here runs on CPU with random tensors or the fake parser; the authors'
checkpoint is used only for key-set checks and skipped when absent."""
import os

import numpy as np
import pytest
import torch

from detection_vis_backend.networks import transrad as T

torch.manual_seed(0)
np.random.seed(0)

HAS_REF = os.path.exists(T.REF_CHECKPOINT)
needs_ref = pytest.mark.skipif(not HAS_REF, reason=f"authors' checkpoint not at {T.REF_CHECKPOINT}")


def test_drop_path_is_identity_in_eval_and_scales_in_train():
    dp = T.DropPath(0.5)
    x = torch.ones(8, 3, 2, 2)
    dp.eval()
    assert torch.equal(dp(x), x)
    dp.train()
    y = dp(x)
    kept = y[:, 0, 0, 0]
    assert set(kept.tolist()) <= {0.0, 2.0}          # survivors are rescaled by 1/(1-p)


def test_trunc_normal_stays_inside_bounds():
    t = torch.empty(10000)
    T.trunc_normal_(t, std=0.02, a=-0.04, b=0.04)
    assert t.min() >= -0.04 and t.max() <= 0.04
    assert abs(t.std().item() - 0.02) < 0.005


RMT_KW = dict(in_chans=256, out_indices=(0, 1, 2, 3), embed_dims=[32, 64, 128, 256],
              depths=[2, 2, 8, 2], num_heads=[4, 4, 8, 16], init_values=[2, 2, 2, 2],
              heads_ranges=[4, 4, 6, 6], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1,
              norm_layer=torch.nn.LayerNorm, patch_norm=True, use_checkpoint=False,
              chunkwise_recurrents=[True, True, False, False], projection=1024,
              layerscales=[False, False, False, False], layer_init_values=1e-6, norm_eval=True)


def test_rmt_backbone_feature_shapes_at_full_size():
    net = T.RMT(**RMT_KW).eval()
    with torch.no_grad():
        f1, f2, f3 = net(torch.randn(1, 256, 256, 256))
    assert tuple(f1.shape) == (1, 64, 32, 32)
    assert tuple(f2.shape) == (1, 128, 16, 16)
    assert tuple(f3.shape) == (1, 256, 8, 8)


def test_rmt_train_mode_keeps_batchnorm_in_eval():
    """norm_eval=True: the reference freezes BatchNorm statistics during training."""
    net = T.RMT(**RMT_KW)
    net.train()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)


@needs_ref
def test_rmt_keys_match_the_authors_backbone():
    sd = torch.load(T.REF_CHECKPOINT, map_location="cpu")
    ref = {k[len("backbone."):] for k in sd if k.startswith("backbone.")}
    ours = set(T.RMT(**RMT_KW).state_dict())
    assert ours == ref, f"missing {sorted(ref - ours)[:5]} unexpected {sorted(ours - ref)[:5]}"
