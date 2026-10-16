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


BODY_KW = dict(input_size=[256, 256, 256], n_class=6)


def test_body_output_shapes_and_anchor_count():
    body = T.TransRADBody(**BODY_KW).eval()
    with torch.no_grad():
        dbox, cls, heads, anchors, strides = body(torch.randn(1, 256, 256, 256))
    assert body.no == 6 + 4 * 16 + 1 + 2
    assert [tuple(h.shape) for h in heads] == [(1, 73, 32, 32), (1, 73, 16, 16), (1, 73, 8, 8)]
    assert tuple(dbox.shape) == (1, 4, 1344) and tuple(cls.shape) == (1, 6, 1344)
    assert tuple(anchors.shape) == (2, 1344) and tuple(strides.shape) == (1, 1344)
    assert body.stride.tolist() == [8.0, 16.0, 32.0]


def test_body_strides_equal_a_dummy_forward():
    """The reference computes stride = input/feature-size with a dummy forward at
    construction; the port fixes it from the architecture and this pins the equality."""
    body = T.TransRADBody(**BODY_KW).eval()
    with torch.no_grad():
        feats = body.backbone(torch.zeros(1, 256, 256, 256))
    assert [256 / f.shape[-2] for f in feats] == body.stride.tolist()


@needs_ref
def test_body_state_dict_matches_the_authors_checkpoint_exactly():
    sd = torch.load(T.REF_CHECKPOINT, map_location="cpu")
    body = T.TransRADBody(**BODY_KW)
    assert set(body.state_dict()) == set(sd)
    assert {k: tuple(v.shape) for k, v in body.state_dict().items()} == {k: tuple(v.shape) for k, v in sd.items()}
    body.load_state_dict(sd, strict=True)
