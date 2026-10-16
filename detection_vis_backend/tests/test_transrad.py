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
