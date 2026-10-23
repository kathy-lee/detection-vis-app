"""CARRADA as a parser and its sample adapters. Uses one real sequence; skipped when the
dataset or the database is unavailable."""
import json
import os

import numpy as np
import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "carrada_golden.json")
with open(FIXTURE) as _f:
    _G = json.load(_f)

pytestmark = pytest.mark.skipif(not os.path.isdir(_G["root_path"]),
                                reason=f"CARRADA dataset not at {_G['root_path']}")


@pytest.fixture(scope="module")
def parser():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TMP_ROOTDIR", _G["tmp_rootdir"])
        from detection_vis_backend.datasets.dataset import DatasetFactory
        f = DatasetFactory()
        f.get_shared_instance("CARRADA", _G["file_id"],
                              parse_args=(_G["file_id"], _G["root_path"], _G["file_name"], _G["config"]))
        yield f.get_instance("CARRADA", _G["file_id"])


def test_parser_len_is_frames_and_serves_grids_without_prepare(parser):
    assert len(parser) == parser.frame_sync > 0
    assert len(parser.rng_grid) == 256 and len(parser.agl_grid) == 256 and len(parser.dpl_grid) == 64
    assert parser.n_class == 4


def test_parser_annotation_is_the_frame_oriented_entry(parser):
    i = next(i for i in range(len(parser)) if parser.annotation(i))
    entry = parser.annotation(i)
    assert entry is parser.annos["{:06d}".format(i)]
    assert "range_angle" in next(iter(entry.values()))


def test_parser_map_and_mask_views(parser):
    assert parser.processed_map("range_doppler", 3).shape == (256, 64)
    assert parser.processed_map("range_angle", 3).shape == (256, 256)
    assert parser.dense_mask("range_doppler", 3).shape[0] == 4


def test_registry_fans_out_the_six_carrada_consumers():
    from detection_vis_backend.datasets.adapters import _REGISTRY
    for m in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti", "MVRECORD", "DAROD", "RadarCrossAttention"):
        assert (m, "CARRADA") in _REGISTRY, m


def _case(model, features):
    return next(c for c in _G["cases"] if c["model"] == model and c["features"] == features)


@pytest.mark.parametrize("model,features", [("RECORD", ["RD"]), ("DAROD", ["RD"]),
                                            ("MVRECORD", ["RD", "RA", "AD"])])
def test_val_and_test_samples_are_unaugmented(parser, model, features):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    c = _case(model, features)
    for split in ("val", "test"):
        a = make_sample_adapter(model, parser, features, c["model_cfg"], c["train_cfg"], splittype=split)
        np.random.seed(0)
        s0 = np.asarray(a[5][features[0]])
        np.random.seed(1)
        s1 = np.asarray(a[5][features[0]])
        assert np.array_equal(s0, s1), split


def test_radar_cross_attention_draws_noise_only_on_train(parser):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    c = _case("RadarCrossAttention", ["RD", "RA", "AD"])
    assert c["train_cfg"].get("gauss_noise_std")
    val = make_sample_adapter("RadarCrossAttention", parser, c["features"], c["model_cfg"], c["train_cfg"], "val")
    train = make_sample_adapter("RadarCrossAttention", parser, c["features"], c["model_cfg"], c["train_cfg"], "train")
    assert val.noise_std == 0.0 and train.noise_std > 0


def test_inference_overlay_frame_is_the_last_frame_of_the_adapter_window(parser):
    # The window length lives on the adapter now, not on the parser; the overlay must
    # still be drawn on the frame the model predicts for.
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    from detection_vis_backend.train.inference import display_frame
    c = _case("RECORD", ["RA"])
    a = make_sample_adapter("RECORD", parser, ["RA"], c["model_cfg"], c["train_cfg"], "test")
    assert a.win_frames > 1
    assert display_frame(a, 10) == 10 + a.win_frames - 1


def test_ad_cache_script_lists_frames_from_the_rca_adapter(parser, monkeypatch):
    import importlib
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    monkeypatch.syspath_prepend(os.path.join(repo, "scripts"))
    from detection_vis_backend.datasets import dataset as dataset_mod
    monkeypatch.setattr(dataset_mod.DatasetFactory, "get_shared_instance", lambda self, *a, **k: parser)
    monkeypatch.setattr(dataset_mod.DatasetFactory, "get_instance", lambda self, *a, **k: parser)
    sys.modules.pop("build_carrada_ad_cache", None)
    script = importlib.import_module("build_carrada_ad_cache")
    c = _case("RadarCrossAttention", ["RD", "RA", "AD"])
    f = {"id": _G["file_id"], "parse": "CARRADA", "path": "/p", "name": "n", "config": "c"}
    frames, cache_dir = script._frames(script._source(f, c["model_cfg"], c["train_cfg"]))
    assert frames and cache_dir == parser.ad_cache_dir
    assert all(0 <= fr < len(parser) for fr in frames)
    sys.modules.pop("build_carrada_ad_cache", None)


def test_rca_annotations_load_lazily_and_leave_parse_annotations_intact(parser):
    before = parser.annos
    rca = parser.rca_annotations()
    assert "tracks" in rca and parser.annos is before
    frame = next(k for k in rca if k != "tracks" and rca[k])
    assert set(rca[frame]) >= {"cls", "idx"}
