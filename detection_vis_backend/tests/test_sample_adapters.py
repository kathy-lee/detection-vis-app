"""The adapter registry and base class, exercised with fake parsers -- no files, no DB."""
import numpy as np
import pytest

from detection_vis_backend.datasets.adapters import (
    Parser, SampleAdapter, make_sample_adapter, _REGISTRY, MIGRATED,
)


class RADDetDataset:            # the registry keys on the class NAME, so the fake shares it
    def __init__(self, n=3):
        self._n = n

    def __len__(self):
        return self._n

    def annotation(self, i):
        return {"classes": [], "boxes": [], "cart_boxes": []}


class NotAParser:
    def __len__(self):
        return 1


def test_fake_parser_satisfies_protocol_without_inheriting():
    assert isinstance(RADDetDataset(), Parser)
    assert Parser not in type(RADDetDataset()).__mro__


def test_missing_annotation_is_a_typeerror_at_construction():
    with pytest.raises(TypeError, match="annotation"):
        make_sample_adapter("RADDet", NotAParser(), ["RAD"], {}, {})


def test_unregistered_pair_names_both_sides_and_lists_registered():
    with pytest.raises(ValueError) as e:
        make_sample_adapter("NoSuchModel", RADDetDataset(), ["RAD"], {}, {})
    msg = str(e.value)
    assert "NoSuchModel" in msg and "RADDetDataset" in msg and "registered" in msg


def test_registry_is_keyed_by_model_and_dataset_and_fans_out_models():
    assert ("RADDet", "RADDetDataset") in _REGISTRY
    assert ("DAROD", "RADDetDataset") in _REGISTRY
    assert _REGISTRY[("RADDet", "RADDetDataset")] is not _REGISTRY[("DAROD", "RADDetDataset")]
    for (model, dataset), cls in _REGISTRY.items():
        assert model in cls.MODELS and dataset == cls.DATASET
    assert MIGRATED == frozenset({"RADDetDataset", "CRUW", "CARRADA"})


def test_base_len_delegates_to_parser_and_keeps_split_and_features():
    class Probe(SampleAdapter):
        MODELS, DATASET = ("Probe",), "RADDetDataset"

        def __getitem__(self, i):
            return {"i": np.array(i)}

    a = Probe(RADDetDataset(n=7), ["RD"], {}, {"transformations": ["hflip"]}, splittype="val")
    assert len(a) == 7
    assert a.features == ["RD"] and a.splittype == "val" and a.transformations == ["hflip"]
    assert a[2]["i"] == 2


# ---------------------------------------------------------------------------------------
# RADDetDataset adapters, against a fake parser that serves a tiny deterministic cube.
# ---------------------------------------------------------------------------------------
from detection_vis_backend.datasets.adapters.raddet import (  # noqa: E402
    RADDetOnRADDet, DARODOnRADDet, encodeToLabels, random_flip,
)

DATA_STATS = {
    "all_classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
    "global_mean_log": 3.2438383, "global_max_log": 10.0805629, "global_min_log": 0.0,
    "global_variance_log": 6.8367246, "max_boxes_per_frame": 30, "trainset_portion": 0.8,
}
RADDET_MODEL_CFG = {"class": "RADDet", "input_size": [256, 256, 64],
                    "anchor_boxes": [[6, 9, 5], [19, 17, 8], [19, 28, 19],
                                     [22, 45, 28], [19, 36, 60], [26, 75, 61]]}
DAROD_TRAIN_CFG = {"transformations": ["hflip", "vflip"],
                   "data_mean": 830.4466, "data_std": 93.25871}


class FakeRADDet:
    """Shares RADDetDataset's NAME (set below) so the registry resolves it."""
    data_stats = DATA_STATS

    def __init__(self, annos):
        self._annos = annos
        rng = np.random.default_rng(0)
        # complex64 like the .npy cubes on disk; complexTo2Channels asserts that dtype
        self._cube = (rng.standard_normal((256, 256, 64))
                      + 1j * rng.standard_normal((256, 256, 64))).astype(np.complex64)

    def __len__(self):
        return len(self._annos)

    def annotation(self, i):
        return self._annos[i]

    def get_RAD(self, i):
        return self._cube

    def get_image(self, i):
        return f"/img/{i}.jpg"

    def _get_collapsed(self, i, view):
        from detection_vis_backend.datasets.utils import complexTo2Channels
        axis = 1 if view == "RD" else -1
        return np.sum(complexTo2Channels(self._cube), axis=axis).astype(np.float32)


FakeRADDet.__name__ = "RADDetDataset"

ONE_CAR = {"classes": ["car"], "boxes": [np.array([100., 120., 30., 10., 12., 6.])], "cart_boxes": [None]}
WRAP = {"classes": ["bus"], "boxes": [np.array([50., 60., 2., 8., 10., 10.])], "cart_boxes": [None]}  # x1 = 2-5 < 0
EMPTY = {"classes": [], "boxes": [], "cart_boxes": []}


def test_raddet_sample_keys_shapes_dtypes():
    a = make_sample_adapter("RADDet", FakeRADDet([ONE_CAR]), ["RAD"], RADDET_MODEL_CFG, {}, "val")
    s = a[0]
    assert set(s) == {"RAD", "label", "boxes", "image_path"}
    assert s["RAD"].shape == (64, 256, 256)                 # transposed (2, 0, 1)
    assert s["label"].shape == (16, 16, 4, 6, 13)
    assert s["boxes"].shape == (30, 7)
    assert s["image_path"] == "/img/0.jpg"
    # channel 6 is objectness (1.0 on the matched anchors); zeros everywhere became 1e-16
    assert s["label"][..., 6].max() == 1.0 and (s["label"] >= 1e-16).all()


def test_raddet_empty_frame_has_no_positive_label():
    a = make_sample_adapter("RADDet", FakeRADDet([EMPTY]), ["RAD"], RADDET_MODEL_CFG, {}, "val")
    s = a[0]
    assert (s["label"] == 1e-16).all() and (s["boxes"] == 0).all()


def test_darod_sample_keys_shapes_and_standardization():
    a = make_sample_adapter("DAROD", FakeRADDet([ONE_CAR]), ["RD"], {"class": "DAROD"}, DAROD_TRAIN_CFG, "val")
    s = a[0]
    assert set(s) == {"RD", "label", "boxes", "image_path"}
    assert s["RD"].shape == (1, 256, 64) and s["RD"].dtype == np.float32
    assert s["label"].tolist() == [3]                        # "car" -> index 2, plus 1
    assert s["boxes"].shape == (1, 4) and (0 <= s["boxes"]).all() and (s["boxes"] <= 1).all()
    # the exact affine chain: (S - 256*mean)/var then (x - data_mean)/data_std
    p = a.parser
    S = p._get_collapsed(0, "RD")
    expected = ((S - 256 * DATA_STATS["global_mean_log"]) / DATA_STATS["global_variance_log"]
                - DAROD_TRAIN_CFG["data_mean"]) / DAROD_TRAIN_CFG["data_std"]
    np.testing.assert_array_equal(s["RD"][0], expected)


def test_darod_wraparound_box_is_split_in_two():
    a = make_sample_adapter("DAROD", FakeRADDet([WRAP]), ["RD"], {"class": "DAROD"}, DAROD_TRAIN_CFG, "val")
    s = a[0]
    assert s["boxes"].shape == (2, 4) and s["label"].tolist() == [5, 5]


def test_darod_empty_frame_returns_empty_arrays_of_the_old_shape():
    a = make_sample_adapter("DAROD", FakeRADDet([EMPTY]), ["RD"], {"class": "DAROD"}, DAROD_TRAIN_CFG, "val")
    s = a[0]
    assert s["label"].shape == (0,) and s["boxes"].shape == (0,)


def test_darod_rejects_features_it_cannot_build():
    with pytest.raises(ValueError, match=r"\['RAD'\]"):
        make_sample_adapter("DAROD", FakeRADDet([ONE_CAR]), ["RAD"], {"class": "DAROD"}, DAROD_TRAIN_CFG, "val")


def test_darod_augments_only_on_train_and_honours_transformations_none():
    parser = FakeRADDet([ONE_CAR])
    val = make_sample_adapter("DAROD", parser, ["RD"], {"class": "DAROD"}, DAROD_TRAIN_CFG, "val")[0]
    np.random.seed(0)                       # first uniform draw with seed 0 is 0.5488 -> no flip
    train_noflip = make_sample_adapter("DAROD", parser, ["RD"], {"class": "DAROD"}, DAROD_TRAIN_CFG, "train")[0]
    np.testing.assert_array_equal(val["RD"], train_noflip["RD"])
    np.random.seed(1)                       # first draw with seed 1 is 0.4170 -> vflip
    train_flip = make_sample_adapter("DAROD", parser, ["RD"], {"class": "DAROD"}, DAROD_TRAIN_CFG, "train")[0]
    assert not np.array_equal(val["RD"], train_flip["RD"])
    # inference's override: a train-split adapter with transformations=None never flips
    inf = make_sample_adapter("DAROD", parser, ["RD"], {"class": "DAROD"}, DAROD_TRAIN_CFG, "train")
    inf.transformations = None
    np.random.seed(1)
    np.testing.assert_array_equal(val["RD"], inf[0]["RD"])


def test_random_flip_is_a_noop_without_transformations():
    frame = np.arange(6.).reshape(2, 3)
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]])
    f, b = random_flip(frame, boxes, None)
    assert f is frame and b is boxes


def test_raddet_parser_has_stats_and_cache_dir_before_prepare(monkeypatch, tmp_path):
    """The adapter reads parser.data_stats at construction, and the cache dir must not
    depend on run config -- both belong to __init__, not prepare_for_train."""
    monkeypatch.setenv("TMP_ROOTDIR", str(tmp_path))
    from detection_vis_backend.datasets.dataset import RADDetDataset
    ds = RADDetDataset()
    assert ds.data_stats["all_classes"][2] == "car"
    assert ds.collapsed_cache_dir.startswith(str(tmp_path))
    assert callable(ds.annotation)
