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
        make_sample_adapter("TransRAD", RADDetDataset(), ["RAD"], {}, {})
    msg = str(e.value)
    assert "TransRAD" in msg and "RADDetDataset" in msg and "registered" in msg


def test_registry_is_keyed_by_model_and_dataset_and_fans_out_models():
    assert ("RADDet", "RADDetDataset") in _REGISTRY
    assert ("DAROD", "RADDetDataset") in _REGISTRY
    assert _REGISTRY[("RADDet", "RADDetDataset")] is not _REGISTRY[("DAROD", "RADDetDataset")]
    for (model, dataset), cls in _REGISTRY.items():
        assert model in cls.MODELS and dataset == cls.DATASET
    assert MIGRATED == frozenset({"RADDetDataset"})


def test_base_len_delegates_to_parser_and_keeps_split_and_features():
    class Probe(SampleAdapter):
        MODELS, DATASET = ("Probe",), "RADDetDataset"

        def __getitem__(self, i):
            return {"i": np.array(i)}

    a = Probe(RADDetDataset(n=7), ["RD"], {}, {"transformations": ["hflip"]}, splittype="val")
    assert len(a) == 7
    assert a.features == ["RD"] and a.splittype == "val" and a.transformations == ["hflip"]
    assert a[2]["i"] == 2
