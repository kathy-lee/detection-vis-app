"""scripts/build_raddet_cache.py must work on a RADDetDataset that is a plain parser:
no prepare_for_train exists any more, and the cache dir is known from __init__."""
import importlib
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ParserOnly:
    """What RADDetDataset offers today: a cache dir, _get_collapsed, len -- and nothing
    model-specific."""
    collapsed_cache_dir = "/nowhere/raddet_collapsed_cache/abc"
    calls = []

    def __len__(self):
        return 2

    def _get_collapsed(self, index, feature_name):
        self.calls.append((index, feature_name))
        return None


@pytest.fixture
def script(monkeypatch):
    for k in ("MODEL_ROOTDIR", "DATASET_ROOTDIR", "TMP_ROOTDIR", "DATABASE_URL", "LOGURU_LEVEL"):
        monkeypatch.setenv(k, os.environ.get(k, "/tmp/unused"))
    monkeypatch.syspath_prepend(os.path.join(REPO_ROOT, "scripts"))
    import smoke_sweep
    from detection_vis_backend.datasets import dataset as dataset_mod
    f = {"id": 1, "parse": "RADDetDataset", "path": "/p", "name": "n", "config": "c"}
    monkeypatch.setattr(smoke_sweep, "fetch_datafiles", lambda: {"RADDetDataset": [f]})
    monkeypatch.setattr(smoke_sweep, "build_split", lambda *a, **k: ([f], None))
    monkeypatch.setattr(dataset_mod.DatasetFactory, "get_instance", lambda self, *a, **k: ParserOnly())
    sys.modules.pop("build_raddet_cache", None)
    mod = importlib.import_module("build_raddet_cache")
    yield mod
    sys.modules.pop("build_raddet_cache", None)


def test_worker_init_needs_only_the_parser(script):
    ParserOnly.calls.clear()
    script._init("RD")
    assert isinstance(script._ds, ParserOnly)
    assert script._one(1) is True
    assert ParserOnly.calls == [(1, "RD")]
