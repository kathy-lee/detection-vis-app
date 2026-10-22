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


def test_rca_annotations_load_lazily_and_leave_parse_annotations_intact(parser):
    before = parser.annos
    rca = parser.rca_annotations()
    assert "tracks" in rca and parser.annos is before
    frame = next(k for k in rca if k != "tracks" and rca[k])
    assert set(rca[frame]) >= {"cls", "idx"}
