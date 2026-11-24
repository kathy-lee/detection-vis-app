"""Golden fixtures hold paths relative to the dataset root, so they are the same on every
machine; the helpers put the root back on load and take it off when digesting."""
import json

from detection_vis_backend.tests.golden_utils import digest_sample, load_golden


def test_load_golden_resolves_paths_under_the_dataset_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATASET_ROOTDIR", "/d")
    monkeypatch.setenv("TMP_ROOTDIR", "/d/tmp")
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"root_path": "CRUW", "config": "CRUW/sensor.json", "cases": []}))
    g = load_golden(f)
    assert g["root_path"] == "/d/CRUW"
    assert g["config"] == "/d/CRUW/sensor.json"
    assert g["tmp_rootdir"] == "/d/tmp"


def test_load_golden_leaves_an_empty_config(monkeypatch, tmp_path):
    monkeypatch.setenv("DATASET_ROOTDIR", "/d")
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"root_path": "RADDet", "config": ""}))
    assert load_golden(f)["config"] == ""


def test_digest_keeps_strings_relative_to_the_dataset_root(monkeypatch):
    monkeypatch.setenv("DATASET_ROOTDIR", "/d")
    assert digest_sample({"image_path": "/d/RADDet/stereo_image/000001.jpg"}) == \
        {"image_path": "RADDet/stereo_image/000001.jpg"}
