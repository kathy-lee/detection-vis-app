"""The machine's data locations come from the environment or the repo's .env file, and
nowhere else."""
import os

import pytest

from detection_vis_backend import paths

ROOTS = ("DATASET_ROOTDIR", "MODEL_ROOTDIR", "TMP_ROOTDIR")


@pytest.fixture
def no_roots(monkeypatch):
    for name in ROOTS:
        monkeypatch.delenv(name, raising=False)


def test_env_file_is_parsed(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# machine paths\n\nDATASET_ROOTDIR=/data\nMODEL_ROOTDIR = "/data/models"\n'
                 "TMP_ROOTDIR='/data/tmp'\nexport OTHER=x\n")
    assert paths.read_env_file(f) == {"DATASET_ROOTDIR": "/data", "MODEL_ROOTDIR": "/data/models",
                                      "TMP_ROOTDIR": "/data/tmp", "OTHER": "x"}


def test_missing_env_file_reads_as_empty(tmp_path):
    assert paths.read_env_file(tmp_path / ".env") == {}


def test_env_file_fills_only_unset_variables(monkeypatch, no_roots, tmp_path):
    f = tmp_path / ".env"
    f.write_text("DATASET_ROOTDIR=/from/file\nMODEL_ROOTDIR=/from/file/models\n")
    monkeypatch.setenv("MODEL_ROOTDIR", "/from/shell/models")
    paths.load_env(f)
    assert os.environ["DATASET_ROOTDIR"] == "/from/file"
    assert os.environ["MODEL_ROOTDIR"] == "/from/shell/models"


def test_roots_come_from_the_env_file(monkeypatch, no_roots, tmp_path):
    f = tmp_path / ".env"
    f.write_text("DATASET_ROOTDIR=/d\nMODEL_ROOTDIR=/d/m\nTMP_ROOTDIR=/d/t\n")
    monkeypatch.setattr(paths, "ENV_FILE", f)
    assert (paths.dataset_root(), paths.model_root(), paths.tmp_root()) == ("/d", "/d/m", "/d/t")
    assert paths.dataset_path("CRUW", "x.json") == "/d/CRUW/x.json"


def test_missing_root_names_the_variable_and_the_file(monkeypatch, no_roots, tmp_path):
    monkeypatch.setattr(paths, "ENV_FILE", tmp_path / ".env")
    with pytest.raises(RuntimeError, match=r"DATASET_ROOTDIR.*\.env"):
        paths.dataset_root()


def test_dataset_path_keeps_an_absolute_path(monkeypatch):
    # Database rows written before paths were stored relative to the root stay usable.
    monkeypatch.setenv("DATASET_ROOTDIR", "/d")
    assert paths.dataset_path("/elsewhere/CRUW") == "/elsewhere/CRUW"


def test_relative_to_dataset_root(monkeypatch):
    monkeypatch.setenv("DATASET_ROOTDIR", "/d")
    assert paths.relative_to_dataset_root("/d/RADDet/img.jpg") == "RADDet/img.jpg"
    assert paths.relative_to_dataset_root("/other/img.jpg") == "/other/img.jpg"


def test_stored_dataset_path_is_joined_to_the_root(monkeypatch):
    monkeypatch.setenv("DATASET_ROOTDIR", "/d")
    assert paths.resolve_stored_path("UWCR/Automotive") == "/d/UWCR/Automotive"
    assert paths.resolve_stored_path("/old/UWCR/Automotive") == "/old/UWCR/Automotive"
    assert paths.resolve_stored_path("") == ""


def test_stored_config_is_joined_only_when_it_names_a_file(monkeypatch, tmp_path):
    (tmp_path / "CRUW").mkdir()
    (tmp_path / "CRUW" / "sensor.json").write_text("{}")
    monkeypatch.setenv("DATASET_ROOTDIR", str(tmp_path))
    assert paths.resolve_stored_config("CRUW/sensor.json") == str(tmp_path / "CRUW" / "sensor.json")
    # NuScenes stores its version name here, not a path.
    assert paths.resolve_stored_config("v1.0-mini") == "v1.0-mini"
    assert paths.resolve_stored_config("") == ""
    assert paths.resolve_stored_config("/abs/cfg.json") == "/abs/cfg.json"


def test_repo_env_file_is_ignored_by_git():
    with open(os.path.join(paths.REPO_ROOT, ".gitignore")) as f:
        assert "/.env" in f.read().split()
