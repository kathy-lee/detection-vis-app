"""The migration that makes the files table's paths relative to DATASET_ROOTDIR: its
rules for each kind of stored value, and that downgrade undoes upgrade."""
import importlib.util
import os

import pytest

MIGRATION = os.path.join(os.path.dirname(__file__), "..", "..", "alembic", "versions",
                         "3d1f6a2b9c47_files_paths_relative_to_dataset_root.py")


@pytest.fixture(scope="module")
def mig():
    spec = importlib.util.spec_from_file_location("relative_paths_migration", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_follows_the_last_one(mig):
    assert mig.down_revision == "752367e53636"


@pytest.mark.parametrize("stored,expected", [
    ("/d/CRUW", "CRUW"),
    ("/d/UWCR/Automotive/radar_config.json", "UWCR/Automotive/radar_config.json"),
    ("/elsewhere/CRUW", "/elsewhere/CRUW"),     # outside the root: left absolute
    ("radical/radarcfg", "radical/radarcfg"),   # already relative
    ("v1.0-mini", "v1.0-mini"),                 # NuScenes version name
    ("", ""),
    (None, None),
])
def test_upgrade_strips_the_root(mig, stored, expected):
    assert mig.to_relative(stored, "/d") == expected


def test_upgrade_does_not_strip_a_sibling_folder_sharing_the_prefix(mig):
    assert mig.to_relative("/d2/CRUW", "/d") == "/d2/CRUW"


def test_downgrade_restores_paths_and_leaves_names(mig, tmp_path):
    root = str(tmp_path)
    (tmp_path / "CRUW").mkdir()
    (tmp_path / "CRUW" / "sensor.json").write_text("{}")
    assert mig.to_absolute("CRUW", root, is_config=False) == f"{root}/CRUW"
    assert mig.to_absolute("CRUW/sensor.json", root, is_config=True) == f"{root}/CRUW/sensor.json"
    assert mig.to_absolute("v1.0-mini", root, is_config=True) == "v1.0-mini"
    assert mig.to_absolute("", root, is_config=False) == ""
    assert mig.to_absolute("/abs/x", root, is_config=False) == "/abs/x"


def test_downgrade_undoes_upgrade_for_the_seed_rows(mig, tmp_path):
    root = str(tmp_path)
    (tmp_path / "UWCR" / "Automotive").mkdir(parents=True)
    (tmp_path / "UWCR" / "Automotive" / "radar_config.json").write_text("{}")
    rows = [(f"{root}/UWCR/Automotive", f"{root}/UWCR/Automotive/radar_config.json"),
            (f"{root}/RADDet_reform_dataset", ""),
            (f"{root}/nuscences_v1.0_mini", "v1.0-mini")]
    for path, config in rows:
        back = (mig.to_absolute(mig.to_relative(path, root), root, is_config=False),
                mig.to_absolute(mig.to_relative(config, root), root, is_config=True))
        assert back == (path, config)
