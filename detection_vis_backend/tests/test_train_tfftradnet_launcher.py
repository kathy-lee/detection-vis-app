"""The T-FFTRadNet command-line launcher picks an input mode that reads the chosen feature."""
import importlib.util
import os

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train_tfftradnet.py")


def _launcher():
    spec = importlib.util.spec_from_file_location("train_tfftradnet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dataset, feature", [
    ("RADDetDataset", "RD"), ("RADDetDataset", "RAD"), ("RADDetDataset", "ADC"),
    ("RADIal", "RD"), ("RADIal", "ADC"),
])
def test_input_mode_defaults_to_the_feature(dataset, feature):
    model_config, _ = _launcher().configs(dataset, feature, None)
    assert model_config["input_mode"] == feature


def test_an_explicit_input_mode_wins():
    model_config, _ = _launcher().configs("RADIal", "RD", "RD_shift")
    assert model_config["input_mode"] == "RD_shift"
