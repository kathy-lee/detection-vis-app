"""Shared by T-FFTRadNet's RADIal and RADDet adapters: the input statistics and which
sample key (feature) each input mode is served under. "The reference repo" is
jgiroux8/T_FFTRadNet 68bc041, the code these adapters reproduce."""
import json
import os

import numpy as np

STATISTICS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfftradnet_statistics.json")

# The training loop feeds the network data[features[0]], so a mode's input is served under
# the feature the user chose, and the two must agree.
MODE_FEATURE = {
    "RD": "RD", "RD_shift_as_published": "RD", "RD_shift": "RD", "RD_no_window": "RD",
    "ADC": "ADC", "RAD": "RAD", "RAD_stored": "RAD",
}


def load_tfftradnet_statistics(dataset, mode):
    """{'input_mean', 'input_std'} of one mode; None for the ADC modes, which the reference
    repo does not normalize. A mode whose statistics are not computed yet raises, naming the script."""
    if mode == "ADC":
        return None
    with open(STATISTICS_FILE) as f:
        stats = json.load(f)[dataset][mode]
    if stats is None:
        raise ValueError(f"T-FFTRadNet input statistics for {dataset} {mode} not computed: "
                         "run scripts/tfftradnet_input_statistics.py")
    return stats


def normalize_channels_last(x, stats):
    """Subtract each channel's mean and divide by its std, channels on the last axis, as
    the reference's loaders do."""
    if stats is None:
        return x
    return ((x - np.asarray(stats["input_mean"])) / np.asarray(stats["input_std"])).astype(np.float32)


def feature_of_mode(model_name, dataset, mode, features):
    """The sample key for mode, checked against the chosen feature."""
    expected = MODE_FEATURE[mode]
    if list(features) != [expected]:
        raise ValueError(f"{model_name} on {dataset}: input_mode {mode!r} reads feature {expected!r}, "
                         f"but the chosen features are {list(features)}")
    return expected
