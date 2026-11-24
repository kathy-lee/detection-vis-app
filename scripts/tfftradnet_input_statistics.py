"""Per-channel input statistics of the T-FFTRadNet modes upstream never measured.

T-FFTRadNet normalizes each input channel with a mean and std from its config. Upstream
(jgiroux8/T_FFTRadNet 68bc041) ships them for RADIal RD and RD-shift and RADDet RD and RAD;
detection_vis_backend/datasets/adapters/tfftradnet_statistics.json holds those, and null
for the modes it lacks: RADIal RD_no_window, RADDet RD_no_window (the paper's
no-windowing rows) and RADDet RAD_stored (this project's variant reading the stored cubes).

The formula is upstream's dataset/print_dataset_statistics.py: for every frame, each
channel's mean and std over its range x Doppler (or range x angle) cells; then the mean of
those per-frame values over the frames. Upstream runs it over all frames, test ones
included; here only the train split is used (decided 2026-11-24), so no test frame shapes
the input:
  RADIal         valeo's sequence split, train recordings, frames with a radar_ADC file
  RADDetDataset  the official train list of original_split_info.json

    uv run python scripts/tfftradnet_input_statistics.py --dataset RADDetDataset --mode RAD_stored [--frames N]

prints the JSON entry for tfftradnet_statistics.json. --frames takes N evenly spaced train
frames instead of all of them.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from detection_vis_backend.paths import dataset_path, load_env  # noqa: E402

load_env()
os.environ.setdefault("LOGURU_LEVEL", "ERROR")

MODES = {"RADIal": ("RD", "RD_shift_as_published", "RD_shift", "RD_no_window"),
         "RADDetDataset": ("RAD", "RAD_stored", "RD", "RD_no_window")}


def channel_statistics(frames):
    """(mean, std) per channel of channels-first frames: the mean over frames of each
    frame's per-channel mean and std."""
    means, stds, n = 0, 0, 0
    for x in frames:
        flat = x.reshape(x.shape[0], -1).astype(np.float64)
        means = means + flat.mean(axis=1)
        stds = stds + flat.std(axis=1)
        n += 1
    return (means / n).tolist(), (stds / n).tolist()


def radial_parser():
    from detection_vis_backend.datasets.dataset import RADIal
    parser = RADIal()
    parser.parse(0, dataset_path("radial"), "RADIal-labeled", "")
    return parser


def raddet_parser():
    from detection_vis_backend.datasets.dataset import RADDetDataset
    parser = RADDetDataset()
    parser.parse(0, dataset_path("RADDet_reform_dataset"), "RADDet_reform_dataset", "")
    return parser


def radial_train_frames(parser):
    """Train frames of valeo's sequence split that have a radar_ADC file."""
    from detection_vis_backend.datasets.adapters.radial import radial_sequence_split
    adc_dir = os.path.join(parser.root_dir, "radar_ADC")
    present = set(os.listdir(adc_dir))
    return [int(i) for i in radial_sequence_split(parser)["train"]
            if f"adc_{parser.sample_keys[i]:06d}.npy" in present]


def raddet_train_frames(parser):
    """Parser indices of RADDet's official train list."""
    with open(os.path.join(parser.root_path, "original_split_info.json")) as f:
        train = set(json.load(f)["train"])
    ids = [os.path.splitext(os.path.basename(p))[0] for p in parser.anno_filenames]
    return [i for i, frame_id in enumerate(ids) if frame_id in train]


def frames_of(dataset, mode, parser, indices):
    """The mode's input, channels first and before normalization, for each index."""
    if dataset == "RADIal":
        from detection_vis_backend.datasets.adapters.radial import radial_complex_adc, radial_rd_from_adc
        for i in indices:
            if mode == "RD":
                rd = parser.get_RD(i)
            else:
                raw = np.load(os.path.join(parser.root_dir, "radar_ADC", f"adc_{parser.sample_keys[i]:06d}.npy"))
                rd = radial_rd_from_adc(radial_complex_adc(raw), window=mode != "RD_no_window",
                                        shift=mode == "RD_shift")
            yield np.transpose(rd, (2, 0, 1))
    else:
        from detection_vis_backend.datasets.adapters.raddet import (
            raddet_rad_stored, raddet_rad_upstream, raddet_rd_upstream)
        for i in indices:
            if mode == "RAD_stored":
                yield raddet_rad_stored(parser.get_RAD(i))
            elif mode == "RAD":
                yield raddet_rad_upstream(parser.get_ADC(i))
            else:
                yield raddet_rd_upstream(parser.get_ADC(i), window=mode == "RD")


def compute(dataset, mode, n_frames=None):
    """(mean, std) of one mode over the train split, or over n_frames evenly spaced
    train frames."""
    if mode not in MODES[dataset]:
        raise ValueError(f"{dataset} mode {mode!r} not in {MODES[dataset]} (ADC is not normalized)")
    parser = radial_parser() if dataset == "RADIal" else raddet_parser()
    indices = radial_train_frames(parser) if dataset == "RADIal" else raddet_train_frames(parser)
    if n_frames:
        indices = [indices[int(k)] for k in np.linspace(0, len(indices) - 1, n_frames)]
    print(f"{dataset} {mode}: {len(indices)} train frames", file=sys.stderr, flush=True)
    return channel_statistics(frames_of(dataset, mode, parser, indices))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, choices=sorted(MODES))
    p.add_argument("--mode", required=True)
    p.add_argument("--frames", type=int, default=None, help="evenly spaced train frames instead of all")
    args = p.parse_args()
    mean, std = compute(args.dataset, args.mode, args.frames)
    print(json.dumps({args.mode: {"input_mean": mean, "input_std": std}}))


if __name__ == "__main__":
    main()
