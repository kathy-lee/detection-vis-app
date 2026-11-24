#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "db-reader @ git+https://github.com/valeoai/RADIal@02f52fc13ee41feacec82fd8e05f5b1f0bf4ce26#subdirectory=DBReader",
#     "cantools",
#     "opencv-python-headless",  # imported by DBReader's camera reader, undeclared in its setup.py
#     "numpy",
#     "pandas",
# ]
# ///
"""Extract the raw radar ADC of every labelled RADIal frame from the raw recordings.

The ready-to-use RADIal release ships range-Doppler spectra (radar_FFT) but no ADC,
which raw-ADC networks need. This reads the recordings with valeo's DBReader and writes,
per labelled frame, the four radar chips' int16 streams stacked as (1048576, 4):

    RADIal-labeled/radar_ADC/adc_{numSample:06d}.npy      (8 MiB each)

That is the earliest stage every raw-ADC consumer shares, so each one decodes it at load
time instead of a preprocessed copy being stored per network. Frames are chosen and named
through labels.csv (dataset, index -> numSample), so the files align with radar_FFT,
radar_Freespace and the split.

Every frame is checked before it is written: its decoded RD spectrum must reproduce
radar_FFT/fft_{numSample:06d}.npy. A frame that does not is reported and not written.

Run it with `uv run` as shown below, if running with `python scripts/...` or `.venv/bin/python scripts/...`, it uses the project environment, which has no DBReader, so it will fail at that import.

    uv run scripts/extract_radial_adc.py [--raw-root DIR] [--labeled-root DIR]

uv builds the script's own environment from the header above, so DBReader and cantools
stay out of the project environment. Idempotent: existing files are skipped, so it can be
re-run after each batch of downloaded recordings.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

N_SAMPLE, N_RX_CHIP, N_CHIRP = 512, 4, 256
# radar_FFT is stored as complex64, so a correct frame matches to float32 precision
# (measured 1.2e-07); a wrong frame is off by order 1.
MAX_REL_ERR = 1e-5


def decode(raw):
    """(1048576, 4) int16 chip streams -> (512, 256, 16) complex ADC cube.

    Each stream interleaves I and Q; the 16 receivers are the chips in order 3, 0, 1, 2,
    as in valeo's SignalProcessing/rpl.py."""
    frames = [np.reshape(raw[0::2, c] + 1j * raw[1::2, c], (N_SAMPLE, N_RX_CHIP, N_CHIRP),
                         order='F').transpose((0, 2, 1)) for c in range(4)]
    return np.concatenate([frames[3], frames[0], frames[1], frames[2]], axis=2)


def rd_spectrum(cube):
    """valeo's range-Doppler processing: DC removal, Hamming windows, range then Doppler FFT."""
    cube = cube - np.mean(cube, axis=(0, 1))
    w_range = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_SAMPLE) / (N_SAMPLE - 1))
    w_doppler = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_CHIRP) / (N_CHIRP - 1))
    range_fft = np.fft.fft(cube * w_range[:, None, None], N_SAMPLE, axis=0)
    return np.fft.fft(range_fft * w_doppler[None, :, None], N_CHIRP, axis=1)


def relative_error(raw, ref):
    """Max deviation of raw's RD spectrum from the reference, relative to the reference peak."""
    return float(np.abs(rd_spectrum(decode(raw)) - ref).max() / np.abs(ref).max())


def labelled_frames(labels, record):
    """[(numSample, frame index)] of a recording, one entry per frame, in index order."""
    rows = labels[labels.dataset == record][['numSample', 'index']].drop_duplicates()
    return [(int(n), int(i)) for n, i in rows.sort_values('index').itertuples(index=False)]


def missing_frames(frames, out_dir):
    """The (numSample, frame index) pairs whose adc file is not written yet."""
    return [(n, i) for n, i in frames if not os.path.exists(os.path.join(out_dir, f"adc_{n:06d}.npy"))]


def extract_record(record_dir, frames, fft_dir, out_dir):
    """Write the adc file of each given frame of one recording; return (written, mismatched).

    mismatched lists (numSample, frame index, relative error) of frames whose RD spectrum
    did not reproduce radar_FFT; those are not written."""
    from DBReader.DBReader import SyncReader

    db = SyncReader(record_dir, tolerance=20000, silent=True)
    written, mismatched = 0, []
    for num, index in frames:
        sample = db.GetSensorData(index)
        raw = np.stack([sample[f"radar_ch{c}"]["data"] for c in range(4)], axis=1)
        ref = np.load(os.path.join(fft_dir, f"fft_{num:06d}.npy"))
        err = relative_error(raw, ref)
        if err > MAX_REL_ERR:
            mismatched.append((num, index, err))
            print(f"    numSample {num} (frame {index}): RD mismatch {err:.2e}, not written", flush=True)
            continue
        np.save(os.path.join(out_dir, f"adc_{num:06d}.npy"), raw)
        written += 1
    return written, mismatched


def default_radial_dir(name):
    """RADIal's folder `name` under this machine's dataset root (the repo's .env). uv runs
    this script in its own environment, without the project installed, so the repo root
    is put on the import path; detection_vis_backend.paths needs only the standard library."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from detection_vis_backend.paths import dataset_path
    return dataset_path("radial", name)


def count_adc_files(out_dir):
    return len([f for f in os.listdir(out_dir) if f.startswith("adc_") and f.endswith(".npy")])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw-root", default=default_radial_dir("RADIal-raw"),
                   help="folder holding the downloaded RECORD@... recordings")
    p.add_argument("--labeled-root", default=default_radial_dir("RADIal-labeled"),
                   help="ready-to-use release: labels.csv, radar_FFT; radar_ADC is written here")
    args = p.parse_args()

    labels = pd.read_csv(os.path.join(args.labeled_root, "labels.csv"))
    fft_dir = os.path.join(args.labeled_root, "radar_FFT")
    out_dir = os.path.join(args.labeled_root, "radar_ADC")
    os.makedirs(out_dir, exist_ok=True)

    records = sorted(set(labels.dataset))
    present = [r for r in records if os.path.isdir(os.path.join(args.raw_root, r))]
    written = skipped = 0
    mismatched = []
    for k, record in enumerate(present, 1):
        frames = labelled_frames(labels, record)
        todo = missing_frames(frames, out_dir)
        skipped += len(frames) - len(todo)
        print(f"[{k}/{len(present)}] {record}: {len(frames)} labelled, {len(todo)} to write", flush=True)
        if not todo:
            continue
        w, m = extract_record(os.path.join(args.raw_root, record), todo, fft_dir, out_dir)
        written += w
        mismatched += m

    n_frames = labels.numSample.nunique()
    done = count_adc_files(out_dir)
    print(f"\nrecordings: {len(present)} of {len(records)} downloaded")
    print(f"frames: {written} written, {skipped} already present, {len(mismatched)} mismatched")
    print(f"radar_ADC now holds {done} of {n_frames} labelled frames "
          f"({done * N_SAMPLE * N_CHIRP * N_RX_CHIP * 4 * 2 * 2 / 2**30:.2f} GiB)")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
