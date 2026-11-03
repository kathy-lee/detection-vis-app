#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "db-reader @ git+https://github.com/valeoai/RADIal@02f52fc13ee41feacec82fd8e05f5b1f0bf4ce26#subdirectory=DBReader",
#     "cantools",
#     "opencv-python-headless",  # imported by DBReader's camera reader, undeclared in its setup.py
#     "gdown",
#     "numpy",
#     "pandas",
# ]
# ///
"""Download the raw RADIal recordings one at a time and keep only their benchmark ADC.

The raw release is about 96 zips on Google Drive, several GB each, far more than the
64.5 GiB of ADC the benchmark frames need. For each recording that still owes frames
this downloads its zip, unpacks it into its own staging folder and deletes the zip,
extracts the frames with extract_radial_adc.extract_record (the same RD check against
radar_FFT), and deletes the staging folder. The staging folder is kept when a frame fails
the check or anything errors, for inspection, and the next recording is tried; a zip
that failed to download or unpack is kept too.

    uv run scripts/fetch_radial_raw.py [--raw-root DIR] [--labeled-root DIR]

Recordings whose benchmark frames are all in radar_ADC already, and recordings with no
benchmark frames, are not downloaded, so a re-run resumes where the last one stopped.
Run it with `uv run` as shown: `python scripts/...` or `.venv/bin/python scripts/...` runs
it in the project environment, which has neither DBReader nor gdown.
"""
import argparse
import glob
import os
import shutil
import sys
import zipfile

import pandas as pd

from extract_radial_adc import count_adc_files, extract_record, labelled_frames, missing_frames

DRIVE_FOLDER_ID = "1ur9Dgr7fjE4v0QbdBKXLtk5t8ey__3wn"
# Room for the largest zip plus its unpacked recording; below this the loop stops.
MIN_FREE_GIB = 30


def pending(listing, labels, out_dir):
    """[(record, drive id, frames still to write)] for zips that owe benchmark frames."""
    records = set(labels.dataset)
    todo = []
    for name, file_id in listing:
        record = name[:-len(".zip")] if name.endswith(".zip") else None
        if record not in records:
            continue
        frames = missing_frames(labelled_frames(labels, record), out_dir)
        if frames:
            todo.append((record, file_id, frames))
    return todo


def recording_dir(unpack_dir, record):
    """The folder under unpack_dir holding the recording's radar streams, nested or not."""
    hits = glob.glob(os.path.join(glob.escape(unpack_dir), "**", f"{glob.escape(record)}_radar_ch0.bin"),
                     recursive=True)
    if len(hits) != 1:
        raise FileNotFoundError(f"{record}_radar_ch0.bin not found once under {unpack_dir} ({len(hits)} hits)")
    return os.path.dirname(hits[0])


def free_gib(path):
    return shutil.disk_usage(path).free / 2**30


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw-root", default="/home/kangle/dataset/radial/RADIal-raw",
                   help="where each zip is downloaded and unpacked, then deleted")
    p.add_argument("--labeled-root", default="/home/kangle/dataset/radial/RADIal-labeled",
                   help="ready-to-use release: labels.csv, radar_FFT; radar_ADC is written here")
    args = p.parse_args()

    import gdown

    labels = pd.read_csv(os.path.join(args.labeled_root, "labels.csv"))
    fft_dir = os.path.join(args.labeled_root, "radar_FFT")
    out_dir = os.path.join(args.labeled_root, "radar_ADC")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.raw_root, exist_ok=True)

    listing = [(f.path, f.id) for f in
               gdown.download_folder(id=DRIVE_FOLDER_ID, skip_download=True, quiet=False)]
    todo = pending(listing, labels, out_dir)
    print(f"{len(listing)} files on Drive, {len(todo)} recordings still owe benchmark frames", flush=True)

    failed = []
    for k, (record, file_id, frames) in enumerate(todo, 1):
        if free_gib(args.raw_root) < MIN_FREE_GIB:
            print(f"only {free_gib(args.raw_root):.0f} GiB free under {args.raw_root}, "
                  f"need {MIN_FREE_GIB}; stopping", flush=True)
            failed.append((record, "disk full"))
            break
        print(f"\n[{k}/{len(todo)}] {record}: downloading for {len(frames)} benchmark frames", flush=True)
        zip_path = os.path.join(args.raw_root, f"{record}.zip")
        unpack_dir = os.path.join(args.raw_root, f".unpack_{record}")
        try:
            if gdown.download(id=file_id, output=zip_path, quiet=False) is None:
                raise RuntimeError("download failed (Drive quota or access)")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(unpack_dir)
            os.remove(zip_path)  # unpacked copy is all extraction needs; frees up to ~10 GB at once
            written, mismatched = extract_record(recording_dir(unpack_dir, record), frames, fft_dir, out_dir)
        except Exception as e:  # one bad recording must not stop the other ninety
            print(f"    {record}: {type(e).__name__}: {e}; files kept", flush=True)
            failed.append((record, str(e)))
            continue
        if mismatched:
            print(f"    {record}: {written} benchmark frames extracted, {len(mismatched)} mismatched; "
                  f"files kept in {unpack_dir}", flush=True)
            failed.append((record, f"{len(mismatched)} mismatched"))
            continue
        shutil.rmtree(unpack_dir)
        print(f"    {record}: {written} benchmark frames extracted\n", flush=True)

    print(f"\n===== radar_ADC holds {count_adc_files(out_dir)} .npy files "
          f"of {labels.numSample.nunique()} benchmark frames =====")
    # for record, why in failed:
    #     print(f"  not finished: {record} ({why})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
