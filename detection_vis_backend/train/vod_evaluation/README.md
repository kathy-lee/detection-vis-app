# VoD devkit evaluation (copied third-party code)

Copied from https://github.com/tudelft-iv/view-of-delft-dataset
(`vod/evaluation/`), commit `6f02afad6a0f6f77815c51d8284201cba9fcf4da`,
license Apache-2.0 (see the upstream LICENSE.txt).

Copied because the devkit ships no pip packaging (conda notebook env only).
Files are verbatim except `evaluate.py`: package-relative import of
`evaluation_common`, `get_frame_list_from_folder` inlined from
`vod/common/file_handling.py`, upstream `__main__` demo removed.
Keeping the algorithm files verbatim keeps the official metric trustworthy
and diffable against upstream; the price is that each evaluation also runs
the 2D-bbox/AOS and BEV passes we do not report (~3x the minimum runtime).

This is the official View-of-Delft metric: KITTI-style 3D AP (R11, rotated
IoU 0.5/0.25/0.25 for Car/Ped/Cyclist) over the entire annotated area and
the driving corridor (`roi`).
