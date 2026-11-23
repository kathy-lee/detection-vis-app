"""Check the recovered ADC -> stored RAD pipeline of RADDet on a random sample of frames.

The pipeline, recovered by raddet_dataset_rad_search.py and raddet_dataset_rad_refine.py
(the RADDet paper and repo do not give it):

    1. antennas tx-major: (256, 64, 4, 2) -> transpose(0, 1, 3, 2) -> (256, 64, 8)
    2. DC removal: subtract the mean over range samples, per chirp and antenna
    3. range FFT with a symmetric Hann window; no window on Doppler or angle
    4. Doppler FFT, fftshift
    5. angle FFT zero-padded to 256, fftshift
    6. transpose to (range, angle, Doppler), reverse range and angle, scale by 1/64

For N frames drawn with a fixed seed from all stored cubes, it prints the min / median /
max of the complex correlation with the stored cube and of the largest element error
relative to the peak, and lists frames with correlation below 0.999 or error above 2 %.
Result 2026-11-22 on 500 frames (ids 47-10284): correlation 0.99996-0.99999, error
0.42-1.2 %, no outliers. Most of the error is the scale: the fitted one is 0.0157, about
0.4 % above 1/64.

    uv run python scripts/raddet_dataset_rad_verify.py N_FRAMES
"""
import os, sys
import numpy as np

root = "/home/kangle/dataset/RADDet_reform_dataset"
n = int(sys.argv[1])
ids = sorted(int(f[:6]) for f in os.listdir(f"{root}/RAD"))
rng = np.random.default_rng(0)
sample = sorted(rng.choice(ids, size=n, replace=False))
win = np.hanning(256)[:, None, None]

def build(raw):
    adc = raw.transpose(0, 1, 3, 2).reshape(256, 64, 8)
    adc = adc - adc.mean(axis=0, keepdims=True)
    x = np.fft.fft(adc * win, axis=0)
    x = np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)
    x = np.fft.fftshift(np.fft.fft(x, 256, axis=2), axes=2).transpose(0, 2, 1)
    return x[::-1, ::-1] / 64

rows = []
for fid in sample:
    x = build(np.load(f"{root}/ADC/{fid:06d}.npy"))
    ref = np.load(f"{root}/RAD/{fid:06d}.npy").astype(complex)
    c = abs(np.vdot(x, ref)) / np.sqrt(np.vdot(x, x).real * np.vdot(ref, ref).real)
    err = np.abs(x - ref).max() / np.abs(ref).max()
    rows.append((fid, c, err))
a = np.array(rows)
print(f"frames {n} (ids {int(a[:,0].min())}-{int(a[:,0].max())})")
for name, col in (("ncorr", 1), ("max rel err (fixed 1/64 scale)", 2)):
    v = a[:, col]
    print(f"{name}: min {v.min():.6f}  median {np.median(v):.6f}  max {v.max():.6f}")
bad = a[(a[:, 1] < 0.999) | (a[:, 2] > 0.02)]
print("outliers (ncorr<0.999 or err>2%):", len(bad))
for fid, c, e in bad[:20]:
    print(f"  {int(fid):06d} ncorr {c:.5f} err {e:.3f}")
