"""Refine the ADC -> stored RAD pipeline found by raddet_dataset_rad_search.py.

Starting from the best combination of the search (tx-major antennas, Hann on range only,
DC removed, range and angle flipped), this compares on the first given frame:

    Hann form       symmetric (np.hanning) or periodic
    DC axes         mean over range; chirps; range and chirps; or all three axes
    range / angle   reversed (x[::-1]) or index-negated (k -> -k mod N)

then applies the best one to every given frame and prints the complex correlation, the
fitted complex scale, the largest element error relative to the peak, and the correlation
of the magnitudes. Result on frames 26, 2103, 5111, 9000: symmetric Hann, DC mean over
range only, both axes reversed; correlation 0.99997, scale 0.0157 (real, about 1/64),
error 0.6-0.7 % of the peak.

    uv run python scripts/raddet_dataset_rad_refine.py FRAME_ID [FRAME_ID ...]
"""
import itertools, sys
import numpy as np

from detection_vis_backend.paths import dataset_path

root = dataset_path("RADDet_reform_dataset")

def ncorr(a, b):
    return float(abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))

def neg(x, axis):
    """Index k -> -k mod N, i.e. the spectrum of the conjugate-time signal."""
    return np.roll(np.flip(x, axis), 1, axis)

wins = {"hann sym": np.hanning(256),
        "hann periodic": 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(256) / 256)}

def build(raw, wname, dc_axes, rflip, aflip):
    adc = raw.transpose(0, 1, 3, 2).reshape(256, 64, 8)
    adc = adc - adc.mean(axis=dc_axes, keepdims=True)
    x = np.fft.fft(adc * wins[wname][:, None, None], axis=0)
    x = np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)
    x = np.fft.fftshift(np.fft.fft(x, 256, axis=2), axes=2).transpose(0, 2, 1)
    x = {"flip": x[::-1], "neg": neg(x, 0)}[rflip]
    x = {"flip": x[:, ::-1], "neg": neg(x, 1)}[aflip]
    return x

ids = [int(a) for a in sys.argv[1:]]
raw0 = np.load(f"{root}/ADC/{ids[0]:06d}.npy"); ref0 = np.load(f"{root}/RAD/{ids[0]:06d}.npy").astype(complex)
res = []
for w, dca, rf, af in itertools.product(wins, [(0, 1), (0,), (1,), (0, 1, 2)], ["flip", "neg"], ["flip", "neg"]):
    res.append((ncorr(build(raw0, w, dca, rf, af), ref0), w, dca, rf, af))
res.sort(reverse=True)
for r in res[:5]:
    print(f"{r[0]:.7f} window={r[1]} dc_axes={r[2]} range={r[3]} angle={r[4]}")

_, w, dca, rf, af = res[0]
for fid in ids:
    raw = np.load(f"{root}/ADC/{fid:06d}.npy"); ref = np.load(f"{root}/RAD/{fid:06d}.npy").astype(complex)
    x = build(raw, w, dca, rf, af)
    k = np.vdot(x, ref) / np.vdot(x, x)
    err = np.abs(k * x - ref).max() / np.abs(ref).max()
    print(f"frame {fid}: ncorr {ncorr(x, ref):.7f}  scale {k:.4g}  max rel err {err:.2e}  mag ncorr {ncorr(np.abs(x), np.abs(ref)):.7f}")
