"""Search which FFT pipeline turns RADDet raw ADC into the stored RAD cube.

RADDet ships, per frame, raw ADC (ADC/{id:06d}.npy, (256, 64, 4, 2) complex) and a
range-azimuth-Doppler cube (RAD/{id:06d}.npy, (256, 256, 64) complex64). The RADDet paper
says only that an FFT is run on each dimension with the azimuth axis zero-padded to 256,
and the RADDet repo has no processing code. This script recovers the missing details by
trying every combination of:

    antenna order   reshape(rx, tx) as T-FFTRadNet does, or tx-major (tx0 rx0-3, tx1 rx0-3)
    window          none, Hann or Hamming, separately on range, Doppler and angle
    DC removal      on or off (mean over range and chirps)
    axis flips      each of range, angle and Doppler flipped or not

and ranking them by normalized complex correlation with the stored cube (1.0 means equal
up to one complex scale). On frame 26 the best is tx-major, Hann on range only, DC
removed, range and angle flipped (0.99979); raddet_dataset_rad_refine.py narrows it
further and raddet_dataset_rad_verify.py checks it on many frames.

    uv run python scripts/raddet_dataset_rad_search.py FRAME_ID
"""
import itertools, sys
import numpy as np

root = "/home/kangle/dataset/RADDet_reform_dataset"
fid = int(sys.argv[1])
raw = np.load(f"{root}/ADC/{fid:06d}.npy")          # (256, 64, 4, 2)
ref = np.load(f"{root}/RAD/{fid:06d}.npy").astype(complex)   # (256, 256, 64)
print("raw", raw.shape, raw.dtype)

def ncorr(a, b):
    """|<a,b>| / (|a||b|): 1.0 iff equal up to one complex scale."""
    return float(abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))

def win(kind, n):
    if kind == "none": return np.ones(n)
    if kind == "hann": return np.hanning(n)
    return np.hamming(n)

orders = {"reshape(rx,tx)": raw.reshape(256, 64, 8),
          "tx-major": raw.transpose(0, 1, 3, 2).reshape(256, 64, 8)}
results = []
for (oname, adc), wr, wd, wa, dc in itertools.product(orders.items(), ["none", "hann", "hamming"],
                                                     ["none", "hann", "hamming"], ["none", "hann", "hamming"], [False, True]):
    x = adc - adc.mean(axis=(0, 1)) if dc else adc
    x = x * win(wr, 256)[:, None, None] * win(wd, 64)[None, :, None] * win(wa, 8)[None, None, :]
    x = np.fft.fft(x, axis=0)
    x = np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)
    x = np.fft.fftshift(np.fft.fft(x, 256, axis=2), axes=2)
    x = x.transpose(0, 2, 1)                              # (range, angle, doppler)
    best = (0, None)
    for fr, fa, fd in itertools.product([0, 1], repeat=3):
        y = x[::-1] if fr else x
        y = y[:, ::-1] if fa else y
        y = y[:, :, ::-1] if fd else y
        c = ncorr(y, ref)
        if c > best[0]: best = (c, (fr, fa, fd))
    results.append((best[0], oname, wr, wd, wa, dc, best[1]))
results.sort(reverse=True)
for r in results[:8]:
    print(f"{r[0]:.5f}  order={r[1]:15s} win range={r[2]:7s} doppler={r[3]:7s} angle={r[4]:7s} dc={r[5]} flips(r,a,d)={r[6]}")
