# How RADDet's stored RAD cubes are computed from ADC

RADDet ships two radar representations per frame. Neither its paper nor its code says how
one is made from the other. This document records how the processing was recovered,
what it is, and why it matters for T-FFTRadNet, whose code builds its own cube from ADC
and ignores the stored one without saying why.

Abbreviations: ADC (analog-to-digital converter samples, the raw radar signal); RAD
(range-azimuth-Doppler, a 3D radar cube); RD (range-Doppler); FFT (fast Fourier
transform); DC (the constant offset of a signal); tx / rx (transmit / receive antenna).

## The two representations

| folder | file | shape | type | frames |
|---|---|---|---|---|
| `RADDet_reform_dataset/ADC/` | `{id:06d}.npy` | (256, 64, 4, 2): range samples × chirps × rx × tx | complex128 | 10,293 |
| `RADDet_reform_dataset/RAD/` | `{id:06d}.npy` | (256, 256, 64): range × azimuth × Doppler | complex64 | 10,158 |

Every frame that has a stored RAD cube also has a label file and an ADC file of the same
number. The 135 extra ADC frames have no labels.

Our RADDet, DAROD and TransRAD runs read the stored cubes. T-FFTRadNet's reference repo
([jgiroux8/T_FFTRadNet](https://github.com/jgiroux8/T_FFTRadNet) at commit 68bc041,
`RADDet/dataset/dataset.py`; "the reference repo" below) reads only `ADC/` and runs its
own FFTs, in all three input modes (ADC, RD and RAD). Its paper and README don't say
why it ignores `RAD/`.

## What the sources say

- **RADDet paper** (`docs/papers/RADDet.pdf`, Sec. III): the raw ADC "have a shape of
  (256, 8, 64)". An "FFT is performed on each dimension". "Zero-padding is used while
  computing FFT on Azimuth dimension", which gives (256, 256, 64). Nothing is said about
  windows, DC removal, antenna order, shifts, axis directions or scale.
- **RADDet repo** (github.com/ZhangAoCanada/RADDet, commit 3ea3975, 2024-01-08): no signal
  processing code. It only loads the finished cubes. The ADC folder was added to the
  release later (README, 2021-06-05).

So the recipe had to be recovered from the data.

## First comparison: the reference repo's cube against the stored one

T-FFTRadNet's RAD mode builds the cube as follows (`dataset.py:107-121`):
1. Reshape the (256, 64, 4, 2) ADC directly to (256, 64, 8).
2. Hamming windows on range and Doppler, then range and Doppler FFTs, with `fftshift` on
   Doppler.
3. Zero-pad the 8 antennas to 256, then angle FFT with `fftshift`.
4. Take the magnitude and transpose to (range, angle, Doppler).

On frames 26, 2103 and 5111, this cube correlated with the magnitude of the stored cube
at 0.10–0.20 as is. Under the best combination of axis flips (range and angle), the
correlation reached 0.48–0.52, and the worst element was still off by about 94 % of the
peak. So taking the magnitude of the stored cube and flipping it does not give the
reference repo's input.

An earlier check compared RD maps summed over the angle axis and found a correlation of
about 0.8. That summing hid the difference, which turned out to lie mostly in the angle
processing.

## The search

`scripts/raddet_dataset_rad_search.py FRAME_ID` builds a candidate cube for every
combination of:

| choice | options |
|---|---|
| antenna order | `reshape` of (rx, tx) as the reference repo does (rx0tx0, rx0tx1, rx1tx0, …), or tx-major (tx0 rx0–3, then tx1 rx0–3) |
| window on range | none, Hann, Hamming |
| window on Doppler | none, Hann, Hamming |
| window on angle | none, Hann, Hamming |
| DC removal | on (mean over range and chirps) or off |
| axis flips | range, angle, Doppler, each reversed or not |

That makes 216 pipelines, each tried with all 8 flip patterns. The FFT steps (Doppler
`fftshift`, angle zero-padding to 256 and `fftshift`) follow the paper and the reference repo.

Each candidate is scored against the stored complex cube by normalised complex
correlation, |⟨a, b⟩| / (‖a‖ ‖b‖). This is 1.0 exactly when the two are equal up to a
single complex scale factor, so it ignores the unknown overall scale while staying
sensitive to phase.

Result on frame 26:

| correlation | antenna order | range window | Doppler window | angle window | DC removed | flips |
|---|---|---|---|---|---|---|
| **0.99979** | tx-major | Hann | none | none | yes | range, angle |
| 0.99512 | tx-major | Hamming | none | none | yes | range, angle |
| 0.84774 | tx-major | Hann | Hamming | none | yes | range, angle |
| 0.80492 | tx-major | Hamming | none | none | no | range, angle |

Every top candidate uses the tx-major order. The reference repo's order did not come close.

## Refinement

`scripts/raddet_dataset_rad_refine.py FRAME_ID [FRAME_ID ...]` starts from the best
candidate and varies the remaining details:

| choice | options |
|---|---|
| Hann form | symmetric (`np.hanning`) or periodic |
| DC mean taken over | range; chirps; range and chirps; all three axes |
| range and angle reversal | plain reversal (`x[::-1]`), or index negation (k → −k mod N, i.e. reversal plus a one-bin roll) |

The best result is a symmetric Hann window, the DC mean over range only (per chirp and
antenna), and plain reversal on both axes. On frames 26, 2103, 5111 and 9000:

| frame | correlation | fitted scale | largest error, relative to the peak |
|---|---|---|---|
| 26 | 0.9999680 | 0.0157 | 0.60 % |
| 2103 | 0.9999709 | 0.0157 | 0.72 % |
| 5111 | 0.9999689 | 0.0157 | 0.65 % |
| 9000 | 0.9999728 | 0.0157 | 0.73 % |

The fitted scale is real-valued and the same on every frame: about 1/64, the number of
chirps.

## Verification

`scripts/raddet_dataset_rad_verify.py N_FRAMES` applies the recipe with a fixed scale of
1/64 to N frames drawn with seed 0 from all 10,158 stored cubes. It flags any frame with
correlation below 0.999 or error above 2 % of the peak.

Run on 2026-11-22 with 500 frames (ids 47–10,284):

| | min | median | max |
|---|---|---|---|
| complex correlation | 0.999960 | 0.999971 | 0.999985 |
| largest error, relative to the peak | 0.42 % | 1.11 % | 1.19 % |

No frame was flagged. The spread across frames is in the fifth decimal of the
correlation, so every frame went through the same processing. The error is higher than in
the refinement (about 1.1 % against 0.6–0.7 %) because 1/64 sits about 0.4 % below the
fitted scale. The remaining residual of about 0.5 % is unexplained. It may come from the
exact window form or the arithmetic precision of RADDet's own pipeline. It does not
change the recipe.

## The recovered recipe

```python
adc = raw.transpose(0, 1, 3, 2).reshape(256, 64, 8)     # tx-major: tx0 rx0-3, tx1 rx0-3
adc = adc - adc.mean(axis=0, keepdims=True)             # DC over range, per chirp and antenna
x = np.fft.fft(adc * np.hanning(256)[:, None, None], axis=0)        # range, Hann
x = np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)                  # Doppler, no window
x = np.fft.fftshift(np.fft.fft(x, 256, axis=2), axes=2)             # azimuth, zero-padded
rad = x.transpose(0, 2, 1)[::-1, ::-1] / 64                         # (range, azimuth, Doppler)
```

## How T-FFTRadNet's processing differs, and what follows

| | stored RAD (RADDet) | T-FFTRadNet RAD mode |
|---|---|---|
| antenna order | tx-major, the physical virtual-array order | rx and tx interleaved by a direct reshape |
| DC removal | mean over range | none (its RD mode removes the mean over range and chirps) |
| range window | Hann | Hamming |
| Doppler window | none | Hamming |
| range and angle axes | reversed | not reversed |
| values | complex | magnitude |

The antenna order matters most. The angle FFT assumes the 8 channels are in array order.
The reference repo's interleaving puts them out of order, so its azimuth spectrum is
scrambled. That is why the reference repo's cube correlates with the stored one at only
about 0.5 even after the best flips.

Neither source says whether the T-FFTRadNet authors knew this. Their code reshapes the
ADC the same way in all modes. In ADC mode the order matters less, because Fourier-Net
learns its own transform weights from that input.

What follows for this project (see the T-FFTRadNet design):
- The T-FFTRadNet reproduction keeps the reference repo's processing, interleaved order included,
  because the published weights and the paper's Table 3 were obtained on it.
- An optional "RAD stored" input mode takes the magnitude of the stored cube, which is
  the same physically ordered cube our RADDet, DAROD and TransRAD runs see. It is trained
  only on request. Its result would show whether the scrambled order costs T-FFTRadNet
  anything.
- Anyone who needs RADDet RAD cubes for the 135 unlabelled frames, or a variant of the
  stored processing, can now compute them from ADC with the recipe above.

## Reproducing this

All three scripts run in the project environment and read the dataset from
`$DATASET_ROOTDIR/RADDet_reform_dataset` (the dataset root is set in the repo's `.env`):

```
uv run python scripts/raddet_dataset_rad_search.py 26
uv run python scripts/raddet_dataset_rad_refine.py 26 2103 5111 9000
uv run python scripts/raddet_dataset_rad_verify.py 500
```

The search takes a few minutes on one frame. The verification takes a few seconds per
frame.
