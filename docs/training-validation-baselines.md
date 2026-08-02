# Published baselines for the 16 wired pairs

Phase 0 of the training-validation campaign. Collected **before** any training run, so
the target is fixed independently of what we measure.

Precedence per pair: **paper → original repo README/checkpoint → "no published baseline"**.

Pass rule (agreed): headline metric within **~20% relative** of the number below, **and**
no degenerate metric (loss decreasing, nothing pinned at 0, predictions non-empty).

Two pairs need a modified rule:

- **RODNet_HGwIv2** has a bracket (83.37 – 85.98) rather than a single target, because
  its exact configuration is not tabulated. Score it as **pass if within 20% of the
  interval** — i.e. ≥ 0.8 × 83.37 ≈ 66.7. Effectively one-sided: still catches a gross
  failure, without inventing a precision the source doesn't support.
- **RECORDNoLstm and RECORDNoLstmMulti on CARRADA** have no published number, so
  RECORD/CARRADA is used as an **upper bound**, not a target — see their rows.

## Caveats that apply to every comparison

- **Splits differ.** This app uses its own split config; papers use their official splits.
  Numbers are not strictly comparable — hence the ballpark rule.
- **Several papers report best-of-N seeds, not a mean.** DAROD is best of 10, RECORD is
  best of 5, RODNet's ablation is single-run. A single run of ours is expected to land
  below their headline.
- **Frameworks differ.** DAROD and RADDet are TensorFlow, RadarPillars is OpenPCDet,
  RECORD is PyTorch Lightning. This repo is plain PyTorch — all are reimplementations.
- **Metrics are not comparable across rows.** AP here means five different things
  (OLS-thresholded, IoU-thresholded, distance-thresholded, BEV mAP). Compare each pair
  only against its own row.

## Baselines

| # | Model | Dataset | Metric | Published | Source |
|---|---|---|---|---|---|
| 1 | FFTRadNet | RADIal | AP / AR @ IoU 0.5 | **96.84 / 82.18** overall (Easy 98.49/91.69, Hard 92.93/64.82) | FFTRadNet Table 2 |
| 2a | RODNet_CDC | CRUW | AP / AR (OLS) | **74.29 / 77.85** | RODNet_v1 Table 2, `CDC` + CRF |
| 2b | RODNet_HG | CRUW | AP / AR (OLS) | **81.10 / 84.26** | RODNet_v1 Table 2, `HG` + CRF |
| 2c | RODNet_HGwI | CRUW | AP / AR (OLS) | **83.76 / 85.62** | RODNet_v1 Table 2, `HGwI` + CRF |
| 2d | RODNet_CDCv2 | CRUW | AP / AR (OLS) | **79.86 / 82.85** | RODNet JSTSP Table IV, Vanilla+CRF+M-Net+TDC |
| 2e | RODNet_HGv2 | CRUW | AP / AR (OLS) | **83.37 / 85.64** | RODNet JSTSP Table IV, Hourglass+CRF+M-Net |
| 2f | RODNet_HGwIv2 | CRUW | AP / AR (OLS) | **83.37 – 85.98** (bracket, no exact row) | see per-variant note |
| 3 | RECORD | CRUW (ROD2021) | AP / AR (OLS) | **72.8 / 82.8** (buffer — confirmed, see note) | RECORD Table I |
| 4 | RECORD | CARRADA | mIoU RA / RD | **34.8 / 58.1** (buffer — confirmed, see note) | RECORD Table V (`RECORD*`) |
| 5 | RECORDNoLstm | CRUW (ROD2021) | AP / AR (OLS) | **59.5 / 75.1** | RECORD Table I (`no lstm, single`) |
| 6 | RECORDNoLstm | CARRADA | mIoU RA / RD | no published number — **upper bound 34.8 / 58.1**, expect ≈18% below | derived from RECORD/CARRADA + CRUW gap |
| 7 | RECORDNoLstmMulti | CRUW (ROD2021) | AP / AR (OLS) | **65.5 / 78.9** | RECORD Table I (`no lstm, multi`) |
| 8 | RECORDNoLstmMulti | CARRADA | mIoU RA / RD | no published number — **upper bound 34.8 / 58.1**, expect ≈10% below | derived from RECORD/CARRADA + CRUW gap |
| 9 | MVRECORD | CARRADA | mIoU RA / RD | **44.5 / 63.2** (buffer); 42.4 / 58.5 (online) | RECORD Table V (`MV-RECORD`) |
| 10 | RADDet | RADDetDataset | AP@0.5 (3D RAD head) | **0.251** (AP@0.1 0.764, AP@0.3 0.563, AP@0.7 0.059) | RADDet Tables I & II |
| 11 | DAROD | CARRADA | mAP@0.5 / @0.3 | **55.83 / 70.68** | DAROD Table I |
| 12 | DAROD | RADDetDataset | mAP@0.5 / @0.3 | **46.57 / 65.56** | DAROD Table I |
| 13 | RAMP_CNN | UWCR | AP / AR | **81.23 / 84.25** overall (ParkingLot 97.38/98.37, Curbside 79.25/84.21, On-road 57.07/64.85) | RAMP-CNN Table III |
| 14 | RadarCrossAttention | CARRADA | AP @2m / @1m | **0.94 / 0.86** overall (Ped 0.91/0.80, Cyc 0.93/0.88, Car 0.97/0.91) | RadarCrossAttention Table I, `Bivar Cross Atten. + offset loss` |
| 15 | PointPillar | ViewOfDelft | mAP entire / corridor | **46.7 / 67.8** (5-frame) | RadarPillars Table I, `PointPillars*` |
| 16 | RadarPillar | ViewOfDelft | mAP entire / corridor | **50.7 / 70.5** (5-frame) | RadarPillars Table I, `RadarPillars (ours)` |

Coverage: **19 of 21 model-dataset combinations have a published number** (16 table rows,
with RODNet expanded to its six variants); 2 are unvalidated, and 1 is bracketed rather
than pointed.

## How the six RODNet variants were mapped

Two papers are involved. **RODNet_v1** (WACV) reports the three non-v2 architectures
directly; the **JSTSP** extension reports one headline (85.98) plus an ablation whose
checkmark columns are hard to read. The mapping was pinned down by cross-matching: the
v1 values **74.29 / 81.10 / 83.76** appear identically in the JSTSP ablation, and their
AR values (77.85 / 84.26 / 85.62) match too — which identifies those rows as
Vanilla+CRF, Hourglass+CRF and Hourglass+CRF+Inception, and so fixes the rest of the
column.

Which JSTSP row applies to each v2 class then follows from `RODNet_model_config.json`:

| class | `mnet_cfg` | `dcn` (TDC) | JSTSP row |
|---|---|---|---|
| CDC / HG / HGwI | absent | absent | v1 architectures — use RODNet_v1 Table 2 |
| CDCv2 | [4,32] | **true** | Vanilla + M-Net + TDC → 79.86 |
| HGv2 | [4,32] | false | Hourglass + M-Net → 83.37 |
| HGwIv2 | [4,32] | false | *no exact row* |

**Two caveats.**

`HGwIv2` is M-Net + Inception **without** TDC, a combination the ablation never
tabulates. The neighbouring rows are Hourglass+M-Net (83.37, no Inception) and
Hourglass+M-Net+TDC+Inception (85.98), so its target is bracketed rather than pointed.
Score it against that range.

The `dcn` setting is **inconsistent across the v2 family** — CDCv2 enables TDC while
HGv2 and HGwIv2 disable it. That may be deliberate or may be a config oversight; either
way the three v2 classes are not a like-for-like family, and their baselines differ
accordingly. Worth confirming before reading anything into their relative scores.

## The two unbaselined CARRADA pairs

The RECORD paper's CARRADA table (Table V) covers only MV-RECORD and single-view
RECORD*; the no-LSTM variants are absent — confirmed, not an oversight in our reading.

They are therefore scored against **RECORD/CARRADA as an upper bound**, not as a target.
Removing the LSTM is precisely what these variants do, so they *should* score lower, and
treating RECORD's number as something to match would penalise a correct implementation.
The expected gap comes from CRUW, where all three are published:

| | AP (CRUW) | vs RECORD |
|---|---|---|
| RECORD | 72.8 | — |
| RECORDNoLstmMulti | 65.5 | −10% |
| RECORDNoLstm | 59.5 | −18% |

So: **must not exceed** RECORD/CARRADA's mIoU, and should land roughly 10% (Multi) and
18% (single) below it. Note RECORDNoLstm at −18% would sit at the edge of a ±20% band if
scored as a direct target — which is exactly why it isn't.

## Conflicting published numbers

**RADDet on its own dataset** has two published values from different protocols:

| source | mAP@0.3 | mAP@0.5 |
|---|---|---|
| RADDet paper (own model, RAD YOLO head) | 56.3 | 25.1 |
| DAROD paper (re-evaluating "RADDet RAD") | 38.32 | 17.13 |

Use RADDet's own figure as the primary target and DAROD's as a lower bound. A result
between the two is not evidence of a bug.

## Protocol notes to carry into the comparison

- **FFTRadNet / RADIal** — 70/15/15 split with no sequence spanning two sets; test split
  is further divided into 'easy' and 'hard'. 100 epochs, Adam, lr 1e-4, ×0.9 every 10.
- **RECORD is the buffer variant — confirmed in code, not assumed.** `RECORD.forward`
  (`network.py:498-501`) loops over the window and calls `self.encoder.__init_hidden__()`
  at `t == 0`, resetting hidden states at the start of every window; the online variant
  never resets. `win_size = 12` matches the paper's buffer window. The buffer column is
  therefore the target for rows 3 and 4, and the online figures (73.5 AP; RA 36.3 /
  RD 61.7) do not apply.
- **RECORD / CRUW** — ROD2021 test annotations are not public, so the authors trained on
  36 sequences and validated on 4 held out of the *training* set. AP/AR over OLS
  thresholds 0.5–0.9.
- **RECORD / CARRADA** — *semantic segmentation* scored by IoU, not detection. Matches
  this repo's CARRADA segmentation path.
- **RAMP_CNN / UWCR** — AP/AR with a size-adaptive distance threshold, CFAR peak
  extraction at threshold 0.2. Nighttime data is test-only and **not** scored
  numerically, so an overall number that includes it is not comparable.
- **RadarCrossAttention / CARRADA** — AP at 2 m and 1 m *distance* thresholds, not IoU.
  The target row assumes bivariate Gaussian labels + centre-offset loss, matching this
  repo's `gauss_type: "Bivar"` and `center_offset: true`.
- **Pillar models / VoD** — VoD has no public test split, so all numbers are on the
  **validation** split. This repo's RadarPillar defaults to `radar_5frames`, so the
  5-frame rows are the right comparison.

## Cross-checks already available

- An existing RadarPillar/VoD run in `trained_models/…Jul-21-2026_08:56:49` peaked at
  **val mAP 50.7**, matching RadarPillars' published 50.7 for the 5-frame entire-area
  setting. Encouraging, though it predates the current code.
- That same run reports `mAR` and `mIoU` as exactly `0.0000` for all 80 epochs while mAP
  climbs — a metric pinned at zero across an entire run is the degeneracy signature the
  campaign is meant to catch. Worth resolving before trusting any pillar comparison.
