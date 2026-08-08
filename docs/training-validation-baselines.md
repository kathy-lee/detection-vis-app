# Published baselines for the 16 wired pairs (24 runs)

Phase 0 of the training-validation campaign. Collected **before** any training run, so
the target is fixed independently of what we measure.

Precedence per pair: **paper → original repo README/checkpoint → "no published baseline"**.

Pass rule (agreed): headline metric within **~20% relative** of the number below, **and**
no degenerate metric (loss decreasing, nothing pinned at 0, predictions non-empty).

Some entries need a modified rule:

- **RODNet_HGwIv2** has a bracket (83.37 – 85.98) rather than a single target, because
  its exact configuration is not tabulated. Score it as **pass if within 20% of the
  interval** — i.e. ≥ 0.8 × 83.37 ≈ 66.7. Effectively one-sided: still catches a gross
  failure, without inventing a precision the source doesn't support.
- **RECORDNoLstm and RECORDNoLstmMulti on CARRADA**, in **both views**, have no published
  number, so the matching RECORD/CARRADA view is used as an **upper bound**, not a
  target — see their rows. That is four entries, not two.

## One pair is not always one run

The 16 wired (model, dataset) pairs are **24 runs**, because two pairs expand:

- **RODNet / CRUW → 6.** `RODNet_CDC`, `CDCv2`, `HG`, `HGv2`, `HGwI`, `HGwIv2` are six
  distinct classes in `RODNet_model_config.json`, each with its own baseline (rows 2a–2f).
- **RECORD-family / CARRADA → 6.** `RECORD`, `RECORDNoLstm` and `RECORDNoLstmMulti` are
  **single-view** on CARRADA: `RECORD_model_config.json` keys them under `CARRADA/RA` and
  `CARRADA/RD` as separate config blocks, and the UI's `find_default_configs` selects by
  `feature[0]`. RA and RD are therefore separately *trained* models, not one model scored
  two ways — which is also why the RECORD paper reports two numbers per row. On CRUW the
  same three models take RA only, so no expansion there.

The remaining 14 pairs are one run each. Rows below are indexed per run.

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
| 4a | RECORD | CARRADA (RA) | mIoU | **34.8** (buffer — confirmed, see note) | RECORD Table V (`RECORD*`) |
| 4b | RECORD | CARRADA (RD) | mIoU | **58.1** (buffer — confirmed, see note) | RECORD Table V (`RECORD*`) |
| 5 | RECORDNoLstm | CRUW (ROD2021) | AP / AR (OLS) | **59.5 / 75.1** | RECORD Table I (`no lstm, single`) |
| 6a | RECORDNoLstm | CARRADA (RA) | mIoU | no published number — **upper bound 34.8**, expect ≈18% below | derived from row 4a + CRUW gap |
| 6b | RECORDNoLstm | CARRADA (RD) | mIoU | no published number — **upper bound 58.1**, expect ≈18% below | derived from row 4b + CRUW gap |
| 7 | RECORDNoLstmMulti | CRUW (ROD2021) | AP / AR (OLS) | **65.5 / 78.9** | RECORD Table I (`no lstm, multi`) |
| 8a | RECORDNoLstmMulti | CARRADA (RA) | mIoU | no published number — **upper bound 34.8**, expect ≈10% below | derived from row 4a + CRUW gap |
| 8b | RECORDNoLstmMulti | CARRADA (RD) | mIoU | no published number — **upper bound 58.1**, expect ≈10% below | derived from row 4b + CRUW gap |
| 9 | MVRECORD | CARRADA | mIoU RA / RD | **44.5 / 63.2** (buffer); 42.4 / 58.5 (online) | RECORD Table V (`MV-RECORD`) |
| 10 | RADDet | RADDetDataset | AP@0.5 (3D RAD head) | **0.251** (AP@0.1 0.764, AP@0.3 0.563, AP@0.7 0.059) | RADDet Tables I & II |
| 11 | DAROD | CARRADA | mAP@0.5 / @0.3 | **55.83 / 70.68** | DAROD Table I |
| 12 | DAROD | RADDetDataset | mAP@0.5 / @0.3 | **46.57 / 65.56** | DAROD Table I |
| 13 | RAMP_CNN | UWCR | AP / AR | **81.23 / 84.25** overall (ParkingLot 97.38/98.37, Curbside 79.25/84.21, On-road 57.07/64.85) | RAMP-CNN Table III |
| 14 | RadarCrossAttention | CARRADA | AP @2m / @1m | **0.94 / 0.86** overall (Ped 0.91/0.80, Cyc 0.93/0.88, Car 0.97/0.91) | RadarCrossAttention Table I, `Bivar Cross Atten. + offset loss` |
| 15 | PointPillar | ViewOfDelft | mAP entire / corridor | **46.7 / 67.8** (5-frame) | RadarPillars Table I, `PointPillars*` |
| 16 | RadarPillar | ViewOfDelft | mAP entire / corridor | **50.7 / 70.5** (5-frame) | RadarPillars Table I, `RadarPillars (ours)` |

Coverage: **20 of the 24 runs have a published number**; 4 are unbaselined
(`RECORDNoLstm` and `RECORDNoLstmMulti` on CARRADA, each in both views) and 1 of the 20
is bracketed rather than pointed (`RODNet_HGwIv2`).


## Measured results (Phase 2/3)

Protocol: official split, the paper's own schedule, repo default configs, **three seeds
(3/7/11)** per pair, final-epoch weights (which is also what the reference's `eval.py`
evaluates -- it restores `latest_checkpoint`, not a best-val checkpoint).

| # | pair | ours (mean ± sd) | published | reference on OUR pipeline | verdict |
|---|------|------------------|-----------|---------------------------|---------|
| 11 | DAROD / CARRADA | **0.5642 ± 0.0140** | 0.5583 | 0.5581 | **match** (+1.1%, 0.4 sd) |
| 12 | DAROD / RADDetDataset | **0.4463 ± 0.0186** | 0.4657 | 0.4478 | **match** (−0.3% vs reference, 0.1 sd) |

Both are mAP@0.5. The other 22 runs are not yet measured.

**Read RADDet against 0.4478, not 0.4657.** The reference implementation itself scores
only 0.4414 on our copy of RADDet (0.4478 through our loader/evaluator), 5.2% under its
own published figure. That is about 1 sd of the seed spread, so it may be seed noise --
but the builder that produced the published number is not recoverable from that repo
(its `config_raddet.json` pins dataset_version 1.0.0 while `raddet.py` declares 1.1.0,
so the published config cannot load the published builder). CARRADA has no such problem:
there the reference reproduces the paper at both thresholds (0.5578/0.7044 against
0.5583/0.7068).

### Run-to-run variance is large enough to change verdicts

Measured over three seeds with everything else fixed:

| pair | sd | relative |
|------|-----|----------|
| DAROD / CARRADA | 0.0140 | 2.5% |
| DAROD / RADDetDataset | 0.0186 | 4.2% |

**A single run cannot resolve a difference below roughly 5%.** Any verdict finer than
that needs seeds; `scripts/smoke_sweep.py --seed N` exists for this. Two cautionary
examples from this campaign, both of which were reported as findings before being
re-measured:

- A single CARRADA run scored 0.5556 against a published 0.5583 and was called a match
  "within 0.5%". Three seeds put the mean at 0.5377 ± 0.0162 at that point in the work --
  the 0.5% was a favourable draw, not precision.
- A single RADDet run scored 0.4320 and its 7.2% shortfall was treated as the thing to
  explain. Three seeds gave 0.4060 ± 0.0227, i.e. a 12.8% shortfall; the first run had
  been the best of the three.

### How these were validated

Matching a published number is weak evidence on its own, because a wrong pipeline can
still train to a plausible score. The stronger check, and the one that found the real
defects, is to run the reference implementation's **trained weights through our own
loader, decoder and evaluator** and confirm we reproduce its score:

| pair | reference's own `eval.py` | same weights, our pipeline |
|------|---------------------------|----------------------------|
| CARRADA | 0.5578 | 0.5581 |
| RADDetDataset | 0.4414 | 0.4478 |

That isolates data/eval from training in one shot and needs no training run. On CARRADA
it read 0.1025 before the gt-box defect was fixed; on RADDet 0.0031 before the input
normalization defect. **Run it first on any pair that has a reference implementation.**

Reference checkpoints, per-class results, the saved comparison tensors and the harness
scripts are preserved outside the repo at `/home/kangle/dataset/darod_reference/`
(see its README). Twenty defects were found and fixed to reach the numbers above; the
commit messages on `feat/validate-baselines` carry the details.

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

## The four unbaselined CARRADA runs

The RECORD paper's CARRADA table (Table V) covers only MV-RECORD and single-view
RECORD*; the no-LSTM variants are absent — confirmed, not an oversight in our reading.
Because CARRADA is single-view, this is four runs (rows 6a/6b/8a/8b), not two.

They are therefore scored against **the matching RECORD/CARRADA view as an upper bound**
(RA against 34.8, RD against 58.1), not as a target.
Removing the LSTM is precisely what these variants do, so they *should* score lower, and
treating RECORD's number as something to match would penalise a correct implementation.
The expected gap comes from CRUW, where all three are published:

| | AP (CRUW) | vs RECORD |
|---|---|---|
| RECORD | 72.8 | — |
| RECORDNoLstmMulti | 65.5 | −10% |
| RECORDNoLstm | 59.5 | −18% |

So per view: **must not exceed** the row-4 mIoU for that view, and should land roughly
10% (Multi) and 18% (single) below it. Note RECORDNoLstm at −18% would sit at the edge of
a ±20% band if scored as a direct target — which is exactly why it isn't.

The gap is measured on CRUW, where the models are stacked-view rather than single-view,
so carrying it to CARRADA assumes the LSTM contributes similarly in both settings. It is
an expectation for triage, not a pass/fail threshold; only the upper bound is binding.

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
  climbs. This is **not** a degeneracy: `RadarPillar_evaluation` returns
  `{'mAP': mAP, 'mAR': 0.0, 'mIoU': 0.0}` (`evaluate.py:1930`) — the two zeros are
  hardcoded constants, because the VoD devkit scores BEV/3D mAP only. Nothing to fix.

## Which metrics each evaluator actually produces

A metric that is structurally absent reads identically to a metric that collapsed to
zero, so any zero-check must know which is which. From the `result = {...}` line of each
evaluator in `evaluate.py`:

| Evaluator | Real | Hardcoded `0` |
|---|---|---|
| `FFTRadNet_evaluation` | mAP, mAR, mIoU | — |
| `RODNet_evaluation` | mAP, mAR | mIoU |
| `RECORD_CRUW_evaluation` | mAP, mAR | mIoU |
| `RECORD_CARRADA_evaluation` | mAP, mAR, mIoU | — |
| `MVRECORD_CARRADA_evaluation` | mAP, mAR, mIoU | — |
| `RADDet_evaluation` | mAP | mAR, mIoU |
| `DAROD_evaluation` | mAP, mAR | mIoU |
| `RAMP_CNN_evaluation` | mAP, mAR | mIoU |
| `RadarCrossAttention_evaluation` | mAP | mAR, mIoU |
| `RadarPillar_evaluation` (also PointPillar) | mAP | mAR, mIoU |

So a zero in a "hardcoded" cell is meaningless, and only a zero in a "real" cell is
evidence of anything.
