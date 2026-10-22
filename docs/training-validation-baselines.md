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
| 17 | TransRAD | RADDetDataset | mAP@0.3 / @0.5 (3D RAD) | **60.16 / 37.77** (authors' checkpoint through their own Test.py; paper Table I 61.90 / 38.76) | TransRAD Table I; `Test_results/AP_Results.txt` in the reference repo |
| 18 | T_RODNet | CRUW | AP / AR (OLS) | **83.27 / 86.98** (pedestrian 82.19/85.41, cyclist 82.28/84.30, car 86.22/92.53) — on four held-out LABELLED training sequences the authors name, so, unlike rows 2–7, locally reproducible | T-RODNet Table I; split in the reference README |

Coverage: **22 of the 26 runs have a published number**; 4 are unbaselined
(`RECORDNoLstm` and `RECORDNoLstmMulti` on CARRADA, each in both views) and 1 of the 20
is bracketed rather than pointed (`RODNet_HGwIv2`).

### Found in Phase 2: no CRUW baseline is locally reproducible

Reading the source papers during the RODNet_CDC run showed that every CRUW number in
the table above was measured on data we do not hold:

- **Rows 2a–2f** (RODNet_v1 Table 2, JSTSP Table IV) were trained on the **full,
  unreleased CRUW corpus** — ~396K frames, ~260K objects, 92% used for training —
  roughly 6–8× the ROD2021 public release (40 annotated sequences) that this repo has.
  The "+CRF" in the source column is not post-processing: Table 2 is an ablation over
  **supervision sources**, camera-only (CDC: 52.62 AP) vs camera-radar-fusion
  auto-annotation (74.29) — a 22-point swing from label quality alone, and both are a
  different annotation regime from ROD2021's shipped labels. Evaluation used the
  authors' private split of the full corpus.
- **Rows 3, 5, 7** (RECORD Table I) were scored by the **ROD2021 challenge server** on
  its 10 hidden test sequences, whose annotations were never released (the paper trains
  36 / validates 4 and submits predictions; Table I's caption says "test set of the
  ROD2021 challenge"). Table I contains no RODNet rows, so there is also no published
  RODNet number on a recoverable split.

Consequences: the RODNet 74.29 and RECORD 72.8 are not comparable **to each other**
(different training data, labels, and test sets), and neither is reachable by local
evaluation. CRUW pairs therefore get **relative validation** — the cross-model ladder
on our fixed split (see the RODNet_CDC closure below) — instead of the absolute ~20%
pass rule. CARRADA, RADDetDataset, RADIal and ViewOfDelft are unaffected: their test
annotations ship with the data and their reference splits are recoverable.


## Measured results (Phase 2/3)

Protocol: official split, the paper's own schedule, repo default configs, **three seeds
(3/7/11)** per pair, final-epoch weights (which is also what the reference's `eval.py`
evaluates -- it restores `latest_checkpoint`, not a best-val checkpoint).

| # | pair | ours (mean ± sd) | published | reference on OUR pipeline | verdict | run dir under `$MODEL_ROOTDIR` |
|---|------|------------------|-----------|---------------------------|---------|--------------------------------|
| 1 | FFTRadNet / RADIal | **95.55 / 86.07** test AP/AR @ IoU 0.5 (1 seed, official split, final weights; freespace mIoU **77.23**, range err 0.122 m, angle err 0.112°) | 96.84 / 82.18 (mIoU 74.0, 0.11 m, 0.17°) | not run (the paper's own split and data are intact — see closure) | **reproduced** (AP −1.3%; AR, mIoU and angle error all *above* published) | `FFTRadNet_RADIal_Sep-08-2026_22:38:45` |
| 2a | RODNet_CDC / CRUW | **38.5 / 44.6** test AP/AR (1 seed, our split; best-val epoch 7) | 74.29 / 77.85 | — | **inconclusive-by-protocol** (see closure section) | `RODNet_CDC_CRUW_Aug-20-2026_16:39:39` |
| 3 | RECORD / CRUW | **41.6 / 48.2** test AP/AR (1 seed, our split; best val 55.6 @ epoch 17) | 72.8 / 82.8 | — | ladder rung 1 — see closure section | `RECORD_CRUW_Aug-23-2026_06:58:13` |
| 4b | RECORD / CARRADA / RD | **53.15** test mIoU (1 seed, official split) | 58.10 | **53.19** | **match** (−0.08% vs reference) | `RECORD_CARRADA_Aug-23-2026_20:58:15`, **epoch 71** (best val mIoU 63.86) |
| 4a | RECORD / CARRADA / RA | **32.74** test mIoU (1 seed, official split) | 34.80 | **33.08** | **match** (−1.0% vs reference) | `RECORD_CARRADA_Aug-25-2026_17:28:24` |
| 9 | MVRECORD / CARRADA | **54.83 RD / 38.89 RA** test mIoU (1 seed, official split) | 63.2 RD / 44.5 RA | **56.35 RD / 38.09 RA** | **match** (−2.7% RD, +2.1% RA vs reference) | `MVRECORD_CARRADA_Aug-30-2026_21:26:56` |
| 10 | RADDet / RADDetDataset | **30.37** test AP@0.5 (1 seed, official split; @0.1 77.76 / @0.3 60.42 / @0.7 6.94) | **25.1** AP@0.5 (@0.1 76.4 / @0.3 56.3 / @0.7 5.9) | not run (not needed — see below) | **reproduced, exceeds published** (+21% @0.5, above at all four thresholds) | `RADDet_RADDetDataset_Sep-01-2026_20:49:42` |
| 11 | DAROD / CARRADA | **56.42 ± 1.40** test mAP@0.5 (3 seeds, official split, final weights) | 55.83 | 55.81 | **match** (+1.1%, 0.4 sd) | `DAROD_CARRADA_Aug-08-2026_08:09:24` + `…_08:47:05` + `…_09:23:58` (3 seeds) |
| 12 | DAROD / RADDetDataset | **44.63 ± 1.86** test mAP@0.5 (3 seeds, official split, final weights) | 46.57 | 44.78 | **match** (−0.3% vs reference, 0.1 sd) | `DAROD_RADDetDataset_Aug-07-2026_22:47:24` + `…_23:42:36` + `DAROD_RADDetDataset_Aug-08-2026_00:38:34` (3 seeds) |
| 13 | RAMP_CNN / UWCR | **13.09 / 29.31** test AP/AR (**our** split, epoch 28; final weights 6.07 / 15.74). Unnormalized control: **17.86 / 34.22** — higher | 81.23 / 84.25 | impossible (no released weights, no reference-side data) | **not reproducible — data only partly released**; regression baseline only | `RAMP_CNN_UWCR_Sep-05-2026_20:12:04` ; unnormalized control `RAMP_CNN_UWCR_Sep-02-2026_21:57:47` |
| 14 | RadarCrossAttention / CARRADA | **88.8 / 78.4** test AP@2m/@1m on the reference's frame-random 70/15/15 protocol; **42.7 / 34.3** on the official sequence split (1 seed each, final weights) | 94.0 / 86.0 | impossible (shipped reference cannot train or evaluate as-is) | **match on the published protocol** (−5.5% / −8.8%); the frame-leaky split inflates the number 2.1× over held-out sequences | `RadarCrossAttention_CARRADA_Sep-08-2026_07:37:29` (frame-random) + `RadarCrossAttention_CARRADA_Sep-07-2026_21:50:01` (official split) |
| 15 | PointPillar / ViewOfDelft | **49.41** val mAP entire-area, final weights (best 52.27 @ epoch 64; 1 seed, official split, 5-frame radar, N×7 point features) | 46.7 | not run (paper's own PointPillars row is the anchor) | **reproduced, exceeds published** (+5.8% final, +12% best) | `RadarPillar_ViewOfDelft_Jul-19-2026_21:44:32` (the base model shares the radarpillar.py path, hence the dir name) |
| 16 | RadarPillar / ViewOfDelft | **50.48** val mAP entire-area, final weights (best 50.72 @ epoch 75; epochs 70–79 all within 50.2–50.7) | 50.7 | 52.56 (fthbng77/RadarPillar README, its own pipeline) | **match** (−0.4% final; best at exact parity) | `RadarPillar_ViewOfDelft_Jul-21-2026_08:56:49` |
| 17 | TransRAD / RADDetDataset | **53.69 / 31.40** test mAP@0.3 / @0.5 (1 seed, official split, best-val-loss EMA weights = epoch 28 of 100; @0.1 67.38 / @0.4 43.81 / @0.6 20.22 / @0.7 10.07) | 61.90 / 38.76 (paper); **60.16 / 37.77** (authors' checkpoint, their Test.py) | **60.11 / 37.73** (authors' checkpoint through OUR adapter, decoder, NMS and evaluator, all 2032 frames; see the TransRAD section) | **within the pass rule, not at parity** (−10.8% / −16.9% vs the authors' checkpoint); port proven at parity, so the gap is the recipe — most likely the missing first training round, see the section | `TransRAD_RADDetDataset_Oct-16-2026_15:10:07` |
| 18 | T_RODNet / CRUW | **81.18 / 85.99** test AP/AR (1 seed, the authors' 36/4 split, final-epoch weights; pedestrian 79.39/85.45, cyclist 81.03/83.06, car 84.26/90.16) | 83.27 / 86.98 | not possible (no weights shipped); the **scorer** was checked instead — the authors' evaluator gives 81.19 / 86.00 on our final detections (see the T-RODNet section) | **reproduced** (−2.5% / −1.1%) | `T_RODNet_CRUW_Oct-20-2026_11:35:31` (epochs 1–88) + `T_RODNet_CRUW_Oct-21-2026_10:04:46` (resumed, epochs 89–100 + test) |


**All three numeric columns are percentages**, so a row can be read across without unit
conversion. Rows 10, 11, 12, 13 and 14 were normalized on 2026-09-09; their sources publish
fractions (RADDet 0.251, DAROD 0.5583, RadarCrossAttention 0.94), and the values here are
those ×100. Row 13 was the one that actually misled: it compared ours as a fraction (0.1799)
against a published percentage (81.23) in the same row, which reads as a 450× gap rather than
17.99 vs 81.23. The published-baselines table at the top of this file still quotes each paper
in that paper's own convention, so rows 10 and 14 appear there as fractions.

### Training cost and model size

Measured 2026-09-09 from the runs above: parameter count built from each run's own
`model_config`, checkpoint size on disk, and wall-clock span between the first and last
timestamped line of its `train_log.log` (so it includes dataset parse and per-epoch
validation — real elapsed time, not GPU-busy time).

| row | pair | params | ckpt | epochs | wall | per-epoch |
|---|------|-------:|-----:|-------:|-----:|----------:|
| 1 | FFTRadNet / RADIal | 3.79 M | 44 M | 100 | 11.4 h | 6.8 m |
| 2a | RODNet_CDC / CRUW | 34.52 M | 395 M | 6 † | 3.3 h | 32.6 m |
| 3 | RECORD / CRUW | 0.70 M | 8 M | 5 † | 5.3 h | **63.6 m** |
| 4b | RECORD / CARRADA RD | 0.70 M | 8 M | 72 ‡ | 6.2 h | 5.2 m |
| 4a | RECORD / CARRADA RA | 0.70 M | 8 M | 100 | 16.7 h | 10.0 m |
| 9 | MVRECORD / CARRADA | 1.97 M | 23 M | 45 † | 14.2 h | 18.9 m |
| 10 | RADDet / RADDetDataset | 10.75 M | 103 M | 100 | 7.7 h | 4.6 m |
| 11 | DAROD / CARRADA | 3.45 M | 40 M | 50 | **0.6 h** | **0.7 m** |
| 12 | DAROD / RADDetDataset | 3.46 M | 40 M | 75 | 0.9 h | 0.7 m |
| 13 | RAMP_CNN / UWCR | **104.22 M** | **1,193 M** | 22 †‡ | 5.1 h | 13.8 m |
| 14 | RadarCrossAttention / CARRADA | 11.62 M | 134 M | 80 | 1.1 h | 0.8 m |
| 15 | PointPillar / ViewOfDelft | 4.84 M | 55 M | 80 | 6.2 h | 4.7 m |
| 16 | RadarPillar / ViewOfDelft | **0.18 M** | 2 M | 80 | 9.6 h | 7.2 m |
| 17 | TransRAD / RADDetDataset | 5.79 M | 2.16 G (paper Table I) | 100 | **23.5 h** | 14.0 m (11.5 train + 2.4 val) |
| 18 | T_RODNet / CRUW | 156.8 M (paper Table IV 159.70 M) | 0.82 G | 100 | **23.9 h** (+ a 1.5 h kill/resume gap) | 14.2 m (12 train + 2 monitor on one sequence; 12 m test pass once) |

† resumed run — the row covers that DIRECTORY only, not the whole schedule, so wall time and
epoch count understate the run (rows 2a, 3, 9, 13). ‡ killed before its test phase (4b, 13).
Rows 15/16 have no test phase at all: VoD ships no public test split, and `train()` logs
"No files assigned to the test split".

**Parameter counts double as a free implementation check.** Three match their papers closely
enough to be evidence: FFTRadNet **3.790 M** against Table 4's 3.79 M (exact); RadarPillar
**0.184 M** against 0.27 M with PointPillar at **4.835 M**, reproducing the paper's ~26×
reduction claim; RadarCrossAttention **11.624 M** against 11.05 M, where the 0.52 M difference
is exactly the affine parameters of `LayerNorm([256,32,32])` (2 × 262,144) — the published
count evidently excludes them. A port whose parameter count is far off its paper's has a
structural defect, and this is the cheapest way to see it.

**RAMP_CNN is the outlier by an order of magnitude**: 104 M parameters, 28× FFTRadNet and
566× RadarPillar, at 1.2 GB per checkpoint. That is why its runs dominate disk (a 74-epoch run
is ~72 GB) and it is worth setting beside its result — the largest model here is also the one
that generalises worst to a held-out capture date.

Efficiency spans two orders of magnitude: DAROD trains a full 50-epoch schedule in **37
minutes**, while RECORD/CRUW costs **64 minutes per epoch**. Roughly **88 GPU-hours** across
these 13 directories, excluding the resumed segments not counted here.

Rows 11/12 are mAP@0.5; rows 4a/4b/9 are mIoU; row 10 is 3D-IoU AP at four thresholds, headline
figure @0.5. Row 10 is the only one compared against the paper rather than a reference anchor,
for the reason given in its closure section — it is the only pair that does not fall short.
The last column names the run directory under `$MODEL_ROOTDIR`
(`/home/kangle/dataset/trained_models`), so the weights behind every number can be found —
and so that pruning old runs does not destroy the campaign's evidence by accident. Each was
identified by matching the reported metric back to the run's `test_eval.csv` / checkpoint
filenames, not by name: several rows come from directories no prose in this file mentions,
and some are two directories because the run was resumed. Row 3 has no published-baseline
row of its own — its target is the ladder ordering, not an absolute figure — but it is
listed here so its runs are protected too.

**A run killed before its test phase leaves its headline number nowhere on disk.** Row 4b
was briefly recorded here as "not on disk" (2026-09-08) on exactly that mistake: its run was
killed at epoch 71 of 100, so it has no `test_eval.csv`, and searching stored results for
53.15 found nothing. The weights were always there — re-evaluating `epoch71` on the test
split reproduces the doc's table **class for class** (Bg 99.45 / Ped 39.22 / Cycl 23.02 /
Car 50.90, mIoU 53.15), which is how the number was produced on 2026-08-24 in the first
place: a standalone evaluation whose output went straight into this file. The same shape
applies to rows 13 and 14's interrupted runs. Before concluding a result's weights are lost,
**re-evaluate the best checkpoint** — absence of a `test_eval.csv` says nothing about
whether the model survives.

Rows 15/16 are VoD **validation**-split numbers (VoD has no public test split), entire-area
mAP in percent from the official devkit, final weights with the best epoch alongside; they
come from the July slice gates recorded in
`docs/superpowers/specs/2026-07-16-radarpillar-design.md:144-164` — see the closure below.
**Row 13 is not a comparison at all**: its "published" cell is filled in only to show what it
is *not* being measured against. 78% of RAMP-CNN's data was never released and every public
raw sequence is training-side for the published model, so no split we can build is
interpretable against 81.23 / 84.25. The other runs are not yet measured
against their baselines. (Row ids follow the published-baselines table above: 4a/4b are
RECORD on CARRADA RA/RD. An earlier revision labelled the RD result `6a`, which in that
table is RECORDNoLstm/CARRADA-RA -- a different pair.)

**Read RECORD/CARRADA RD against 53.19, not 58.10** — the same situation as RADDet
below, but established far more sharply. The reference implementation itself, run from
a pristine checkout of `colindecourt/record@52b6f37` on its own pinned stack (torch
1.12.0+cu116, pytorch-lightning 1.8.1) against our CARRADA copy, scores **53.19** test
mIoU where the paper reports 58.10. Ours scores 53.15 — a gap of 0.04, with the same
per-class shape on both sides:

| class | reference on our data | ours | published |
|-------|----------------------|------|-----------|
| Bg | 99.44 | 99.45 | 99.60 |
| Ped | 40.35 | 39.22 | 46.60 |
| Cycl | 20.52 | **23.02** | 28.60 |
| Car | 52.46 | 50.90 | 57.50 |
| **mIoU** | **53.19** | **53.15** | 58.10 |

Both implementations miss the published number in the same classes by similar margins,
and ours is the better of the two on cyclist. So the ~5-point deficit is **not in this
repo's port** -- it is a property of the data copy or of some aspect of the published
setup that neither implementation reproduces, and it cannot be closed by changing our
code. The validation trajectories also tracked each other throughout (best val mIoU
63.05 reference vs 63.86 ours, peaking at the same epoch).

Everything that differed from upstream was enumerated before the run: five dataset and
project paths (upstream ships unusable `path_to_datasets/` placeholders that its own
Dockerfile substitutes), `n_epoch` 500 -> 100 to match our budget, and one logging line.
That last one is a finding in itself: `carrada_trainer_sv.py:158` calls
`self.log(test_results_log)` with a dict, which raises `TypeError` on the pinned
pytorch-lightning 1.8.1, so the published code **cannot run as-is**. Until it is fixed,
`val_metrics/rd_prec` is never logged and the `ModelCheckpoint` / `EarlyStopping`
monitors in `main_carrada.py` have nothing to track -- meaning the precision-monitored
checkpoint selection was never exercised by the authors in this code state.

**Where the remaining ~5 points went -- four hypotheses tested and eliminated**, so
they do not need re-litigating:

| hypothesis | verdict | evidence |
|------------|---------|----------|
| our port is defective | ruled out | reference scores 53.19 on our data, we score 53.15 |
| our CARRADA copy differs from theirs | ruled out | processed-RD mean 58.4271 vs the `rd_stats_all.json` constant 58.4711 (0.08%); min/max gaps are sampling artefacts of reading 121 of ~12.5k frames. NOTE the comparison must use `range_doppler_processed/`, which is what the loader reads when `process_signal: true` (`dataloaders.py:113`), not `range_doppler_numpy/` |
| wrong split option | ruled out | `--data_split` and `--reproduce` are parsed but never read on the CARRADA path (`--data_split` is only asserted in `eval_cruw.py`) |
| the published number came from a different eval path | ruled out | `eval_carrada.py` builds the same dataloader and calls the same `trainer.test`, hitting the same `Evaluator` and global confusion matrix |

What survives is that **the published artifact is not the code that produced the
published numbers** -- proven by the `TypeError` above, which makes a validation epoch
impossible at this commit. Whatever the authors actually ran differed from what they
released, and the difference is not recoverable from the repository. Both sides here are
also single-seed. Further chasing of the gap through this repo's code is not warranted.

**Read RECORD/CARRADA RA against 33.08, not 34.80.** The same reference run was repeated
for the RA view (same pristine checkout, same pinned stack, same CARRADA copy, 100-epoch
budget, no early stop triggered, tested on its precision-best checkpoint at epoch 76):

| class | reference on our data | ours | published |
|-------|----------------------|------|-----------|
| Bg | 99.23 | 99.17 | 99.70 |
| Ped | 8.07 | 7.36 | 10.30 |
| Cycl | **4.71** | **3.55** | 1.40 |
| Car | 20.32 | **20.86** | 27.70 |
| **mIoU** | **33.08** | **32.74** | **34.80** |

A 1.7-point shortfall, ~5% relative -- the same signature as the RD view, on a second
independent view. Two details worth keeping:

- **The val-to-test drop is small here** (34.59 -> 33.08, 1.5 points) against RD's ~10.
  Background contributes ~25 of the 34.8 total and transfers between splits essentially
  unchanged, so almost the whole drop lands on the three object classes.
- **The shortfall is not uniform**: car is 7.4 low and pedestrian 2.2 low, but cyclist is
  3.4 points *higher* than published (4.71 vs 1.40). A published cyclist IoU of 1.4 is
  close enough to zero that it is plausibly noise in the authors' own run, which is a
  reason to weight the RA aggregate lightly in either direction.

**Our RA run lands at 32.74, −0.34 against that anchor (−1.0%).** 100 epochs, seed 3,
official split, hflip only, early stop parked, tested on the mAP-best checkpoint
(epoch 73, val mAP 0.3574 / mIoU 0.3501); 16h37m wall-clock, no resume. Two properties
make this a match rather than a near-miss:

- **The per-class deltas against the reference are mixed in sign** (Bg −0.06, Ped −0.71,
  Cycl −1.16, Car **+0.54**). An implementation difference pushes the classes the same
  way -- as the gap to *published* does, where both our run and the reference are low on
  car and high on cyclist. Mixed signs at this magnitude are run-to-run variance.
- **The val peak is unresolvable to better than this.** Mean epoch-to-epoch |Δ mIoU| over
  the last 30 epochs is 0.0080, while the top-8 epochs by val mIoU span only 0.0069 --
  nine epochs sit within one noise-step of the peak (0.3427-0.3501). Which of them wins
  on val is arbitrary, so a 0.34-point test difference is below what this protocol can
  resolve. Selecting the test-best checkpoint instead would be test-set selection and was
  not done.

Note our val mIoU (0.3501) *exceeds* the reference's (0.3459) while our test mIoU comes in
0.34 lower -- consistent with both runs sampling the same distribution rather than one
being systematically better.

**Running the RA view also required a third patch to the reference**, and this one is
substantive rather than cosmetic: `CarradaDatasetRangeAngle.__getitem__`
(`datasets/carrada/dataloaders.py:330`) files the range-ANGLE mask under the key
`'rd_mask'` -- copy-pasted from the range-Doppler class -- while
`carrada_trainer_sv.py:174` reads `batch['ra_mask']` for that view. The RA path therefore
raises `KeyError` on its first validation batch and **cannot train for a single epoch as
published**. Since Table V reports an RA number, that path certainly ran for the authors,
which independently confirms the conclusion above: the released artifact is not the code
that produced the published table.

**Read MVRECORD/CARRADA against 56.35 RD / 38.09 RA, not 63.2 / 44.5.** Same pristine
checkout and pinned stack, 100 epochs, tested on the reference's own precision-best
checkpoint (epoch 99, `global_prec` 0.5538):

| class | RD IoU (ours-run reference) | RA IoU (ours-run reference) |
|-------|------------------|------------------|
| Bg | 99.53 | 99.52 |
| Ped | 45.21 | 17.91 |
| Cycl | 26.23 | 10.59 |
| Car | 54.43 | 24.35 |
| **mIoU** | **56.35** (published 63.2) | **38.09** (published 44.5) |

A 6.9-point shortfall on RD (−10.8%) and 6.4 on RA (−14.4%) -- the same signature as the
single-view runs, now on the multi-view model, and the largest gaps measured so far.

**The val-to-test drop is what matters here, and it is much larger than RA's.** Best
validation was **65.29 RD** (epoch 99) and **42.53 RA** (epoch 92) -- RD *above* its
published 63.2. Test then came in 8.9 points lower on RD and 4.4 lower on RA, against
only 1.5 for the RECORD/CARRADA RA run. So a mid-run reading of the val curve suggests
MVRECORD reproduces its paper on RD; the test evaluation says it does not. Do not quote
MVRECORD validation figures as if they were the anchor.

Two caveats attach to this anchor, both forced and both documented rather than worked
around:

- **batch 4 with `accumulate_grad: 2`, not batch 8.** MV-RECORD at the published batch
  size OOMs on a 24 GB 3090 (20.61 GiB allocated at the failure). Accumulation preserves
  the published *effective* batch, but is not identical to it: weighted CE normalises by
  the sum of class weights present in the batch (see the eval-batch section above), so two
  batches of 4 do not compose into one batch of 8. Gradient direction is close -- the
  model is LayerNorm, which is per-sample -- but this is a real deviation.
- **Resumed at epoch 53.** The run was stopped and restarted from `last.ckpt`; Lightning
  restores optimizer, scheduler and callback state, but epoch 53 draws a fresh shuffle
  rather than continuing the interrupted epoch's order. Same training procedure, different
  random draw.

Resuming also required a fifth patch to the pristine repo, and it is the first that only
appears on resume: `carrada_trainer.py`'s `on_train_start` now moves Adam's `step` state
back to CPU. PL 1.8.1 calls `_optimizer_to_device` immediately after
`optimizer.load_state_dict` (`strategies/strategy.py:369`) and moves *every* optimizer
state tensor to CUDA, including `step` -- which torch 1.12 deliberately excludes from that
cast (`optim/optimizer.py:181`) because the non-capturable path asserts
`not step_t.is_cuda` (`optim/adam.py:255`). Lightning undoes torch's own guard and resume
dies. Restoring `step` to CPU keeps the resumed run on the identical Adam code path;
setting `capturable=True` would also have worked but would have switched the anchor to a
different implementation mid-run.

One thing this run settles: **the training can collapse, but does not do so reliably.** An
earlier attempt (killed at epoch 14 by the USB drive dropping off the bus) saw RD fall
from 55.59 at epoch 9 to ~25.0 -- the background-only value -- and stay there, with
`global_prec` then selecting that degenerate state as its best checkpoint. The completed
run passed the same epochs cleanly and never collapsed in 100 epochs. So the collapse is a
real failure mode of these hyperparameters, not a deterministic one, and the
precision-monitored selection will not protect against it.

### Closed: MVRECORD / CARRADA (Phase 2) — match against the reference anchor

Our run, same official split, same `_reference` config (100 epochs, batch 4 ×
`accumulate_grad` 2, val/test batch 8), tested on the mAP-best checkpoint:

| | ours | reference on our pipeline | published |
|---|---|---|---|
| mIoU RD | **54.83** | 56.35 | 63.2 |
| mIoU RA | **38.89** | 38.09 | 44.5 |
| mAP | 0.4853 | — | — |
| mAR | 0.8218 | — | — |

**−1.52 on RD (−2.7%), +0.80 on RA (+2.1%)** — the two views disagree in sign, which is
what run-to-run variance looks like and not what an implementation difference looks like.
A defect in code the two views share (dataset, backbone, loss, metric chain) would move
both the same direction. Per class:

| class | RD ours | RD reference | Δ | RA ours | RA reference | Δ |
|-------|---------|--------------|---|---------|--------------|---|
| Bg | 99.48 | 99.53 | −0.05 | 99.58 | 99.52 | +0.06 |
| Ped | 42.03 | 45.21 | −3.18 | 18.53 | 17.91 | +0.62 |
| Cycl | 25.73 | 26.23 | −0.50 | 11.74 | 10.59 | +1.15 |
| Car | 52.08 | 54.43 | −2.35 | 25.73 | 24.35 | +1.38 |
| **mIoU** | **54.83** | **56.35** | −1.52 | **38.89** | **38.09** | +0.80 |

Note the sign structure carefully: **within** a view every class moves the same way, and
only **between** views do the signs flip. That is expected -- each view has its own
decoder head and all four of its classes ride on it -- so this is two independent
observations that disagree, not eight. Do not read the eight per-class deltas as eight
pieces of evidence. The shared Background class is flat to ±0.06 on both views, which is
the part that would move first if the common code path differed.

The largest single gap is RD Pedestrian at −3.18, and it is a precision effect rather than
a miss: recall on that class is 0.9175 against precision 0.4368, i.e. we over-predict
pedestrians in range-Doppler slightly more than the reference does.

**The validation trajectories are statistically indistinguishable**, which is the stronger
evidence and does not depend on a single checkpoint draw. Over epochs 40-87 (post-warmup,
n=48 each):

| | ours | reference | diff | Mann-Whitney |
|---|---|---|---|---|
| RD | 59.34 ± 2.73 | 59.58 ± 2.58 | −0.23 | p = 0.68 |
| RA | 37.61 ± 1.94 | 37.92 ± 1.99 | −0.31 | p = 0.40 |
| mAP | 0.4971 ± 0.0234 | 0.4995 ± 0.0226 | −0.0024 | p = 0.61 |

The mean differences are an order of magnitude below the epoch-to-epoch spread. Both runs
also happened to peak on mAP at **epoch 99** (ours 0.5543, the reference's `global_prec`
0.5538) and therefore tested the same epoch index. Val-to-test drop was 10.16 RD / 3.19 RA
for ours against 8.94 / 3.68 for the anchor -- the transfer penalty behaves the same on
both sides, and it is again large enough that the val curve alone would have been
misleading (our best val RD was 64.99, close to the published 63.2).

Ours was **resumed at epoch 55** from its own epoch-54 checkpoint, so it carries the same
kind of fresh-shuffle discontinuity as the anchor (resumed at 53). `train()` restores
weights, optimizer, scheduler, `global_step` and `history`; the interrupted epoch's
remaining batches are discarded and that epoch is redone from its start.

One artifact to ignore in `docs/phase2-mvrecord-carrada.json`: the run is labelled
`verdict: suspect` with note `loss 0.321 -> 0.4571 (NOT down)`. That is `smoke_sweep`'s
first-vs-last val-loss heuristic applied to the resumed segment only, and this model's val
loss oscillates in 0.25-0.46 with no trend while mIoU climbs -- the reference does exactly
the same. `epochs_recorded: 45` likewise counts only epochs 55-99; the checkpoint history
holds all 100.

**Read RADDet against 0.4478, not 0.4657.** The reference implementation itself scores
only 0.4414 on our copy of RADDet (0.4478 through our loader/evaluator), 5.2% under its
own published figure. That is about 1 sd of the seed spread, so it may be seed noise --
but the builder that produced the published number is not recoverable from that repo
(its `config_raddet.json` pins dataset_version 1.0.0 while `raddet.py` declares 1.1.0,
so the published config cannot load the published builder). CARRADA has no such problem:
there the reference reproduces the paper at both thresholds (0.5578/0.7044 against
0.5583/0.7068).

### Closed: RODNet_CDC / CRUW (Phase 2) — inconclusive-by-protocol, healthy per scene

Verified ready before launch (2026-08-20): config, GT distribution (640/640 labels
scannable), sample variation, CDC backbone structurally identical to the reference,
`nn.BCELoss` both sides, and the full OLS metric chain against the official devkit.
Ran with `--config-variant _reference` (lr 1e-5, StepLR(5, 0.1)), budgeted 25 epochs,
**stopped at 10**: the x0.1-every-5 anneal has lr at ≤1e-7 from epoch 10, and val had
plateaued at 50±3 since epoch 3. Two GPU windows, joined via `--resume` (checkpoints
carry optimizer, scheduler, epoch and history; only the RNG draws differ from an
uninterrupted run, within the measured seed spread).

**The split is ours, not the paper's** (30/5/5, stratified by scene type so on-road
sequences appear in all three splits; the published split is not recoverable from the
paper, the reference repo, or the data).

**Found and fixed during this run: per-epoch evaluation contamination.**
`write_dets_results_single_frame` opens its results file with `'a+'`, and every epoch's
validation writes into the same exp dir — so epoch N was scored against the union of
all detections from epochs ≤ N. Duplicates count as false positives under OLS matching,
so logged val mAP *decayed* monotonically after the anneal (48.5 → 23.3 over epochs
4–9) while mAR climbed — a signature worth remembering, since it looks exactly like
overfitting. Same trap existed in `RECORD_CRUW_evaluation`; both now delete stale
result files first (RAMP_CNN scores in memory, unaffected). The checkpoints themselves
were clean; all 10 were re-evaluated into fresh directories.

**Found and fixed after the first closure draft: test-stride mismatch in the evaluator.**
`RODNet_evaluation`'s frame-pop loop advanced by `train_stride` (4) regardless of split,
while the test loader windows at `test_stride` (8) — so only frames 0–3 of every
8-frame advance were ever written (~50% of test frames, 20% on the long `ONRD002`), the
overlap averaging was misaligned, and every unwritten frame scored as a miss. The
reference (`yizhou-wang/RODNet` `tools/test.py`) uses `test_stride` in both places; the
`train_stride` was our port's error, and the val path was only accidentally correct
(4/4). A related boundary flush was added for our single cross-sequence loader (the
reference builds one dataset per sequence, so it never carries a stack tail across
sequences). The first-draft test numbers (18.5 / 20.9) were artifacts of this bug; the
corrected numbers below come from re-evaluating the same epoch-7 checkpoint with the
fixed evaluator.

Clean val trajectory (re-evaluated, AP / AR):

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|---|---|---|---|---|---|---|---|---|---|
| val mAP | 4.0 | 31.9 | 36.1 | 50.2 | 48.3 | 51.9 | 46.7 | **53.5** | 51.4 | 49.8 |
| val mAR | 10.5 | 52.9 | 52.5 | 62.9 | 59.0 | 61.0 | 56.1 | **62.9** | 60.4 | 57.7 |

Best epoch 7: **val 53.5 / 62.9, test 38.5 / 44.6** (fixed evaluator; epoch 8 test:
36.9 / 43.3, epoch 9: 36.9 / 42.8 — the same ordering as val, so best-val selection
and the reference's latest-checkpoint convention differ by under 2 points here). To
rule out any remaining protocol asymmetry, the val sequences were also re-scored under
the exact test protocol (stride 8, test eval path): **51.8 / 62.7** — within 2 points
of the per-epoch figure, so the per-epoch val numbers stand and the remaining val−test
gap is ~13 points under an identical protocol.

Per-sequence, epoch 7 (fixed evaluator, AP / AR):

| split | sequence | AP | AR |
|-------|----------|-----|-----|
| test | `PM1S014` (parking) | 89.2 | 93.5 |
| test | `PBMS007` (parking) | 69.0 | 75.2 |
| test | `PMS3000` (parking) | 64.6 | 75.6 |
| test | `ONRD002` (on-road) | 23.8 | 30.8 |
| test | `CM1S004` (campus) | 13.2 | 15.5 |
| val | `PM1S013` (parking) | 92.1 | 95.8 |
| val | `BMS1002` (parking) | 92.1 | 96.9 |
| val | `PMS2000` (parking) | 63.3 | 75.5 |
| val | `PBMS004` (parking) | 43.7 | 54.2 |
| val | `ONRD001` (on-road) | 29.9 | 43.3 |

**Why the verdict is inconclusive-by-protocol, not fail.** The published 74.29 is not a
reachable target (see "no CRUW baseline is locally reproducible" above), and the
remaining ~13-point val−test gap is per-sequence composition: same weights, same
protocol, and the spread runs 13.2 (`CM1S004`, a scene type with no val counterpart) to
92.1; test drew the two hardest sequences of the whole pool while val's hardest is
29.9. The parking scores sit at the published per-scene ceiling (RECORD Table I: PL
95.0; RAMP-CNN: ParkingLot 97.38) — a broken implementation does not score 89–92 AP on
anything. What the closure cannot exclude is a residual training deficit on the hard
scenes (`CM1S004`, on-road); that is exactly what the ladder check below resolves.

**Pre-registered ladder check** (the falsifiable test that replaces the absolute rule):
when the RECORD family runs on this same split, the published ordering must reproduce —
RECORD (72.8) > NoLstmMulti (65.5) > NoLstmSingle (59.5), with RODNet_CDC below RECORD
— and per-sequence scores cross-check the split story: if RECORD also scores ~15 on
`CM1S004`, the sequence is the cause; if it scores ~50 there, revisit RODNet_CDC's
training. Same data on both sides, so deviation from the ladder is
implementation signal, not split noise.

### RECORD family: verified, ready to run

The full pre-run comparison against `colindecourt/record` (the same checklist that
found defects in every other family) came back **clean** -- the only real defect
remains the scheduler frequency, fixed via `RECORD_train_config_reference.json`:

- **Model code**: encoder/decoder/IR/LSTM layers and the NoLstm variant are verbatim
  copies of the reference; wrapper forwards (RECORD, RECORDNoLstm, MVRECORD incl.
  cross-view skip connections) match line-for-line.
- **Architecture configs**: programmatic deep-diff of `RECORD_model_config.json`
  against all four reference YAMLs (CRUW, CARRADA RD/RA, MVRECORD): exact match,
  including CARRADA's asymmetric [2,1] strides and [1,0] output paddings.
- **Losses**: identical on every path -- BCE (CRUW); `mean([wCE, 10*SoftDice])` with
  byte-identical dice/one-hot code and class-weight order (CARRADA);
  `mean(rd_loss + ra_loss)` (MVRECORD).
- **Input assembly**: NoLstm reshapes match the reference's `on_before_batch_transfer`;
  GT is the last-frame confmap on both sides.
- **Evaluators**: CRUW reuses the devkit-verified OLS chain with the identical
  truncated-first-window protocol; CARRADA's `Evaluator` is the same verbatim MVRSS
  copy on both sides.

Two protocol deviations were found and one closed:

1. **The reference early-stops** (patience 20 on CARRADA pixel precision, 7 on CRUW
   val loss), so its published epoch counts are ceilings. `train()` now supports a
   config-driven `early_stop` block and the reference config carries the reference's
   own values. For CARRADA this also aligns checkpoint selection: our `mAP` there *is*
   mean pixel precision, the metric the reference monitors.
2. **CRUW checkpoint selection still differs**: the reference keeps best-val-loss, we
   keep best-val-mAP. Documented, not changed -- selecting on the comparison metric is
   the defensible side of the deviation.

#### MVRECORD: re-audited before Phase 2 (2026-08-26), one defect found and fixed

The family audit above covered MVRECORD's model code, architecture config and loss. The
per-view re-check before launching it found one real defect and confirmed the rest:

- **Evaluator (DEFECT, fixed)**: `MVRECORD_CARRADA_evaluation` averaged `range_doppler`
  with *itself*, so the reported mIoU was the RD score alone and RA was computed and
  discarded. Correcting the typo is not sufficient -- the reference logs `rd_miou` and
  `ra_miou` separately (`executors/carrada_trainer.py:191,195` val, `322,326` test) and
  builds `global_` aggregates for acc/prec/dice ONLY, never mIoU, matching the paper's
  pair (63.2 RD / 44.5 RA). The evaluator now returns `mIoU_rd` and `mIoU_ra`.
- **Dataset**: our multi-view `__getitem__` is line-for-line the reference's
  `CarradaDataset.__getitem__` -- same annotation masks, processed/raw selection,
  `dstack`+`rollaxis`, one flip decision shared across all three views, the fake-mask
  trick for AD, `add_temp=True`. Normalisation sits in `__getitem__` for us and in the
  trainer (all three stages) for them: same operation, same order relative to the flip.
  The three stats constants are byte-identical to the reference's JSON, `ad_stats_all`
  included -- the first check to exercise it, since AD is unused by the RD/RA runs. Our
  CARRADA copy has `angle_doppler_processed`. Flips are gated on `split == 'train'` on
  both sides, so val/test see no augmentation.
- **Loss**: `wce_w10sdice` builds 2 criteria, so the reference takes its `nb_losses < 3`
  branch and the coherence loss (CoL) never applies to the published config. Our
  `get_loss` (`network.py:700-706`) is that branch line-for-line.
- **Train config**: Adam + lr 1e-3 + `StepLR(20, 0.9)` == the reference's Adam +
  `ExponentialLR(gamma=0.9, frequency=20)`; batch 8; hflip only; byte-identical per-view
  class weights. `test_checkpoint` on `mAP` is exact: the reference monitors
  `val_metrics/global_prec = (rd['prec'] + ra['prec'])/2`, and `get_pixel_prec_class`
  returns `prec = np.mean(prec_by_class)` (`evaluate.py:971`), which is term-for-term
  what our `mAP` averages.
- **`config_name: 'mv-record-s'`** inside `models/configs/mv_record.yaml` is a dead
  label: never read by any code, `get_models` has no `MV-RECORD-S` branch, and the README
  gives `config_mvrecord_carrada.yaml` as *the* MV-RECORD buffer training command. The
  architecture we build is the one that command builds.

**Eval batch size: adopted the reference's, and fixed a defect it exposed.** The
reference builds its val *and* test loaders with `train_cfg['batch_size']` (8), not 1.
Measured on our box, batch 8 is **~4x faster** (9.5 -> 37.5 samples/s; validation on the
RA run's shape would drop from 4m16s to ~1m05s per epoch, ~5h over 100 epochs), so
MVRECORD's reference config now uses batch 8 for val and test. RD/RA are left at their
measured batch-1 settings; their numbers stand.

Three things were checked before making that change:

- **The comparison metrics are unaffected in substance.** The `Evaluator` accumulates one
  global confusion matrix, so each sample is counted once regardless of grouping. The
  measured drift between batch sizes is ~8e-06 mIoU -- float non-determinism from
  batch-shape-dependent kernels, not a semantic change; each batch size is bit-reproducible
  across repeats. (An earlier revision of this file said batching "cannot move the metric";
  that was too strong -- it cannot move it *systematically*, but it does move it at 1e-5.)
- **A real defect, now fixed.** Every evaluator accumulated
  `running_loss += loss.item() * batch_size` using the CONFIGURED batch size, while
  `get_loss` reduces over the ACTUAL batch -- so a short final batch was over-weighted
  by `configured/actual`. Reachable whenever the split is not a multiple of the batch
  size (`drop_last` is unset, so partial batches do occur). Demonstrated on CARRADA val at
  batch 16 (151 full + 8): reported 1.86479782 against the correct 1.85897204. Fixed at
  all nine sites plus a tenth instance of the same class in `RADDet_evaluation`, whose mAP
  was normalised by `batch_size * len(dataloader)` rather than the frame count.
  **No completed result changes**: RD/RA/CRUW/RODNet all ran eval at batch 1, where
  configured == actual always, and the bug touches only the loss, never the confusion
  matrix. DAROD (val batch 4/16) had inflated val *losses*, but its verdict rests on
  mAP@0.5 and it has no loss-based early stop or checkpoint selection.
- **The loss is not comparable across batch sizes**, by construction and on both sides:
  `CrossEntropyLoss(weight=..., reduction='mean')` divides by the sum of class weights
  present in the batch, so batches whose frames differ in class composition do not
  decompose into per-sample losses (measured: 6 of 25 batches differ, max 0.0153). Read
  MVRECORD's val loss only against itself, never against the batch-1 RD/RA curves.

### TransRAD: validated against the authors' checkpoint (2026-10-16)

TransRAD (radar-lab/TransRAD @8f9b9ad, GPL-3.0; a pristine clone with the shipped
`model_data/best_epoch_weights.pth` lives at `/home/kangle/dataset/transrad_reference/`)
is the first network added through the sample-adapter mechanism: `RADDetDataset` is
untouched and `TransRADOnRADDet` builds the reference's sample (log-magnitude cube,
Doppler resized 64→256 by nearest neighbour, edge boxes clamped to their larger part).

**The strong check ran first.** `scripts/transrad_reference_check.py` loads the authors'
checkpoint into our port with `strict=True` (650 keys, 5,788,131 parameters, no key map)
and scores the official 2032-frame test split through our adapter, collate, decoder,
two-stage NMS and evaluator:

| IoU | authors (Test.py, 2028 frames) | ours (2032 frames) |
|---|---|---|
| 0.1 | 0.7309 | 0.7304 |
| 0.3 | 0.6016 | 0.6011 |
| 0.4 | 0.4958 | 0.4951 |
| 0.5 | 0.3777 | 0.3773 |
| 0.6 | 0.2496 | 0.2498 |
| 0.7 | 0.1293 | 0.1292 |

Per-class AP@0.3 ours / authors: person 0.268 / 0.269, bicycle 0.338 / 0.334, car
0.727 / 0.728, motorcycle 0 / 0, bus 0.763 / 0.763, truck 0.635 / 0.636. Every gap is
under 0.001, i.e. the four frames their `drop_last` skips plus float order. Data path,
model, decoding, NMS and metric are therefore proven against the reference; whatever a
training run scores is attributable to the recipe.

**Deviations from the reference, all deliberate:** official 7314/812 split instead of
their unsorted 5 % hold-out (≈7720/406; the test set is identical); all 2032 test
frames scored; `SyncBatchNorm` → `BatchNorm2d` (same keys, same single-GPU maths);
feature strides fixed at 8/16/32 instead of a dummy forward at construction; per-step
prints dropped; the two `timm` helpers vendored. The recipe in
`train/TransRAD_train_config.json` carries the script's *effective* values, which differ
from the paper's prose: lr 3e-4 (the script clamps 1e-3 by batch size), cosine with a
3-epoch quadratic warmup and the last 5 epochs held at 3e-6, Adam betas (0.937, 0.999),
EMA 0.9999 with tau 2000, grad clip 10, batch 4, 100 epochs, seed 11, best checkpoint by
EMA validation loss. Three config-gated additions to `train()` carry it: `cos_warmup`,
`ema`, `optimizer.betas`; every existing config is bit-for-bit unchanged.

**Training result (row 17).** `TransRAD_RADDetDataset_Oct-16-2026_15:10:07`, 100 epochs,
23.5 h on the RTX 3090, 14.0 min per epoch. Validation mAP@0.3 rose to 0.55 by epoch 29
and stayed between 0.54 and 0.56 for the remaining 70 epochs (best 0.562 at epoch 34),
while validation loss bottomed at 34.68 at epoch 28 and climbed to 50.3 by epoch 99 as
training loss fell from 19.5 to 5.6: the model overfits from epoch 30 on and the
cosine tail does not recover it. The reference's rule (lowest EMA validation loss) selects
epoch 28, which scores on the 2032-frame test split:

| IoU | 0.1 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| ours, epoch 28 | 0.674 | **0.537** | 0.438 | **0.314** | 0.202 | 0.101 |
| authors' checkpoint | 0.731 | 0.602 | 0.496 | 0.378 | 0.250 | 0.129 |

Per-class AP@0.3 ours / authors: car 0.680 / 0.728, bus 0.500 / 0.763, truck 0.555 /
0.636, person 0.195 / 0.269, bicycle 0.159 / 0.334, motorcycle 0 / 0. The headline is
10.8% under at IoU 0.3 and 16.9% under at 0.5: inside the campaign's ~20% rule, not a
match. Since the gate shows our pipeline reproduces the authors' weights to the third
decimal, the gap is in how the weights were trained, and the most likely cause is in
the paper itself (§III.E): "after model convergence, adjusting α1 = 40 and α2 = 15 and
a second round of transfer learning can significantly improve objectness and
classification accuracy". The shipped `yolo_training.py` carries exactly those gains
(with 30 and 7.5 still in comments) and the shipped `Train.py` starts from
`best_epoch_weights.pth`, i.e. the published script IS the second round. This run
applied the second-round gains from random init, which is neither round. A faithful
reproduction would be round one with gains 30 / 7.5 from scratch, then the shipped
script from round one's best epoch; the second round's schedule is not documented and
would have to be guessed. Smaller contributors: 7314 training frames instead of about
7720, and a different (and larger) validation set driving the checkpoint choice.

### T-RODNet: the first CRUW result comparable to its paper (2026-10-21)

**Why this row is different from every other CRUW row.** Rows 2–7 were measured on data
we do not hold (the unreleased full corpus, or the challenge server's hidden test set).
The T-RODNet authors instead hold out **four of the forty labelled ROD2021 sequences**
as their test set — `2019_04_09_BMS1001`, `2019_04_30_MLMS001`, `2019_05_23_PM1S013`,
`2019_09_29_ONRD005`, named in their README — and train on the other 36. We have all
four with annotations, and the authors ship their four ground-truth files in
`evaluate/gt/`, byte-identical to our `TRAIN_RAD_H_ANNO` files. So the published
83.27 / 86.98 is the first CRUW number this repo can compare to **absolutely**, under
the 20 % rule, instead of through the relative ladder.

**Result.** Final-epoch weights on the four sequences, stride 8, our RODNet
post-processing and OLS evaluator:

| | AP | AR |
|---|---|---|
| **ours** | **81.18** | **85.99** |
| paper Table I | 83.27 | 86.98 |
| | −2.5 % | −1.1 % |

| sequence | AP | AR | | class | ours AP / AR | paper AP / AR |
|---|---|---|---|---|---|---|
| BMS1001 | 83.3 | 84.4 | | pedestrian | 79.4 / 85.5 | 82.2 / 85.4 |
| MLMS001 | 74.9 | 80.0 | | cyclist | 81.0 / 83.1 | 82.3 / 84.3 |
| PM1S013 (parking) | 96.9 | 98.4 | | car | 84.3 / 90.2 | 86.2 / 92.5 |
| ONRD005 (on-road) | 83.3 | 90.2 | | | | |

**Verdict: reproduced.** One seed; the per-class gaps are 1–3 points, uniform across
classes, which is seed-sized for a single run rather than a systematic miss.

**What was and was not proven at parity.** The reference ships no trained weights, so
the checkpoint-through-our-pipeline gate used for TransRAD was impossible. Two things
were checked instead:

- **The port.** `networks/trodnet.py` is a class-for-class port of `Nets/T_RODNet.py`
  (commit 39a2444, MIT) with `einops.rearrange` replaced by permutes and the FLOPs code
  dropped; its state-dict key set and parameter count (156,805,004) equal the
  reference class loaded from the clone (`test_trodnet.py`). The paper's Table IV says
  159.70 M; the reference's own "31.54 M" comment is thop's count of the modules its
  forward calls, not of the model.
- **The scorer** (`scripts/trodnet_scorer_check.py`). The authors revised the public
  CRUW evaluation package; their `cruw.eval.evaluate_rod2021` was run from the clone
  on the same detection files as ours. On 16,363 RODNet detections over ten sequences
  the two agree to 0.001 AP / 0.003 AR with full-precision scores (their detection-file
  format rounds scores to 0.01, which reorders near-ties and moves AP by ≤ 0.25); on
  the final T-RODNet detections they give **81.19 / 86.00** against our 81.18 / 85.99.
  Their pooled "total" and our object-count-weighted summary are the same number.

**Recipe** (`T_RODNet_train_config.json`, transcribed from
`configs/config_T_Rodnet_win16.py` and `tools/train.py`): Adam lr 1e-4, weight decay 0;
`CosineAnnealingLR` over the 100 epochs, stepped per epoch (the new `cosine` branch in
`train.py`); batch 4; 16-frame windows, train stride 4, test stride 8; random chirp
per window in training, chirp 0 at test (`is_random_chirp=False` in their test
script); `max_dets` 20, peak and OLS thresholds 0.3; seed 3. **Loss: `nn.MSELoss` on
the raw output**, which is what `tools/train.py:135` does — the paper's text says
smooth-L1. **Batch 4** is the shipped config; the paper's text says 2. No validation
set and no checkpoint selection: the authors test their final checkpoint, and so does
this row (`test_checkpoint` omitted).

**Monitor.** The training loop needs a val split, so the per-epoch curve was scored on
one of the four test sequences (`MLMS001`, `--monitor one`) at stride 4 — a monitor
only, never used for selection. It rose 39.4 → 63.7 (epoch 3) → a 75–80 band from
epoch 9, peaked at 79.5 (epoch 58), and settled at 75.8 / 80.6 once the learning rate
fell below ~2e-5; its stride-8 test score is 74.9 / 80.0. Training loss 0.00351 →
0.00025, still falling slowly at the end.

**Cost, and what made it affordable.** The paper quotes 15 h on a 3080 Ti (training
only, no per-epoch evaluation; the reference uses no AMP, no special loader). Our first
timing run took 32.4 min/epoch (→ 54 h) for two reasons outside the recipe: CRUW lived
on the 18 TB spinning disk (6.4 ms per cold chirp read, 16 reads per window), and the
40 sequence pickles hold their pre-rendered confidence maps as **float64 — 19.9 GB** —
which the trainer and every loader worker held in RAM (30 GB host), starving the page
cache so every epoch re-read the files cold. Fixes, none touching the recipe: the
radar files were moved to the NVMe (0.5 ms per read); the parser now serves the maps
through a memory-mapped sidecar `.npy` (same bytes, one shared copy; host memory
30 → 10 GB; `CRUW._memory_mapped_confmaps`); the monitor scores one sequence instead of
four. Result: **14.2 min/epoch** (≈12 train + 2 monitor), 12 min for the one test
pass, 23.9 h for the run. GPU peak 343 W / 73 °C. The run was killed once at epoch 89
by the harness cleaning up an orphaned terminal session (no OOM, no GPU fault) and
resumed from the epoch-88 checkpoint; optimizer, schedule and history restore, so only
the shuffle of the last 12 epochs (lr ≤ 3.5e-6) differs from an uninterrupted run.

**Also in this branch: CRUW migrated to the sample-adapter mechanism.** T-RODNet is
the first consumer added through `datasets/adapters/cruw.py`; the six RODNet and three
RECORD variants moved with it. Byte-level goldens captured on the unrefactored code
(`tests/test_cruw_golden.py`, 27 model×split cases) pass, and a 10-epoch RODNet_CDC
rerun through the adapters (`RODNet_CDC_CRUW_Oct-19-2026_18:28:54`) scored 39.9 / 44.8
against the August run's 36.9 / 42.8 (final epoch) and 38.5 / 44.6 (best-val), with
the same epoch-by-epoch val curve.

### RADDet: audited before Phase 2 (2026-09-01), two defects found and fixed

RADDet has **no reference weights**: the pre-trained `.ckpt` the README offers
(`README.md:214`) is a Google Drive link that now 404s. So the weights-through-our-pipeline
check that closed CARRADA and DAROD/RADDet is unavailable, and this pair falls under the
"when no trained reference weights exist" procedure above. All six steps were run against
a pristine worktree of `ZhangAoCanada/RADDet@75f4603` at `/home/kangle/projects/raddet-pristine`
(the existing `RADDet` checkout is not usable as a reference — it carries uncommitted debug
`print`s, `batch_size` 3 → 1, and a `break` after the first training batch).

| step | result |
|------|--------|
| 1. Config against the reference's | schedule mismatch (below) and a single-vs-four mAP threshold gap (both now closed); the other 12 fields match, including all four `global_*` constants, `max_boxes_per_frame` 30, the six anchors and every IoU threshold |
| 2. Ground-truth distribution | no clamping: 0.6% of boxes in the extreme range bins, 1.3% in azimuth, all 6 anchors used |
| 3. Do consecutive samples differ? | 0/11 identical inputs, 0/11 identical labels |
| 4. Structural diff of model and losses | **two defects, both fixed** |
| 5. Round-trip the evaluator on ground truth | mAP 0.9958 |
| 6. Small-subset overfit | mAP@0.1 0.015 → **0.909** in 400 steps |

**Defect 1: `boxDecoder` transposed the range and azimuth grid offsets.** The reference's
`tf.transpose(perm=[1,0,2,3,4])` (`head_YOLO.py:49`, commented "*swap axes seems necessary,
don't know why*") exists solely to undo `tf.meshgrid`'s default `indexing='xy'`, which
returns `(g1, g0, g2)`. Our port switched to `indexing="ij"`, which does not swap, and kept
the transpose — so it re-introduced the swap it was written to cancel, storing offset
`(j, i, k)` at cell `(i, j, k)`. Shape-preserving on the square 16×16 range/azimuth grid, so
nothing raised, and only the 64 diagonal cells of 1024 were correct. It is fatal to box
regression rather than merely inaccurate: decode is `(sigmoid(raw_xyz) + grid) * strides`
while `encodeToLabels` placed the box at cell `(i,j,k)` because its true range lies in
`[i*stride, (i+1)*stride)`, so fitting it needs `sigmoid(raw_x) = i - j + frac`,
unsatisfiable for `i != j`. Objectness and class still train, so it presents as bad
localisation, not a crash.

**Defect 2: `categoryLoss` used torch's default `reduction='mean'`.** The reference's
`tf.nn.sigmoid_cross_entropy_with_logits` (`loss_functions.py:55`) is elementwise and is
masked by `gt_conf` *before* the sum in `lossYolo`. Ours returned a scalar, so `gt_conf *
scalar` masked nothing, the class axis collapsed to 1 (making `lossYolo`'s sum run over the
wrong shape), and the gradient at the cell owning the object was scaled by `1/N` with
N = 36,864 instead of by its own `gt_conf`. Measured on a single-object frame: the reference
gives gradient to exactly the 6 class logits of the positive cell; ours gave it to all 1152
elements with the positive cell at 0.5% of correct magnitude. `focalLoss` beside it already
passed `reduction='none'`.

Everything else in the model and loss path was compared and matches: backbone repeat/expansion
schedule and max-pool downsampling, the YOLO head's two convolutions and channel-split reshape,
`lossYolo`'s three-term aggregation, `yolo1Loss`, `focalLoss` including its broadcasting, and
`iou3d`. Anchors match exactly once the reference's own `sortAnchorBoxes` (`loader.py:34`,
by 3D area) is applied — the raw `anchors.txt` order is not what `train.py` uses.

**Step 6 is what makes those two findings actionable**, and it doubles as their control. Same
seed, same 20 samples, same 400 steps, only the fixes differing:

| step | fixed mAP@0.1 | pre-fix mAP@0.1 | fixed loss | pre-fix loss |
|------|---------------|-----------------|------------|--------------|
| 100 | 0.1913 | 0.0000 | 5.24 | 1389.74 |
| 200 | 0.6605 | 0.0219 | 3.22 | 1287.79 |
| 400 | **0.9092** | 0.0513 | 2.24 | 1999.72 |

Pre-fix the loss oscillates between 537 and 2400 with no trend — the signature of defect 1,
whose squared position error cannot fall below the cell-offset mismatch. **A RADDet run before
these fixes could not have produced a meaningful number at any training length.**

The round-trip (step 5) does *not* cover either defect: it feeds the label tensor straight into
`yoloheadToPredictions`, bypassing `boxDecoder` and the loss entirely. It validates NMS and mAP
only. Its 0.42% shortfall was traced and is not a port bug — `nms_iou3d_threshold: 0.1` is
aggressive enough to merge two genuine pedestrians 12 range bins apart (frame 6433), which is
the reference's own configured threshold and therefore a recall ceiling inherited from the
published setup. It is why `person` scores 0.980 while every other class is exactly 1.000.

#### The reference has no reachable stopping point, and no model selection

`total_train_batches = 1000 * 7314 // 3 = 2,438,000` steps with no early stopping; `train.py`
saves every 10,000 steps until the process is killed, `CheckpointManager(max_to_keep=3)` deletes
everything older, and `evaluate.py:80` restores `manager.latest_checkpoint`. So the published
number is whatever the weights were when the author stopped the process — a manual judgement
recorded nowhere in the repo, README or config. The author could watch `ap/ap_all`, logged to
TensorBoard every 10,000 steps, and stop on a plateau.

Two consequences. The **selection rule is exactly reproducible** — final weights, no best-val
checkpoint — and `RADDet_train_config_reference.json` therefore carries neither `test_checkpoint`
nor `early_stop`, matching the DAROD convention already used by rows 11/12. But the **run length
is not recoverable**, and it is the one parameter that cannot be made faithful. 100 epochs
(243,800 steps at batch 3) covers the 24.6-epoch warmup plus 61 decay events, matching the length
used for the RECORD family; since the reference does not early-stop, treat it as a floor rather
than an unbiased match.

`RADDet_train_config_reference.json` also needed a new scheduler type. The reference's rule,
`lr <- lr_end + decay*(lr - lr_end)` every 3000 steps after 60,000, is geometric decay of
`(lr - lr_end)` and so approaches 1e-5 as a **floor**; `StepLR` and `ExponentialLR` both scale
lr itself and decay to zero. At epoch 100: reference 1.72e-05, `StepLR(10, 0.9)` 3.49e-05,
`ExponentialLR(0.9)` 2.66e-09. The `decay_to_floor` type added to `train()` is a `LambdaLR` on
the step count stepped per iteration, verified bit-identical to the reference's loop at every
step from 1 to 243,800. Batch size is set to the reference's 3 rather than the repo default's 4,
because the schedule's thresholds are counted in optimizer steps: 60,000 steps is 24.6 epochs at
batch 3 and 32.8 at batch 4.

#### Reporting all four published thresholds

The paper reports AP at four IoU thresholds (0.1 0.764 / 0.3 0.563 / **0.5 0.251** / 0.7 0.059)
while `RADDet_evaluation` took a scalar. The reference splits this across two config blocks —
`MODEL.mAP_iou3d_threshold` 0.3 for training and validation, `EVALUATE.mAP_iou3d_threshold`
`[0.1, 0.3, 0.5, 0.7]` for the final test — and `RADDet_train_config_reference.json` now mirrors
that with `mAP_iou3d_threshold` plus `mAP_iou3d_threshold_list`.

**All four come from one forward pass.** The IoU threshold affects only TP/FP matching, never the
predictions, so inference and NMS are shared and only `mAP()` repeats — one accumulator per
threshold, which is exactly what the reference does (`evaluate.py:118-146`). The extra cost is
negligible against a forward pass over 2032 test frames.

The scalar stays primary: it drives the val curve and the `mAP` key that `train()`'s
checkpoint-selection contract reads, so nothing about selection changes. The others arrive as
`mAP@0.1` / `@0.5` / `@0.7`, alongside per-class `AP@<thr>_<class>` — the per-class figures were
already being computed and discarded, and every other closed row in this campaign is argued per
class. `train()` carries all of them into `val_eval.csv` and `test_eval.csv` with no changes,
since the val path pads extras into `history` and the test path writes the returned dict
wholesale. Verified on the ground-truth round-trip: 1.0 at every threshold.

Classes absent from a split score 0.0 rather than NaN, which is the reference's convention
(`evaluate.py:151-154`) and matters for RADDet's tail classes — motorcycle appears once in 300
sampled training frames.

#### Closed: RADDet / RADDetDataset (Phase 2) — reproduced, above the published figures

100 epochs on the `_reference` config, official split, final-epoch weights. Test:

All figures below are percentages, matching the results table (the paper publishes fractions;
these are those ×100).

| threshold | ours | published | delta |
|-----------|------|-----------|-------|
| AP@0.1 | **77.76** | 76.4 | +1.8% |
| AP@0.3 | **60.42** | 56.3 | +7.3% |
| AP@0.5 | **30.37** | **25.1** | **+21.0%** |
| AP@0.7 | **6.94** | 5.9 | +17.6% |

Per class:

| class | @0.1 | @0.3 | @0.5 | @0.7 |
|-------|------|------|------|------|
| car | 86.4 | 72.1 | 41.8 | 10.1 |
| person | 68.1 | 35.3 | 5.3 | 0.2 |
| truck | 61.4 | 55.0 | 30.1 | 6.8 |
| bus | 57.9 | 52.6 | 39.5 | 5.3 |
| bicycle | 49.7 | 32.5 | 3.8 | 0.0 |
| motorcycle | 9.5 | 9.5 | 4.8 | 0.0 |

Bus and motorcycle are non-zero on test (52.6 and 9.5 at @0.3) after sitting at 0.0-11.1
throughout validation — the test split carries more instances of both than the 812-frame val
split does. Do not read the validation per-class figures for the tail classes as evidence about
the model.

**No reference anchor was run, and it is not needed here.** The anchor exists to answer one
question: when our number falls SHORT of the paper, is that our port or the artifact? RECORD /
CARRADA RD needed it (53.15 ours against 58.10 published; the reference itself scored 53.19, so
the deficit was not ours). RADDet has no shortfall to explain — we are above the published
figure at every threshold, on the official split, under the reference's own selection rule. A
reference run scoring lower would only add a second data point about the artifact; one scoring
higher would not invalidate ours. It would also have cost a second 320 GB copy in the original
`train/RAD/part_N/` layout plus a TensorFlow rebuild, since the pristine repo's venv has a
Python 3.12 interpreter against 3.10 site-packages.

**Selection was reference-faithful and cost almost nothing.** No `test_checkpoint` key, so
`train()` tested the final-epoch weights, matching `evaluate.py:80`'s `manager.latest_checkpoint`.
Validation plateaued rather than decaying — best 61.56 at epoch 41 against 59.80 at epoch 99,
so final-weights testing gave up 1.8 points relative to the best epoch:

```
val mAP@0.3 every 10 epochs:
23.2  56.5  59.1  60.0  60.0  60.9  61.1  60.2  60.9  60.0
```

That is worth recording because it was not obvious in advance. Around epoch 13 the losses
diverged sharply and kept going — train 7.46 → 0.30 while val went 4.93 → 5.07, a 17x gap by the
end — which looked like overfitting that the stop-whenever convention would punish. It never
reached AP. The loss is dominated by box-regression and focal terms over all 6144 cells, while AP
depends only on the handful of confident detections surviving NMS. **On this pair the val loss is
not a usable proxy for the metric**, in either direction.

There is also essentially **no val-to-test drop** (val @0.3 59.80, test 60.42), unlike MVRECORD's
8.9-point fall on RD. So the result does not rest on a lucky checkpoint draw.

Two caveats stay attached. The comparison is against the **paper**, not against the reference on
our data, which everywhere else in this campaign has been the more honest target. And **run length
is a free parameter**: we ran 100 epochs where the reference nominally runs 1000 and stops by
hand, so a different stop could have landed elsewhere.

`docs/phase2-raddet.json` again reads `verdict: suspect` from `smoke_sweep`'s first-vs-last
val-loss heuristic. Here the heuristic is correct on its own terms — val loss genuinely rose from
4.93 to 5.07 — and still says nothing about the result, for the reason above.

#### If an anchor is ever wanted for this pair

Not needed for row 10 (above), but recording what it would take. Running the reference needs a
**second 320 GB copy**: our `RADDet_reform_dataset` has a flat `RAD/*.npy`, while the reference
globs `train_set_dir/RAD/*/*.npy` (two levels, `part_1` … `part_5`) and reads train and test from
separate directories. That layout exists only as `RADDet_dataset` on the USB disk (8126 train +
2032 test). The pristine repo also needs its environment rebuilt — the venv at
`/home/kangle/projects/RADDet/.venv` has a Python 3.12 interpreter against 3.10 site-packages, so
TensorFlow will not import. The one thing it would buy is neutralising the unknown-duration
problem, by running both sides for the same number of steps.

### RAMP_CNN / UWCR: the published number cannot be reproduced — 78% of the data was never released

Established 2026-09-02, while preparing the Phase 2 run. This is a stronger statement than
the earlier "RAMP_CNN's split is still unknown", and it is confirmed by the author rather
than inferred.

**The public dataset is a fifth of the paper's.** RAMP-CNN Table II gives 67,198 training
and 25,098 test frames, 92,296 total. The public UWCR release contains 19,754 frames across
the 22 sequences in `Automotive/` (~898 each), plus one further sequence available only in
converted form — 20,677 over 23 sequences. Its own README states *"The overall dataset
contains approximately 19800 frames"*, so this is the release's full extent, not a partial
download. The paper's **test set alone (25,098) is larger than everything released.**

The paper is not describing the public dataset: its §VI-A is titled "UW Camera-Radar (CR)
Dataset" and lists parking lot, curbside, campus road, city road and freeway, while the
release README says collection was done "on the campus, road, and parking lot **during the
daytime**". The paper's fourth test scenario is 3,600 **nighttime** frames, which the
release therefore cannot contain.

**The author confirms both points** in
[issue #16](https://github.com/Xiangyu-Gao/Radar-multiple-perspective-object-detection/issues/16)
(2023-10-12), answering exactly this question:

> "Yes, this raw radar set is a subset of the data where the RAMP-CNN model is trained &
> tested."

and, asked which of the 23 sequences are train vs test:

> "'2019_05_28_pm2s012' is in the testing set for ramp-cnn paper, and others are in the
> training set."

**That is the decisive fact for this row.** All 22 sequences in `Automotive/` — everything
we have in raw form — are *training* data for the published model. The only public test
sequence is `2019_05_28_pm2s012`, which exists solely in `train_test_data/` as pre-converted
RA/RV/VA slices (1,800 frames, a single parking-lot sequence). So:

- Any test set built from the 22 sequences is made of frames the published model trained on.
  A number from it compares generalisation against memorisation and is not interpretable
  against 81.23 / 84.25 in either direction.
- Table III's four-scenario breakdown (ParkingLot / Curbside / On-road / nighttime) cannot be
  reconstructed: one public test sequence, and no nighttime data at all.
- The repo's `config.py` demo split — train `2019_04_30_pbms002`, val/test `2019_05_28_pm2s012`
  — is not arbitrary after all. It is a one-sequence-per-side miniature of the real split,
  drawn from the correct sides.

It also explains the Phase 1 figure. mAP 0.9295 with mAR reaching 1.0000 on a single
validation sequence was recorded as "probably an easy one"; the likelier reason is that the
3-sequence smoke split validated on training-side data.

**Consequence for row 13: not reproducible — data not released.** What remains possible, and
worth doing, is a measurement of *this repo's* RAMP_CNN on a split we define over the 22
sequences. That is internally valid — our model would train and test on genuinely disjoint
data — and useful as a regression baseline. It is **not** a comparison to the published
figure, and must not be written as one.

#### Defect found while preparing that run: unmapped class ids crashed the loader

`UWCR.__getitem__` did `category = self.label_map[int(obj[1])]` with no membership check,
and the enclosing `except` catches only `pd.errors.EmptyDataError`, so a `KeyError`
propagated and killed the dataloader. The public labels contain class id **1**, which is in
neither our `label_map` nor the reference's `release_dataset_label_map` (both are
`{0, 2, 3, 5, 7, 80}`): 100 rows across 99 frames in `2019_05_09_mlms003` and 6 rows in 6
frames of `2019_05_09_bm1s007`. **Two of the 22 sequences could not be loaded at all.**

The reference guards this explicitly (`utils/convert_annotations.py:50-53`):

```python
if int(line[1]) in release_dataset_label_map:
    type_ = release_dataset_label_map[int(line[1])]
else:
    continue
```

Ours now skips the same way, decrementing `n_obj`. Never seen before because every run so far
used a 3-sequence smoke split that excluded both sequences — the same reason the split
question went unexamined. Unmapped class *names* were already safe: `get_class_id` returns
-1000, the reference's `noise` id, for `motorbike` and `bus`.

Measured class distribution over the 22 sequences, after decoding the ids
(`0` pedestrian, `2` car, `3` motorbike, `5` bus, `7` truck, `80` cyclist):

| class | instances |
|-------|-----------|
| pedestrian | 14,849 |
| car | 12,369 |
| cyclist | 10,253 |
| truck | 454 |
| motorbike | 194 |
| unmapped (id 1) | 106 |

#### The paper's schedule also differs from the repo's config

Worth recording because `RAMP_CNN_train_config_reference.json` was built from `config.py`
before the paper was read:

| | paper §VI-C | repo `config.py` |
|---|---|---|
| cyclic lr range | 5e-6 → **5e-5** | 1e-6 → 1e-5 |
| cycle duration | 860 iterations | 430 up + 430 down = 860 ✓ |
| phase 1 | batch **5**, 10 epochs | batch 3 |
| phase 2 | StepLR from 5e-6, ×0.2 every 5 epochs, batch **4**, 24 epochs | absent |
| total epochs | **34** | 100 |

The paper uses a two-phase schedule at 5× the repo's learning rates, with the batch size
changing between phases. Since the published number is unreachable anyway, the schedule
choice for our own baseline is a judgement call rather than a fidelity requirement — but the
config should not claim to be the paper's when it is the repo's.

#### Preprocessing audit (2026-09-05): our slices are faithful; `train_test_data` is not a valid target

The first full-split run reached val mAP **0.1586**, far under the smoke split's 0.9295, so
the preprocessing was audited end to end against the reference. **It came back clean**, and
the negative result is recorded here so it is not re-derived.

What was checked, in order:

1. **Code.** `get_ADC` / `get_RD` / `get_RA` / `get_RV_VA_slice` are line-for-line ports of
   `slice3d.py`'s `range_fft` / `produce_RA_slice` / `produce_RV_slice` / `produce_VA_slice`.
2. **Input.** The reference ships nine frames of its own `slice_sample_data` (key `adc_data`,
   already in `samples × antennas × chirps`) that overlap the public `Automotive/` raw ADC
   (key `adcData`, `128×255×4×2`). Our `get_ADC` conversion — concatenate the two TX groups
   along the antenna axis, then swap axes — is **bit-identical** to the reference's own
   array on all nine: `array_equal = True`, `maxabsdiff = 0`.
3. **Output.** Running the reference's **unmodified** `slice3d.py` on the reference's **own**
   `slice_sample_data` and comparing against the reference's **own** shipped
   `train_test_data` gives, for frames 2–10: RA *r* 0.137–0.442, RV 0.367–0.495, VA
   0.954–0.969 — essentially the same numbers ours produces.

**The reference's own code, on its own input, does not reproduce its own shipped slices.**
`train_test_data` is therefore a **different vintage of the data** and cannot serve as a
verification target for anything. `slice_sample_data` is the valid artifact, and against it
we are exact.

Two hypotheses were pursued and killed along the way; both are dead ends, do not revisit:

- **Missing DC-offset removal in `get_ADC`.** The ADC carries a large DC term (52.2% of
  range-FFT energy in bin 0), and removing it raises agreement with `train_test_data` to
  0.974–0.997. That is real but it is **agreement with the wrong vintage** — the reference
  does no such thing, and adding it would make us *diverge* from the reference. A fix, a
  cache purge and a discarded 73-epoch run were all proposed on this basis and all withdrawn.
- **`filter_static`** (`slice3d.py:81`, subtract the mean over chirps). Defined but **never
  called anywhere** in the reference repo; `main()`'s "after the denoise" docstring is stale.

Methodological note: an "ours vs reference" correlation is uninterpretable without a
"reference vs reference" control. RA complex-correlation of 0.022 was briefly flagged as a
defect; the control showed the reference's own *adjacent frames* correlate at 0.067 in the
same 2-channel representation, because phase decorrelates frame to frame. **Magnitude**
correlation is the meaningful statistic for RA. Run the control first.

#### Defect found by that audit: inputs were never normalized

The audit did turn up one genuine gap. **The reference standardizes every input view before
the forward pass, unconditionally** — `train_dop.py:91` sets `Norm = True` as a plain local
in `main()` (no flag, no config override) and applies it to all three views at `:228-234`;
`test.py` does the same at `:17` and `:177-189`. The `finished ra normalization` lines in the
reference README's own sample output (`:98-99`) are that code running.

Our `UWCR.__getitem__` did not. Its only `normalize_*` call was `normalize_confmap`, which
acts on the **label** confmap, not the input.

The reference's `config.py` constants **cannot be copied**, because they describe the
`train_test_data` vintage: its slices sit at RA std ~0.008 / RV ~0.035 / VA ~0.024, while
slices computed from the public raw ADC are **~11,000× larger**. So the constants were
re-derived on our own 14-sequence train split (`scripts/compute_uwcr_norm.py`, over 819M RA
elements; val and test excluded):

| view | our mean | our std | ref `mean1`/`std1` | mean÷std, ours vs ref | std ratio |
|------|---------:|--------:|-------------------:|:---------------------:|----------:|
| RA | -0.3253 | 328.464 | -5.052e-05 / 0.029227 | -0.0010 vs -0.0017 | 11,238× |
| RV | +442.468 | 1843.665 | 0.038193 / 0.167542 | 0.2400 vs 0.2280 | 11,005× |
| VA | +715.317 | 1172.141 | 0.062615 / 0.087097 | 0.6103 vs 0.7189 | 13,458× |

The **ratios** matching the reference's is the check that ours are the same quantity, merely
at a different scale (RA's ~0 mean is what real/imag parts of a complex FFT must give).

Wired as `input_norm` in both `RAMP_CNN_train_config.json` and `..._reference.json` — the two
are kept identical here because this is a pipeline correctness fix, not a choice of recipe —
read in `prepare_for_train`, applied in `__getitem__`. One constant pair per **view**, pooled
over that view's channels, matching the reference's `data[ib, :] = (data[ib, :] - mean1) /
std1`, which hits RA's real and imaginary channels together. Deliberately **not** folded into
`get_frame_slices`: that source is hashed into `frame_cache_dir`, so every constant change
would invalidate 7.5 GB of cache for an elementwise affine map. The reference likewise
normalizes at consumption, in the train loop.

The reference keys two constant sets off capture month (`date_detail < 9` picks `mean1` vs
`mean2`). Every UWCR sequence we hold is month 04 or 05, so one set covers all of them and
that branch has nothing to select between. Note the `config.py` comments call these "old
data" / "new data", but the *selector* is calendar month.

**Consequence for the paused run.** The epoch-73 checkpoint (val mAP 0.1586) was trained on
unnormalized inputs whose scale is ~11,000× what the network now receives. It **cannot be
resumed** across this change — its first-layer weights are fit to the old scale. It is
instead the **unnormalized control** for a clean A/B against a fresh normalized run.

`README:73` shows the reference's own sample run printing `No data augmentation`, so the
missing augmentation port is *not* a fidelity gap against the repo's default recipe and is
deprioritized.

#### Result of the A/B on VALIDATION — superseded on 2026-09-09, see the test result below

> **The verdict in this subsection does not survive the held-out test split.** It is kept
> because the reasoning is sound on the evidence it had, and because the way it failed is the
> point: a validation-only A/B on this split reversed direction when the held-out capture date
> was finally scored. Read the "Test-set result" subsection that follows before quoting
> anything here.


Two runs on the identical 14/3/5 date-held-out split and the identical `_reference` config,
differing **only** in whether `input_norm` is applied:

| run | epochs on disk | best AP | best AR | at epoch |
|---|---|---|---|---|
| control, unnormalized (`Sep-02-2026_21:57:47`) | 13–73 | 0.1586 | 0.2668 | 63 |
| normalized (`Sep-05-2026_12:38:17` + `_20:12:04`) | 0–40 | **0.1799** | **0.3179** | **28** |

Compared over the **same 28 epochs** (13–40), which is the honest comparison since the runs
are of different length:

| | control | normalized | delta |
|---|---:|---:|---:|
| mean AP | 0.1286 | 0.1367 | **+6.3%** |
| mean AR | 0.2226 | 0.2399 | **+7.8%** |
| peak AP | 0.1579 | 0.1799 | **+13.9%** |
| peak AR | 0.2633 | 0.3179 | **+20.7%** |

**Verdict: the fix is real but modest.** Take the mean columns as the effect size (+6–8%);
the peak columns flatter it, because a peak over 28 noisy epochs is partly a max-statistic.
The clearer win is *when*: the normalized run beat the control's 73-epoch best at epoch 28,
and its val loss bottomed at 16.60 (epoch 10) against the control's best visible 18.53
(epoch 13; the control's epochs 0–12 are no longer on disk, so its true minimum is unknown).

**This closes the open question above, negatively.** Normalization does not account for the
0.9295 → 0.1586 gap — it moves the number by ~6%, not by 6×. The gap is the split. The 0.9295
was measured on a 3-sequence smoke split validating on training-side data (see above); ~0.15
is what this implementation scores when train and val are genuinely disjoint by capture date.
Nothing further is attributable to a preprocessing defect, and the preprocessing audit found
none.

#### Test-set result (2026-09-09): the normalization A/B reverses, and row 13 becomes a test number

Neither RAMP_CNN run ever reached its test phase — both were killed mid-schedule — so
everything above is validation. The held-out test split was always there (555 windows, the 5
sequences of capture date `2019_05_29`, absent from train and val), it had simply never been
scored. Evaluating the stored checkpoints directly, each under **its own** `train_info.txt`
(the control predates `input_norm`; scoring it under the current config would feed it inputs
~11,000× smaller than it trained on):

| checkpoint | val AP / AR | **test AP / AR** | val → test |
|---|---|---|---|
| normalized, epoch 28 (val-best) | 17.99 / 31.79 | **13.09 / 29.31** | −27% / −8% |
| normalized, epoch 40 (final) | 12.92 / 21.46 | **6.07 / 15.74** | −53% / −27% |
| **control (unnormalized), epoch 63** | 15.86 / 26.68 | **17.86 / 34.22** | **+13% / +28%** |

**Two findings, in order of how much they change what is written above.**

**1. The validation A/B does not replicate on held-out data; its direction reverses.** On
validation the normalized run wins (17.99 vs 15.86). On test the control wins on both metrics
— 17.86 vs 13.09 AP (+36%) and 34.22 vs 29.31 AR (+17%). The "+6.3% mean AP / +7.8% mean AR"
above, and the claim that normalization is "validated" by it, are **withdrawn**. Note also the
shapes: the control's score *rose* from val to test while the normalized run's *fell* by more
than a quarter.

State the limits honestly rather than claiming the opposite result. This is a two-checkpoint
comparison, not a matched experiment: the runs trained for different lengths (63 vs 28
epochs), both checkpoints are val-selected and so carry selection bias, and the test set is a
single homogeneous capture date. The defensible claim is **"the validation-based A/B does not
replicate on held-out data, with the direction reversed"** — not "normalization hurts".

**What this does NOT undo: the normalization fix itself stands.** The reference standardizes
every input view unconditionally (`train_dop.py:91`, `:228-234`; `test.py:17,177-189`), and
matching it is a fidelity requirement, not a claim contingent on a score. What was wrong was
the evidence offered for it, not the change.

**2. Final-weights selection would halve this pair's number.** Epoch 28 tests at 13.09 against
epoch 40's 6.07. Everywhere else in this campaign the final-epoch convention is harmless
because those runs trained to convergence; this one was stopped mid-plateau, well into the
overfitting phase described below, so its final weights are not representative. Row 13
therefore reports epoch 28 — and says so — rather than silently applying a convention that
does not fit.

Row 13 now carries the **test** figure (13.09 / 29.31). It remains a regression baseline for
this repo, not a reproduction: the split is ours, and 78% of the paper's data was never
released.

#### Stopped at epoch 40 of 100, deliberately

Both runs plateau, and the second one was ended early on that evidence rather than run to the
config's 100-epoch ceiling:

```
epoch  28      29      30      31      32      33      34      35      36      37      38      39      40
AP     0.1799  0.1369  0.1406  0.1258  0.1479  0.1455  0.1402  0.1110  0.1476  0.1391  0.1517  0.1480  0.1292
loss   20.99   25.06   26.10   29.63   23.58   27.79   26.98   25.38   28.68   27.79   25.97   28.06   33.42
```

Twelve epochs with no new best (closest 0.1517 at 38, 16% under epoch 28), while **val loss
climbs monotonically** — minimum 16.60 at epoch 10, and epoch 40's 33.42 is the worst of the
run, double that. AP flat while val loss doubles is overfitting that began well before epoch
28, which also means epoch 28 is likely a favourable fluctuation on a plateau rather than a
peak still being climbed.

The control is the reason to trust that reading rather than suspect a premature stop: it
*did* run past this point, for 60 more epochs, and stayed flat the whole way — 0.11–0.158
from epoch 22 to 73, with its "best" 0.1586 at epoch 63 barely above its 0.1579 at epoch 35.
A late peak there was noise on a plateau, not late learning. Two independent runs showing the
same plateau is enough; the remaining 60 epochs would have cost ~10 h to re-confirm it.

There is **no early-stopping criterion in the RAMP-CNN recipe** to appeal to here (unlike
RECORD/CRUW, which stopped itself), so this is our judgement call and is recorded as one.

#### Row 13 status: measured, not reproduced

**AP 0.1799 / AR 0.3179**, RAMP_CNN on UWCR, our 14/3/5 date-held-out split, epoch 28 of a
41-epoch run under the repo's `config.py` recipe with inputs standardized on train-split
constants.

This is **not** comparable to the published 81.23 / 84.25 and must never be quoted against
it — 78% of the paper's data was never released and all 22 public raw sequences are on the
*training* side of the published split (see the top of this section). The number is a
**regression baseline for this repo**: a value this implementation reaches on a split we
control, against which future changes can be measured. Row 13 stays flagged *not
reproducible — data not released*.

### RadarCrossAttention / CARRADA: audited before Phase 2 (2026-09-07), six defects fixed

Row 14, published **0.94 / 0.86** (mean AP at 2 m / 1 m over pedestrian, cyclist, car;
per class 0.91/0.80, 0.93/0.88, 0.97/0.91 — RadarCrossAttention Table I, "Bivar Cross
Atten. + offset loss"). Reference: `ravikothari510/crossattention_radar_detector`,
checkout `/home/kangle/projects/crossattention_radar_detector` @3e2169c. Same pre-run
checklist as the other families; the audit found the port faithful in its arithmetic and
wrong in its configuration.

**Faithful, verified:**

- **Network** — `Encode`/`Decode`/`CrossAttention` verbatim; 11.624 M parameters at T=1 on
  both sides. The paper's 11.05 M differs by 0.52 M, which is exactly the affine
  parameters of `LayerNorm([256,32,32])` (2 × 262,144) — the published count evidently
  excludes them or that layer ran without them. The code is what we follow.
- **Losses** — focal (α=2, class weights 1.33/2.81/0.86 broadcast to 3×256×256),
  centre-offset BCE on the 9×9 patch, heading MSE: verbatim, weights 1/1/10.
- **Annotations** — `new_gt_anno.json` is byte-identical to the reference's `gt_anno.json`
  (md5 `c7f74ae0…`). 30 sequences, 6,074 tracks, 8,600 instances (paper: "almost 9000").
  Every frame inside a track interval is annotated, so T=1 windows never miss a label.
- **GT maps** — `bi_var_gauss`/`get_center_map`/`get_orent_map`/`plain_gauss`: AST-identical
  to `bi_gauss_update.py` (the reference's `bi_var_gauss` returns all three; ours is split).
- **Preprocessing** — RD (double flip, DC columns 31:34 zeroed) and AD (inverse FFT over
  range of the RAD cube, power, sum, dB, transpose, DC rows zeroed): line-for-line
  `dataset_process.py`. Note the AD map is **not** CARRADA's shipped `angle_doppler_*`
  (MVRECORD's input): different recipe, shape and scale, correlation 0.74 on a test frame.
- **Evaluator** — `create_default`, `peaks_detect`, `pol2cord`, `distance`, `orent`
  AST-identical to `val_utils.py`; `metrics_center`/`validation` differ only in the tensor
  guards below. `distribute`, `update_peaks` and `association` are **vectorized
  re-implementations** with identical output (`test_rca_association.py` compares them
  against verbatim copies of the reference loops on random peak sets, ties included):
  the reference's NMS is O(n²) in Python with a `torch.cat` per element, and the first
  launch of this run spent 25 min on 17 of 83 val batches at epoch 0 — an untrained model
  has ~10⁴ peaks per frame above the 0.1 threshold — with an ETA of 1 h 40 for one
  epoch's validation. The only semantic addition is that `update_peaks` clamps a shifted
  peak to the map, where the reference would raise (past row 255) or silently wrap
  (row −2 → 254, i.e. 0.4 m → 50 m).
  The AP is `np.trapz(P, R)` with **no (0, 1) origin point**, so a perfect detector scores
  1 − 1/G — ≤0.2% on this test split (G = 471–1222 per class). Kept: it is the published
  metric. Pinned exactly by `test_rca_evaluation.py`, which also checks that a 1.37 m
  localisation error passes at 2 m and fails at 1 m.

**The reference as shipped cannot have produced the published numbers.** Two expressions
evaluate a whole tensor as a bool: `noise_add`'s `if random.choice([0,1]) and r_map != 0`
(training augmentation — raises on every noisy draw; 12 of 20 calls in a direct test) and
`metrics_center`'s `if pred_c != 0` (evaluation with centre-offset on — raises on the first
batch). Neither has ever been legal Python/torch. So the published run used a different
version, and "reference on our pipeline" is not available for this row. Three more
reference-side facts that bound what a comparison can mean:

- **Split is unrecoverable and leaky.** `data_split.py` shuffles *frames* 70/15/15 with
  `random.seed(1)` placed *after* `main()`; `test.py` scores the val list. Frames from every
  sequence land in every split. The paper says 70/10/20.
- **T=1 input lags its label by 4 frames.** Tracks are stacked oldest-first and the loader
  takes `[:no_frames]` — the oldest — while the target is the track's last frame. Whether
  the published run had this cannot be known; we feed the labelled frame (zero lag), which
  is what "single radar frame" means.
- **Normalization constants are stats of the un-zeroed maps.** Measured on 308 train frames:
  RA 669.9/1459.5 vs their 678.5/1352.6; RD/AD means 32.6/33.7 vs 34.1/35.3; the RD/AD std
  values match only before DC-zeroing (3.10/2.96 vs 2.93/2.07; zeroing inflates both to
  ~7.9). They describe our data — unlike RAMP_CNN's — and are applied to zeroed maps in the
  reference too, so they are copied verbatim.

**Six defects fixed on our side** (all silent — the Phase 1 smoke run scored 0.4367 car-only
through every one of them):

1. **Inputs were never normalized.** The branch carried a literal `# TODO: normalize`. Now
   `input_norm` in both train configs, applied per view in `__getitem__`.
2. **Wrong model.** `win_size 5` and `orientation true` are the paper's *heading ablation*
   (Table II, no AP published). Table I is "single radar frame", T=1, heading off — the
   reference CLI defaults. Model config and both train configs changed to T=1/off.
3. **Wrong schedule.** Batch 4 with StepLR(10, 0.9) against the reference's batch 16 with
   `ReduceLROnPlateau(patience=4, factor=0.1)` stepped on val sum-MSE, 80 epochs, AMP.
   `train()` gained a `plateau` scheduler branch (stepped after validation on a named
   evaluator metric) and config-gated AMP (`"amp": true`; 1.41 GiB / 0.27 s per batch-16
   iteration vs 2.72 GiB / 0.60 s fp32 — the reference trains under autocast+GradScaler).
4. **Evaluator crashed with heading off** — `torch.all(pr_o)` on the int 0 the network
   returns for a disabled head — and scored only 2 m. Now guards on `torch.is_tensor`,
   scores 2 m and 1 m in one pass, returns per-class APs and `val_mse` (the reference's
   plateau/checkpoint metric) as extras.
5. **No augmentation.** The reference's evident intent (N(0, 0.1) per view with p=½, after
   normalization, train only) is ported as `gauss_noise_std`. The paper's "horizontal flip"
   exists nowhere in the reference code and is not invented.
6. **AD maps recomputed every epoch** — 254 ms per frame (vs 10 ms to load the cube), so
   ~29 worker-hours over 80 epochs. Now content-addressed on disk
   (`carrada_ad_cache/<hash>`, 6,366 frames, 399 MB, built by
   `scripts/build_carrada_ad_cache.py`); `__getitem__` measures 4–6 ms.

**Split and expectation.** Official CARRADA 18/6/6 by sequence (`data_seq_ref.json`, the
split the RECORD/MVRECORD/DAROD rows used): **3,815 / 1,326 / 1,225** frames at T=1, with
5,082 / 1,676 / 1,842 instances. The published protocol trains and scores on interleaved
frames of the same sequences; ours holds sequences out. **Expect our number below 0.94 /
0.86 for that reason alone**, and read the gap as protocol before implementation. Test-split
caveat: cyclists are scarce there (149 instances vs 1,222 pedestrians, 471 cars), so the
cyclist AP will be the noisiest of the three. A protocol-matched bracket — `splitmode
random [0.7, 0.15, 0.15]`, frame-level like the reference — is available if the official
number needs interpreting.

Pre-run checks: sample path verified on a train and a test sequence (normalized RA
−0.06/1.19, RD −0.38/2.69, AD −0.61/3.85 — the RD/AD std > 1 is the DC-zeroing, faithful;
noise on for train only; no heading target); evaluator round-trip on ground truth scores
exactly 1 − 1/G at both thresholds; official-split dataloaders build to the counts above.

#### Result on the official split (2026-09-07): 0.427 / 0.343 — not a match

`RadarCrossAttention_CARRADA_Sep-07-2026_21:50:01`, 80 epochs in 65 min, final weights,
test = the 6 held-out sequences (1,225 frames):

| AP | ped | cyc | car | **mean** |
|---|---|---|---|---|
| @2m, ours | 0.395 | **0.066** | 0.820 | **0.427** |
| @1m, ours | 0.311 | 0.060 | 0.660 | **0.343** |
| @2m / @1m, published | 0.91 / 0.80 | 0.93 / 0.88 | 0.97 / 0.91 | **0.94 / 0.86** |

Less than half the published mean, and the per-class shape carries the information: car
is close (0.82 vs 0.97), pedestrian is halved, cyclist has collapsed — 885 predictions
for 155 ground truths, precision 0.044. Val, also held out by sequence, agrees on the
shape (final ped 0.31 / cyc 0.22 / car 0.91).

**Checkpoint selection is not the explanation.** The best-val checkpoint (epoch 14, val
mAP 0.511 against 0.481 final) scores **0.417 / 0.338** on test — no better: car worse
(0.755), cyclist marginally better (0.094). The val peak did not transfer.

**The reference's schedule selected against the VRU classes.** Val trajectory:

| epoch | mAP@2m | ped | cyc | car | val MSE | lr |
|---|---|---|---|---|---|---|
| 9 | 0.381 | 0.153 | 0.114 | 0.876 | 3169 | 1e-4 (val focal loss minimum) |
| 14 | **0.511** | **0.424** | 0.252 | 0.856 | 2337 | 1e-4 |
| 17 | 0.460 | 0.316 | **0.281** | 0.868 | 2088 | 1e-4 |
| 30 | 0.448 | 0.251 | 0.184 | 0.908 | 1634 | → 1e-5 (first plateau drop) |
| 39 | 0.478 | 0.296 | 0.223 | **0.916** | 1363 | → 1e-6 |
| 79 | 0.481 | 0.312 | 0.221 | 0.909 | 1323 | 1e-8 (drops at 57, 62) |

`ReduceLROnPlateau` steps on the val sum-MSE between the sigmoid map and the Gaussian
target — the reference's own criterion — and that quantity **improved monotonically to
the end** while pedestrian AP fell 26% and cyclist 21% from their epoch-14/17 peaks. The
MSE is dominated by background pixels and the large, bright car blobs; the classes the
paper's contribution is about barely register in it. So the schedule kept the LR at 1e-4
for 30 epochs past the VRU peak (val focal loss had bottomed at epoch 9), then froze the
weights in the overfit state. Faithful to the recipe, and the recipe is what did it.

Test-split caveat, pre-registered above: 155 cyclists is a thin sample, and val (345
cyclists) scores that class at 0.22 rather than 0.07 — the test number is noisy as well
as low.

The published protocol scores val frames interleaved with training frames of the same
sequences; the bracket run on that protocol (`--split random`, 70/15/15 by frame, seeded)
is recorded below, and is what decides between "protocol" and "implementation".

#### Bracket on the published protocol: 0.888 / 0.784 — match; the gap is the split

`RadarCrossAttention_CARRADA_Sep-08-2026_07:37:29`, identical code, recipe and seed;
`--split random` = train.py's `random_split` over all 6,366 frames, 70/15/15 (4,456 /
954 / 956), seeded — frames of every sequence in every split, as the reference's
`data_split.py` does. Val/test built from their own instances (noise off); the `random`
splitmode used to slice all three splits from one train-prepared instance, which would
have evaluated on augmented inputs — fixed for every model as part of this run.

| test AP | ped | cyc | car | **mean** | vs published |
|---|---|---|---|---|---|
| @2m | 0.887 | 0.826 | 0.951 | **0.888** | −5.5% |
| @1m | 0.760 | 0.739 | 0.853 | **0.784** | −8.8% |
| published | 0.91 / 0.80 | 0.93 / 0.88 | 0.97 / 0.91 | 0.94 / 0.86 | |

Every class within 11% at both thresholds; val peaked at 0.899 (epoch 47) and finished at
0.895, with the plateau drops landing at 46/67/76 instead of 30/39/57 — the leaky split
keeps val MSE improving longer, so the schedule also behaves as the paper describes ("80
epochs ... where the evaluation losses were achieving the minima").

**Verdict: match on the published protocol (−5.5% / −8.8%, inside the ~20% rule).** The
same weights-recipe scores **0.427 / 0.343** when sequences are held out and **0.888 /
0.784** when frames are — a 2.1× difference that is entirely the split, since nothing else
changed. The published 0.94 / 0.86 is a frame-leaky number: neighbouring frames of the
same pedestrian sit on both sides of the split, and the VRU classes, whose radar
signatures barely generalise across scenes, lose almost everything once they cannot be
memorised (cyclist 0.83 → 0.07). Row 14 carries both figures; the official-split one is
what this model earns on unseen data and is the regression baseline for this repo.

### Closed: FFTRadNet / RADIal — reproduced against the paper (2026-09-09)

Row 1, and the least equivocal comparison in this file. `FFTRadNet_RADIal_Sep-08-2026_22:38:45`,
100 epochs in 11h22m, final weights, official split (6,231 / 986 / 1,035 frames), test
evaluation via the reference's `run_FullEvaluation` path — thresholds 0.1–0.9 at IoU 0.5.

| metric | ours | published | delta |
|---|---|---|---|
| AP | **95.55** | 96.84 | −1.3% |
| AR | **86.07** | 82.18 | **+4.7%** |
| freespace mIoU | **77.23** | 74.0 | **+4.4%** |
| range error | 0.122 m | 0.11 m | +11% (worse) |
| angle error | **0.112°** | 0.17° | **−34% (better)** |

**Why this row carries more weight than most.** Almost nothing had to be assumed. The data
is the paper's own — 8,252 labelled frames over 91 sequences, matching §5 exactly. The split
is the reference's own sequence-level 4-val/4-test list, not one we invented. The network is
AST-identical to the pristine reference block for block, the loss and encoder identical term
for term, and the schedule is the paper's (Adam 1e-4, ×0.9 every 10 epochs, 100 epochs). So
unlike RadarCrossAttention there is no protocol ambiguity available to explain a gap, and
unlike RAMP_CNN the data is not a fifth of the paper's. Compared directly against the paper,
like RADDet.

The one metric on the wrong side is range error (0.122 m vs 0.11 m), while angle error is a
third better than published — consistent with the paper's own claim that the MIMO
pre-encoder learns angle-of-arrival well, and not worth chasing.

**Final weights landed on a normal epoch.** The config sets no `test_checkpoint` and no
early stopping, matching the reference, so `train()` tested the weights at loop exit. Epoch
99's val mAP (0.9743) sits near the run's ceiling (0.9784 @ epoch 70) and its mIoU (0.6575)
is mid-band, clear of the periodic segmentation dips described below — so the convention cost
nothing here. Had epoch 99 landed on a dip, the honest move would have been to report it and
note the selection effect, not to substitute a better epoch.

**Two behaviours worth carrying forward, neither of which affected the result:**

- **Periodic segmentation excursions.** Epochs 8, 39, 52, 61, 79, 89 and 95 each show a sharp
  val-loss spike with mIoU dropping to 0.32–0.63 while mAP is untouched. The segmentation
  head carries a 100× loss weight, so one bad epoch there dominates the total. Every instance
  recovered within one or two epochs.
- **Val loss is again a poor guide to the comparison metric.** It bottomed at 100.86 (epoch
  12) and rose to 173.92 by epoch 99 — a 72% increase — while test AP came out at 95.55. That
  is now three pairs running: RAMP_CNN's rising val loss accompanied a genuine plateau,
  RadarCrossAttention's plateau criterion actively selected against the VRU classes, and here
  a 72% rise cost nothing at all. **Read val loss as a training-health signal only; the
  comparison metric is the one to select and stop on.**

### Closed: PointPillar and RadarPillar / ViewOfDelft — the July slice gates stand (2026-09-08)

Rows 15/16 were measured in July under the RadarPillar implementation spec, whose gates
were stricter than this campaign's: an overfit-one-batch gate, a decode/NMS round-trip
gate, then the like-for-like training run — which **failed first** (32.4 mAP, run
`…Jul-17-2026_18:51:11`), was diagnosed by source audit against the official VoD devkit
and the fthbng77 yamls (4 vs 7 point features, no augmentation, no FOV filter, no point
shuffle), passed on the aligned recipe, and was then lifted by a second audit that found
OpenPCDet's BatchNorm `eps=1e-3, momentum=0.01`. Full record:
`docs/superpowers/specs/2026-07-16-radarpillar-design.md:144-164`.

| | params | best (epoch) | **final weights** | published | run |
|---|---|---|---|---|---|
| PointPillar (base, wide backbone) | 4.835 M | 52.27 (64) | **49.41** | 46.7 | `RadarPillar_ViewOfDelft_Jul-19-2026_21:44:32` |
| RadarPillar (C=32, BN aligned) | 0.184 M | 50.72 (75) | **50.48** | 50.7 | `RadarPillar_ViewOfDelft_Jul-21-2026_08:56:49` |

Both under the same conditions as the other closed rows: official split, official metric,
one seed (3407), 80 epochs of the reference's OneCycle recipe, `radar_5frames`. The paper's
+4.0 RadarPillars-over-PointPillars delta does **not** reproduce on this pipeline — our
PointPillar already carries RadarPillars' data recipe (32 points/pillar, rotation
augmentation) that the paper's 46.7 baseline lacks, so the base lands 5.6 above the paper's
and the extras trade ~2 mAP for a 26× parameter reduction (spec §slice 4).

**Why no re-run in September.** The one open caveat was "predates the current code".
Checked: since the last of those runs, exactly one commit touches `view_of_delft.py`,
`radarpillar.py` or the VoD evaluator — the BN alignment, which *is* the Jul-21 run; the
evaluator is unchanged since 2026-07-18; and the shared trainer's default path (no AMP,
clock scheduler, `original` split) was proved bit-identical before and after this
session's changes. Nothing those numbers depend on has moved.

### Closed: RECORD / CRUW (Phase 2) — first ladder rung, ordering confirmed

Reference config (`--config-variant _reference`: lr 1e-3, batch 16, ExponentialLR-
equivalent ×0.9 every 10 epochs, early stop patience 7 on val loss, 100-epoch
ceiling), same stratified 30/5/5 split and fixed evaluator as the RODNet closure.
Trained 2026-08-22/23 in two segments (~41 min/epoch measured, joined via
`--resume`; a system suspend killed the first segment mid-epoch-7 and it resumed
from the epoch-6 checkpoint).

**Ended by the reference's own early stopping at epoch 17**: patience 7 on val
loss, whose best was epoch 9. Note the stop criterion had gone quiet while the
comparison metric had not — three of the last five epochs set val mAP records —
so 55.6 is a floor on what this config reaches, not a converged ceiling. This is
the documented val-loss/val-mAP selection deviation (see above), left faithful to
the reference rather than tuned.

Val trajectory (frame-filtered per-epoch eval, AP / AR):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|---|---|---|---|---|---|---|---|---|
| val mAP | 26.5 | 36.5 | 41.0 | 42.1 | 49.0 | 51.1 | 52.4 | 48.5 | 49.3 |
| val mAR | 54.6 | 63.6 | 59.4 | 64.0 | 65.0 | 65.8 | 65.2 | 66.2 | 63.9 |

| epoch | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|-------|----|----|----|----|----|----|----|----|
| val mAP | 50.0 | 54.5 | 52.2 | 55.1 | 43.4 | 53.7 | 55.3 | **55.6** |
| val mAR | 66.0 | 67.3 | 66.8 | 67.1 | 59.0 | 65.6 | 68.8 | **66.8** |

Best val epoch 17: **55.6 / 66.8**. Final test (stride 1, all frames): **41.6 /
48.2**.

**The pre-registered ladder check passes on its first rung.** Same split, same
evaluator, same protocol on both sides:

| test sequence | RECORD | RODNet_CDC |
|---------------|--------|------------|
| `PM1S014` (parking) | **94.2** / 96.7 | 89.2 / 93.5 |
| `PMS3000` (parking) | **81.0** / 86.1 | 64.6 / 75.6 |
| `PBMS007` (parking) | **76.9** / 85.4 | 69.0 / 75.2 |
| `ONRD002` (on-road) | 28.2 / 34.3 | 23.8 / 30.8 |
| `CM1S004` (campus) | 12.2 / 13.5 | 13.2 / 15.5 |
| **Overall** | **41.6 / 48.2** | 38.5 / 44.6 |

- **RECORD > RODNet_CDC**, the published ordering, on val (55.6 vs 53.5) and test
  (41.6 vs 38.5), winning four of five sequences.
- **`CM1S004` is confirmed intrinsically hard, not a RODNet training deficit.**
  RECORD scores 12.2 there — *below* RODNet's 13.2 — while beating it everywhere
  else. Two unrelated architectures agreeing to within a point on the one
  sequence type absent from training is the pre-registered signature of a data
  problem, and it retires the open question from the RODNet closure.
- **The val frame-filter fix is verified end-to-end**: the epoch-6 checkpoint
  scores 51.14/65.77 on the filtered per-epoch path vs 51.39/65.67 on a full
  stride-1 val eval — agreement within 0.25 AP, every sequence within ~1 point.

Still open for the full ladder: `RECORDNoLstmMulti` and `RECORDNoLstmSingle` must
land below RECORD and in the published order (65.5 > 59.5).

### What the smoke sweep is, and what its numbers are not

`scripts/smoke_sweep.py` runs every wired (model, dataset) pair end-to-end — build the
dataloaders, train, evaluate, write a verdict — on a deliberately tiny slice: `--split smoke`
takes about three files and the config's default epochs. **Its purpose is to prove the
plumbing, not to produce a score.** A pair that cannot complete one epoch cannot be compared
to anything, so this runs first and cheaply; the full measurements come later on official
splits with each paper's published schedule (`--split original --config-variant _reference`).

**A number from the smoke sweep is not comparable to anything in the results table above**,
and two entries below show why: RadarCrossAttention's 0.4367 is car-only, because a 3-file
slice of CARRADA contains no pedestrian or cyclist frames at all; RAMP_CNN's 0.9295 came from
validating on training-side data. Both look like scores and neither is one.

The same script produces the real measurements too — only the flags differ — so a report file
says which kind it is through its `splitmode` field, not its name. Results live in
`docs/smoke-sweep.json` (every pair, one sweep) and `docs/phase2-*.json` (one file per
measured run).

Historical note on the labels: this campaign ran in stages — **0** collect published
baselines before any training, **1** the smoke sweep, **2** the measured runs, **3** verdicts
— and section headings below still carry those numbers. Phases 2 and 3 were merged in
practice, which is why the results table is headed "Phase 2/3".

### Phase 1 closure

Phase 1 (plumbing, not score) ended with four runs verdicted `degenerate`. All four are
now resolved, and **only two of them were model defects** -- the other two were
*measurement* defects, where a working model was reported as producing nothing:

| pair | Phase 1 verdict | resolved | what it actually was |
|------|----------------|----------|----------------------|
| DAROD / CARRADA | degenerate | validated, 0.5642 | real: 20 porting defects |
| DAROD / RADDetDataset | degenerate | validated, 0.4463 | real: same porting defects |
| RadarCrossAttention / CARRADA | degenerate | **ok**, mAP 0.4367 | evaluator: `TP/GT` with GT 0 gave nan |
| RAMP_CNN / UWCR | degenerate | **ok**, mAP 0.9295 | 3 evaluator + 3 data/config defects |

**Treat `degenerate` as a flag for investigation, not a conclusion.** It means the
evaluator reported zeros, which conflates a broken model with a broken metric.

Neither of the two `ok` results above is comparable to its published baseline yet:

- **RadarCrossAttention 0.4367 is car-only.** The smoke sweep's 3-file sequence split of
  CARRADA contains no pedestrian or cyclist frames, so two of three classes have no
  ground truth and are excluded from the mean. The published 0.94/0.86 averages all
  three. Needs CARRADA's official split.
- **RAMP_CNN 0.9295 is 3 of 22 sequences**, with a single validation sequence (mAR hits
  1.0000 at several epochs, so it is likely an easy one). The published 81.23/84.25 is
  the full dataset. Phase 2 additionally needs `CyclicLR` support in `train()` (the
  reference runs `CyclicLR(base_lr=1e-6, max_lr=1e-5)` with `IS_Steplr=False`) and the
  paper's train/test split, which Phase 0 did not record.

**RAMP_CNN's metric is a threshold, not a ramp.** `detect_peaks` requires a strict local
maximum in a 3x5 window, so mAP stays at *exactly* 0 until the predicted confmap sharpens
past that bar, then rises steeply:

| epoch | 0-7 | 8 | 10-11 | 12 | 13 | 19 |
|-------|-----|---|-------|----|----|----|
| mAP | 0.0000 | 0.1833 | ~0.15 | 0.5761 | 0.9103 | 0.9295 |

Every earlier attempt ran 6 epochs, inside the flat region. A zero here means "not yet",
not "never" -- do not conclude a pair is broken from a short run against a peak-detection
metric.

### Phase 0 gaps found in use

- **Training schedules were not collected**, only published scores and eval protocols.
  Every schedule used so far (DAROD's 50/75 epochs, RAMP_CNN's lr) came from the
  reference implementations' configs, not the papers. Pairs without a public reference
  will need the paper text.
- **Data splits were not collected either** (RAMP_CNN's is still unknown).
- `train()` supports `step`, `exp` and `OneCycleLR` only. RAMP_CNN needs cyclic,
  RECORD/MVRECORD need early stopping, RadarCrossAttention needs plateau. These are
  trainer gaps that will block faithful Phase 2 runs.

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

**When no trained reference weights exist**, as for RAMP_CNN, the same idea still works
one level down: compare the reference's *code and data* against ours, numerically, in
this order. All six RAMP_CNN defects were found this way and none was visible from our
source alone.

1. **Config against the reference's config.** Cheapest possible check. Caught RAMP_CNN's
   lr being 10x too high (1e-4 vs 1e-5), which made the loss diverge.
2. **Ground-truth distribution.** Measure *where* the labels land, do not read the code
   that makes them. Caught degrees being fed to a radian angle grid: 622 of 640 labels
   clamped to the extreme azimuth bins, where the peak detector does not even scan.
3. **Do consecutive samples differ?** One line. Caught the radar input being indexed by
   window position instead of frame, so all 110 samples shared one input while their
   labels varied -- unlearnable by construction, and invisible to every other check.
4. **Structural diff of model and losses** against the reference. Confirmed 81 layer
   definitions plus all forward bodies identical, ruling the model out.
5. **Round-trip the evaluator on ground truth.** Feeding RAMP_CNN's detector its own
   labels yielded 80 detections from 80 gt cells, ruling the detector out.
6. **Single-batch and small-subset overfit.** Separates "cannot learn" from "has not
   learned yet": 20 samples for 400 steps reached mAP 0.6189, which proved capability
   while the full run still read 0.

Three times during this work the conclusion "the code is fine, it just needs more
training" was drawn from absence of evidence, and once it was wrong -- the frameid defect
above was found only after that claim had already been made twice. Prefer a cheap
positive test (3, 5, 6) over an argument from everything-else-checks-out.

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
  setting. Promoted to rows 15/16 on 2026-09-08 after confirming its code paths are
  unchanged since (see the Pillar closure).
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
| `MVRECORD_CARRADA_evaluation` | mAP, mAR, mIoU_rd, mIoU_ra (+ mIoU, their mean) | — |
| `RADDet_evaluation` | mAP | mAR, mIoU |
| `DAROD_evaluation` | mAP, mAR | mIoU |
| `RAMP_CNN_evaluation` | mAP, mAR | mIoU |
| `RadarCrossAttention_evaluation` | mAP | mAR, mIoU |
| `RadarPillar_evaluation` (also PointPillar) | mAP | mAR, mIoU |

So a zero in a "hardcoded" cell is meaningless, and only a zero in a "real" cell is
evidence of anything.
