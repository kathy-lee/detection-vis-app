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

| # | pair | ours (mean ± sd) | published | reference on OUR pipeline | verdict |
|---|------|------------------|-----------|---------------------------|---------|
| 11 | DAROD / CARRADA | **0.5642 ± 0.0140** | 0.5583 | 0.5581 | **match** (+1.1%, 0.4 sd) |
| 12 | DAROD / RADDetDataset | **0.4463 ± 0.0186** | 0.4657 | 0.4478 | **match** (−0.3% vs reference, 0.1 sd) |
| 2a | RODNet_CDC / CRUW | **38.5 / 44.6** test AP/AR (1 seed, our split; best-val epoch 7) | 74.29 / 77.85 | — | **inconclusive-by-protocol** (see closure section) |
| 4b | RECORD / CARRADA / RD | **53.15** test mIoU (1 seed, official split) | 58.10 | **53.19** | **match** (−0.08% vs reference) |
| 4a | RECORD / CARRADA / RA | not yet run | 34.80 | **33.08** | reference anchor established |

Rows 11/12 are mAP@0.5; rows 4a/4b are mIoU. The other runs are not yet measured
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

| class | reference on our data | published |
|-------|----------------------|-----------|
| Bg | 99.23 | 99.70 |
| Ped | 8.07 | 10.30 |
| Cycl | **4.71** | 1.40 |
| Car | 20.32 | 27.70 |
| **mIoU** | **33.08** | **34.80** |

A 1.7-point shortfall, ~5% relative -- the same signature as the RD view, on a second
independent view. Two details worth keeping:

- **The val-to-test drop is small here** (34.59 -> 33.08, 1.5 points) against RD's ~10.
  Background contributes ~25 of the 34.8 total and transfers between splits essentially
  unchanged, so almost the whole drop lands on the three object classes.
- **The shortfall is not uniform**: car is 7.4 low and pedestrian 2.2 low, but cyclist is
  3.4 points *higher* than published (4.71 vs 1.40). A published cyclist IoU of 1.4 is
  close enough to zero that it is plausibly noise in the authors' own run, which is a
  reason to weight the RA aggregate lightly in either direction.

**Running the RA view also required a third patch to the reference**, and this one is
substantive rather than cosmetic: `CarradaDatasetRangeAngle.__getitem__`
(`datasets/carrada/dataloaders.py:330`) files the range-ANGLE mask under the key
`'rd_mask'` -- copy-pasted from the range-Doppler class -- while
`carrada_trainer_sv.py:174` reads `batch['ra_mask']` for that view. The RA path therefore
raises `KeyError` on its first validation batch and **cannot train for a single epoch as
published**. Since Table V reports an RA number, that path certainly ran for the authors,
which independently confirms the conclusion above: the released artifact is not the code
that produced the published table.

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
