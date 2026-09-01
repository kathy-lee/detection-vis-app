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
| 4a | RECORD / CARRADA / RA | **32.74** test mIoU (1 seed, official split) | 34.80 | **33.08** | **match** (−1.0% vs reference) |
| 9 | MVRECORD / CARRADA | **54.83 RD / 38.89 RA** test mIoU (1 seed, official split) | 63.2 RD / 44.5 RA | **56.35 RD / 38.09 RA** | **match** (−2.7% RD, +2.1% RA vs reference) |

Rows 11/12 are mAP@0.5; rows 4a/4b/9 are mIoU. The other runs are not yet measured
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

One artifact to ignore in `docs/phase2-mvrecord-carrada.{md,json}`: the run is labelled
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

#### An anchor for this pair is possible but expensive

Unlike every other closed row, running the reference needs a **second 320 GB copy**. Our
`RADDet_reform_dataset` has a flat `RAD/*.npy`; the reference globs `train_set_dir/RAD/*/*.npy`
(two levels, `part_1` … `part_5`) and reads train and test from separate directories. That layout
exists only as `RADDet_dataset` on the USB disk (8126 train + 2032 test). The pristine repo also
needs its environment rebuilt — the venv at `/home/kangle/projects/RADDet/.venv` has a Python
3.12 interpreter against 3.10 site-packages, so TensorFlow will not import. An anchor run would
neutralise the unknown-duration problem by running both sides for the same number of steps, which
is the only way to make row 10's verdict as sound as the others.

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
| `MVRECORD_CARRADA_evaluation` | mAP, mAR, mIoU_rd, mIoU_ra (+ mIoU, their mean) | — |
| `RADDet_evaluation` | mAP | mAR, mIoU |
| `DAROD_evaluation` | mAP, mAR | mIoU |
| `RAMP_CNN_evaluation` | mAP, mAR | mIoU |
| `RadarCrossAttention_evaluation` | mAP | mAR, mIoU |
| `RadarPillar_evaluation` (also PointPillar) | mAP | mAR, mIoU |

So a zero in a "hardcoded" cell is meaningless, and only a zero in a "real" cell is
evidence of anything.
