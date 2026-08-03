# Phase 1 - smoke sweep

`--split smoke`, config-default epochs

Purpose is the plumbing, not the score.

**Not covered by this sweep** (1): `FFTRadNet/RADIal`.

| Run | Split | Files | Epochs | Status | Verdict | Time (s) | Notes |
|---|---|---|---|---|---|---|---|
| `RODNet_CDC/CRUW` | sequence | 3 | 1 | ok | ok | 196.9 | loss 0.03746 (single epoch, no trend); metrics non-zero: mAP=0.03419, mAR=0.03431 |
| `RODNet_CDCv2/CRUW` | sequence | 3 | 1 | ok | ok | 213.7 | loss 0.02546 (single epoch, no trend); metrics non-zero: mAP=0.1463, mAR=0.8045 |
| `RODNet_HG/CRUW` | sequence | 3 | 1 | ok | ok | 295.6 | loss 0.276 (single epoch, no trend); metrics non-zero: mAP=0.0679, mAR=0.08299 |
| `RODNet_HGv2/CRUW` | sequence | 3 | 1 | ok | ok | 311.6 | loss 0.03012 (single epoch, no trend); metrics non-zero: mAP=0.3151, mAR=0.331 |
| `RODNet_HGwI/CRUW` | sequence | 3 | 1 | ok | ok | 974.8 | loss 0.1058 (single epoch, no trend); metrics non-zero: mAP=0.2974, mAR=0.3093 |
| `RODNet_HGwIv2/CRUW` | sequence | 3 | 1 | ok | ok | 1015.3 | loss 0.01142 (single epoch, no trend); metrics non-zero: mAP=0.5598, mAR=0.6024 |
| `RECORD/CRUW` | sequence | 3 | 1 | ok | ok | 212.4 | loss 0.06235 (single epoch, no trend); metrics non-zero: mAP=0.1823, mAR=0.2449 |
| `RECORDNoLstm/CRUW` | sequence | 3 | 1 | ok | ok | 178.1 | loss 0.09515 (single epoch, no trend); metrics non-zero: mAP=0.111, mAR=0.1557 |
| `RECORDNoLstmMulti/CRUW` | sequence | 3 | 1 | ok | ok | 174.5 | loss 0.06854 (single epoch, no trend); metrics non-zero: mAP=0.05491, mAR=0.08064 |
| `RECORD/CARRADA/RA` | sequence | 3 | 1 | ok | ok | 177.9 | loss 1.82 (single epoch, no trend); metrics non-zero: mAP=0.2793, mAR=0.4456, mIoU=0.2767 |
| `RECORD/CARRADA/RD` | sequence | 3 | 1 | ok | ok | 90.5 | loss 1.803 (single epoch, no trend); metrics non-zero: mAP=0.3534, mAR=0.4908, mIoU=0.3507 |
| `RECORDNoLstm/CARRADA/RA` | sequence | 3 | 1 | ok | ok | 40.4 | loss 1.835 (single epoch, no trend); metrics non-zero: mAP=0.2739, mAR=0.3886, mIoU=0.2707 |
| `RECORDNoLstm/CARRADA/RD` | sequence | 3 | 1 | ok | ok | 20.6 | loss 1.815 (single epoch, no trend); metrics non-zero: mAP=0.3096, mAR=0.4926, mIoU=0.3065 |
| `RECORDNoLstmMulti/CARRADA/RA` | sequence | 3 | 1 | ok | ok | 40.5 | loss 1.84 (single epoch, no trend); metrics non-zero: mAP=0.2961, mAR=0.3691, mIoU=0.2877 |
| `RECORDNoLstmMulti/CARRADA/RD` | sequence | 3 | 1 | ok | ok | 21.0 | loss 1.814 (single epoch, no trend); metrics non-zero: mAP=0.314, mAR=0.4898, mIoU=0.311 |
| `MVRECORD/CARRADA` | sequence | 3 | 1 | ok | ok | 339.5 | loss 0.143 (single epoch, no trend); metrics non-zero: mAP=0.3088, mAR=0.4769, mIoU=0.3415 |
| `RADDet/RADDetDataset` | original | 1 | 1 | ok | ok | 2678.7 | loss 1355 (single epoch, no trend); metrics non-zero: mAP=0.01458 |
| `DAROD/CARRADA` | sequence | 3 | 1 | ok | degenerate | 28.9 | loss 2.079 (single epoch, no trend); pinned at exactly 0 across all epochs: mAP, mAR |
| `DAROD/RADDetDataset` | original | 1 | 1 | ok | degenerate | 2709.2 | loss 1.767 (single epoch, no trend); pinned at exactly 0 across all epochs: mAP, mAR |
| `RAMP_CNN/UWCR` | sequence | 3 | 1 | ok | degenerate | 688.7 | loss 66.86 (single epoch, no trend); pinned at exactly 0 across all epochs: mAP, mAR |
| `RadarCrossAttention/CARRADA` | sequence | 3 | 1 | ok | degenerate | 334.9 | loss 880 (single epoch, no trend); evaluator produced none of its metrics (mAP) |
| `PointPillar/ViewOfDelft` | original | 1 | 1 | ok | ok | 343.7 | loss 1.284 (single epoch, no trend); metrics non-zero: mAP=21.66 |
| `RadarPillar/ViewOfDelft` | original | 1 | 1 | ok | ok | 213.4 | loss 1.373 (single epoch, no trend); metrics non-zero: mAP=6.19 |

## Totals

- **degenerate**: 4
- **ok**: 19
