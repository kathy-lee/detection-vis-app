# Phase 1 - smoke sweep

`--split original`, `--epochs 100`

Purpose is the plumbing, not the score.

**Not covered by this sweep** (23): `FFTRadNet/RADIal`, `RODNet_CDC/CRUW`, `RODNet_CDCv2/CRUW`, `RODNet_HG/CRUW`, `RODNet_HGv2/CRUW`, `RODNet_HGwI/CRUW`, `RODNet_HGwIv2/CRUW`, `RECORD/CRUW`, `RECORDNoLstm/CRUW`, `RECORDNoLstmMulti/CRUW`, `RECORD/CARRADA/RA`, `RECORD/CARRADA/RD`, `RECORDNoLstm/CARRADA/RA`, `RECORDNoLstm/CARRADA/RD`, `RECORDNoLstmMulti/CARRADA/RA`, `RECORDNoLstmMulti/CARRADA/RD`, `MVRECORD/CARRADA`, `DAROD/CARRADA`, `DAROD/RADDetDataset`, `RAMP_CNN/UWCR`, `RadarCrossAttention/CARRADA`, `PointPillar/ViewOfDelft`, `RadarPillar/ViewOfDelft`.

| Run | Split | Files | Epochs | Status | Verdict | Time (s) | Notes |
|---|---|---|---|---|---|---|---|
| `RADDet/RADDetDataset` | original | 1 | 100 | ok | suspect | 27852.0 | loss 4.927 -> 5.074 (NOT down); metrics non-zero: mAP=0.598 |

## Totals

- **suspect**: 1
