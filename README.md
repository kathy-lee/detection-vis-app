# detection-vis-app
A web application for automotive sensor data visualizaion, analysis and detection

## Running the app

Dev mode uses two long-running processes, each started by its own `run.sh`.
Run each in its own terminal:

```
./detection_vis_backend/run.sh    # brings up db + redis (docker), runs alembic,
                                   # starts the RQ worker + API on :8001
./detection_vis_ui/run.sh          # Streamlit UI on :8501
```

- Both `db` (Postgres) and `redis` run as detached docker containers with persistent
  named volumes, so they survive a `Ctrl-C` + re-run of `run.sh`. `db` (Postgres) holds dataset/file metadata and trained-model records. `redis` backs the RQ job queue and history. 
- Training and re-training run as background jobs on an **RQ worker** (backed by       redis). The RQ worker is coupled to the backend (`run.sh`), training jobs are
  monitored on the **Jobs** page.
- To abort a long training, use the **Stop** button on the Jobs page — it ends
  just that job while keeping the backend and worker up. Killing the backend
  instead takes the whole app down and leaves the job to be reconciled to
  `failed` on the next boot; either way the GPU is freed (the horse dies and the
  driver reclaims its VRAM).

## Available datasets

Datasets currently registered:

| Dataset | Native feature(s) | Link |
|---|---|---|
| RaDICaL | radar ADC, image, depth image | https://moodoki.github.io/radical_sdk/; https://databank.illinois.edu/datasets/IDB-3289560 |
| RADIal | radar ADC, lidar point cloud, image | https://github.com/valeoai/RADIal |
| CRUW (ROD2021) | radar RA, image | https://www.cruwdataset.org/rod2021 |
| CARRADA | radar RAD, image | https://github.com/valeoai/carrada_dataset |
| RADDet | radar RAD, image | https://github.com/ZhangAoCanada/RADDet |
| UWCR | radar ADC | https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection |
| Astyx | radar/lidar point cloud, image | (no valid link for now) |
| ViewOfDelft | radar/lidar point cloud, image | https://intelligent-vehicles.org/datasets/view-of-delft/ |
| NuScenes | radar/lidar point cloud, image | https://www.nuscenes.org/ |

Native feature(s) = the feature-flag columns set on the dataset's `files` rows; a model
can train on a dataset only if it consumes one of these. The dataset's code identifier
(`files.parse`, matched by the compatibility guard) is the name above, except **RADDet**
whose class is `RADDetDataset`.



## Adding a new dataset

Adding a dataset has two parts — a **code** part (a parser class) and a **data**
part (DB rows) — linked by one string: the class name stored in `files.parse`.
At runtime the backend does `DatasetFactory().get_instance(files.parse, id)`,
which instantiates the class by name via `globals()[class_name]()`, so
`files.parse` **must exactly match** a class defined in
`detection_vis_backend/datasets/dataset.py`.

**End-to-end flow (once both parts exist):**

```
add class to dataset.py  ─┐
                          ├─▶ restart backend ─▶ UI lists it via "/datasets"
insert directories+files ─┘        
                (SQL)
```

**Part 1 — Code.** Add a `Dataset` subclass in
`detection_vis_backend/datasets/dataset.py` (see `RaDICaL`, `RADIal`,
`CARRADA`, … as templates). It needs:
- `parse(self, file_id, file_path, file_name, config)` — extract per-frame
  features to disk;
- feature getters for the data types it has (`get_RD`, `get_RA`, `get_RAD`,
  `get_radarpointcloud`, `get_lidarpointcloud`, image/depth handling, …);
- `get_feature(...)`, `__getitem__`, `__len__` for the dataloader and the UI.

No explicit registration is needed — defining the class in that module is enough.

**Part 2 — Data.** There is no create API; directories/files rows are seeded
directly into Postgres with SQL (the checked-in seed is `alembic/script.sql`,
a hand-maintained data dump — not an alembic migration and not run
automatically). Insert:
- `directories` — the dataset tree (use `parent_id` for sub-datasets);
- `files` — one row per datafile, setting `parse` (your new class name),
  `path`, `name`, `size`, `config`, `directory_id`, the native-data feature
  booleans (`ADC/RAD/RA/AD/RD/spectrogram/radarPC/lidarPC/image/depth_image`),
  and `labeled`.

**Gotchas:**
- `DatasetFactory` is a cached singleton (keyed by `file_id`); it does not
  re-parse an already-loaded file, and a failed parse poisons the cached
  instance — **restart the backend** after code changes so files are re-parsed.
- `alembic/script.sql` and the live DB can drift: its `INSERT`s have no
  `ON CONFLICT`, so re-running the whole file against a populated DB duplicates
  rows. Apply only the new rows to the live DB, and keep `script.sql` in sync.
- To be trainable, a new dataset's files should expose feature flags that some
  model in `detection_vis_backend/networks/model_features.json` supports, or the
  train page's compatibility guard will block them.

## Available models

Model architectures currently registered:

| Model | Input feature(s) | Output detection format | Paper & repo |
|---|---|---|---|
| FFTRadNet | RD | RA-plane objectness-probability map + range/angle location regression (+ free-space segmentation); single class(vehicle) with fixed box: 4m×1.8m | https://github.com/valeoai/RADIal/tree/main/FFTRadNet |
| RODNet (CDC, CDCv2, HG, HGv2, HGwI, HGwIv2) | RA | RA per-class confidence map (peak-picked) | https://github.com/yizhou-wang/RODNet |
| RECORD (RECORD, RECORDNoLstm, RECORDNoLstmMulti, MV-RECORD) | RD or RA (MV-RECORD: RD+RA+AD)| per-class object-confidence map, peak-picked into detections (no regression), on the input feature plane. MV-RECORD: 2 per-class confidence/segmentation maps (RD, RA), each peak-picked into detections (no regression)| https://github.com/colindecourt/record |
| RADDet | RAD | per-class 3D boxes (center + size) in RA & RD space | https://github.com/ZhangAoCanada/RADDet |
| DAROD | RD | 2D boxes in RD plane | https://github.com/colindecourt/darod |
| RAMP_CNN | RD+RA+AD | per-class confidence map in RA plane, peak-picked into detections | https://github.com/Xiangyu-Gao/Radar-multiple-perspective-object-detection |
| RadarCrossAttention | RD+RA+AD | per-class map + center-offset + orientation in RA plane, peak-pick + offset | https://github.com/ravikothari510/crossattention_radar_detector |
| RadarPillar | radarPC (VoD 5-frame accumulation by default) | 3D boxes [x,y,z,l,w,h,yaw] in the radar frame, per-class anchors + rotated NMS; official VoD devkit mAP | https://github.com/fthbng77/RadarPillar (RadarPillars, arXiv:2408.05020) |


Three output paradigms: **fixed-size localizer** (FFTRadNet), **confidence-map peak-pickers**
(RODNet, RECORD, MVRECORD, RAMP_CNN, RadarCrossAttention — the last also adds center-offset
and heading), and **true box regressors** (RADDet 3D/YOLO, DAROD 2D/Faster-RCNN,
RadarPillar 3D/anchor-based).

## Adding a new model / network

Adding a network architecture is code plus a few config/registry entries,
linked by one string: the class name. On the train page the chosen model name
is stored as `model_config['class']`, and training does
`NetworkFactory().get_instance(model_config['class'], params)`, which
instantiates the class by name via `globals()[class_name](**params)`. So the
**same class-name string must match everywhere**: the `model_features.json` key
(= the dropdown entry) = the class defined in
`detection_vis_backend/networks/network.py` = the `<Model>_model_config.json` /
`<Model>_train_config.json` file prefixes = the `collate_func` key in
`train/train.py`.

**Where each piece lives:**

```
network.py  (class NewModel(BaseNetwork))
   +  networks/NewModel_model_config.json   (architecture params -> __init__ kwargs)
   +  train/NewModel_train_config.json       (training hyperparams)
   +  networks/model_features.json entry     (adds it to the dropdown + guard)
   +  collate_func entry in train/train.py
        └─▶ restart backend ─▶ appears in Train page ─▶ trainable
```

1. **Code.** Define `class NewModel(BaseNetwork)` in
   `detection_vis_backend/networks/network.py` (import building blocks from a
   submodule if useful, as `FFTRadNet` does). Implement `forward(...)`,
   `init_lossfunc(train_config, device)`, and `get_loss(pred, label, config)`.
   No explicit registration is needed — `NetworkFactory` finds it via
   `globals()`. The constructor receives the model config (minus the `class`
   key) as keyword args, so its `__init__` signature must match the JSON keys.

2. **Model config.** Add `networks/NewModel_model_config.json` with the
   architecture params. Plain (non-nested) configs are loaded automatically by
   `find_default_configs` in `3_Train_model.py`; only add a branch there if the
   config must be nested by dataset/feature (as `RECORD` / `RODNet` / `DAROD`
   are).

3. **Train config.** Add `train/NewModel_train_config.json` with the training
   hyperparameters (optimizer, losses, epochs, dataloader/split, …).

4. **Feature map.** Add `"NewModel": [feature codes]` to
   `networks/model_features.json`. This both lists it in the train-page dropdown
   (`model_zoo` is derived from these keys) and drives the feature-compatibility
   guard.

5. **Collate.** Add a `collate_func` entry for `NewModel` in `train/train.py`
   (`default_collate` or a custom collate). Required — training indexes
   `collate_func[model_config['class']]` and raises `KeyError` otherwise.

