# Dataset–Network Compatibility

This document records what it takes for a dataset to train a given network in this
app, then lists the current datasets and networks against those criteria and
summarises which pairs are compatible.

- **Section 1 — Compatibility checkpoints:** the criteria a `(dataset, network)`
  pair must satisfy.
- **Section 2 — Datasets and their data features.**
- **Section 3 — Networks and their training-data requirements / preprocessing.**
- **Section 4 — Compatibility table** (every dataset × every network).

---

## 1. Compatibility checkpoints

A dataset can train a network only when **all** of the following hold. The first
three are the obvious ones; the rest are the requirements that a naive "it has the
right input and some labels" reading silently skips. Meeting 1–3 is **necessary but
not sufficient** — a pair must clear every checkpoint below.

### C1. Input feature: modality, axis meaning, and value semantics
Not just tensor shape — three things must line up:
- **Modality match.** The dataset must provide the feature the network consumes
  (RD, RA, AD, RAD cube, raw ADC, radar point cloud, lidar point cloud, …).
- **Axis semantics.** Two `256×256` tensors are not interchangeable: the network
  expects specific physical axes (e.g. range × azimuth), and a same-shaped map that
  is range × Doppler, or transposed, will not train correctly.
- **Value semantics.** Complex ADC vs. magnitude vs. dB-scaled vs. real/imag
  channels must match what the network assumes.

### C2. Annotation(Label or Ground-truth) representation
The annotation must be the **kind** of label the network's loss and the evaluator
operate on, in the right space:
- **Label format.** What kind/shape the label is: Confidence maps / keypoint object locations (RODNet,
  RAMP-CNN) vs. 3D boxes `[x,y,z,l,w,h,yaw]` (pillar models) vs. 2D boxes on an
  RD/RAD map vs. per-pixel segmentation masks. A dataset full of 3D boxes does not
  directly train a confidence-map model, and vice versa.
- **Coordinate system.** The GT's coordinates must be expressed in — or convertible
  to — the coordinate system the network's loss operates in. Example: the VoD pillar
  models (PointPillar / RadarPillar) train in the **radar** coordinate system, but
  VoD's KITTI labels are authored in the **camera** coordinate system, so they are
  converted to radar coordinates before target assignment.
- **Spatial fit to the model's grid.** Once in the right coordinate system, the GT
  must land inside the finite, discretized region the model actually predicts over,
  with a known calibration:
  - For pillar models: objects must fall inside the configured
    `point_cloud_range` / `voxel_size` (a box beyond the range yields no target
    cell and is silently dropped).
  - For map-grid models: the annotation's real (range, angle, Doppler) must map onto
    the feature bins (known bin→physical calibration).
  Unlike coordinate-system/representation/taxonomy — which are intrinsic to the
  label — this one is a *relation between the label and the model's config*: widening
  `point_cloud_range` can flip it without touching the annotation.
- **Class taxonomy.** The detection head and anchors are fixed to a class set(VoD → Car/Ped/Cyclist with those specific anchor sizes);
  a dataset with a different class set needs the head/anchors reconfigured.

### C3. Preprocessing to input + target is doable on this dataset
Every step the network needs to turn raw dataset files into its input tensor **and**
its training target must be feasible from what the dataset actually stores:
normalization, complex→real/imag channel split, RAD→channel reshape, pillarization,
confidence-map (ConfMap) rendering, windowing into `T` frames, anchor/box target
assignment, etc. If a step needs information the dataset does not carry, the pair
fails here.

### C4. Temporal structure (for sequence models)
Sequence networks (`win_size > 1`: RODNet family, RECORD/RECORDNoLstmMulti,
MVRECORD, RAMP-CNN, RadarCrossAttention) need **ordered consecutive frames** with
per-frame annotation. A dataset of independent, unordered single frames can satisfy
C1–C3 for such a network and still be untrainable. Single-frame networks
(`win_size = 1`) do not impose this.

### C5. Sufficient labeled samples and splits
Enough annotated samples, plus train/val/test splits, in the format the training
loop expects.

---

## 2. Datasets and their data features

Legend: **GT** = ground-truth representation. **Seq** = temporally ordered /
sliding-window capable.

| Dataset | Modalities (shape) | Annotation | Classes | Seq |
|---|---|---|---|---|
| **RADIal(labeled)** | RD (range:512, Doppler:256, Rx:16, complex(real+imag)); radarPC `Nx9[range, azimuth, elevation, power, doppler, x, y, z, v]`; lidarPC `Nx4[x, y, z, reflectance]`; image | 2D box in image, single center point in RAD-plane/lidarPC/radarPC:[laser_X_m,laser_Y_m,laser_Z_m,radar_X_m,radar_Y_m,radar_R_m,radar_A_deg,radar_D_mps,radar_P_db]; freespace `*.png`. | 1 (vehicle) | ✓ |
| **RADIal(Raw)** | ADC (Tx:12, Rx:16, Calibrated Azi bins: 751); | — | — | ✓ |
| **CRUW** (ROD2021) | RA(range:128,azimuth:128, complex, there are 255 raw chirps per frame, only 4 are kept: chirp_ids at [0, 64, 128, 192]); image. | object centers  in RA plane:`[range_idx, angle_idx, category]`. | 3 (ped, cyclist, car) | ✓ (win_size=16) |
| **CARRADA** | RA (256×256,real); RD (256×64,real); AD (256×64,real); RAD (256×256×64,real); image. | 3 types of annotations in both RA and RD plane: sparse points; 2D boxes; dense mask. | 4 (bg, ped, cyclist, car) | ✓ (win_size) |
| **RADDetDataset** | RAD (range:256, azimuth:256, Doppler:64, complex); image. | **YOLO 3D anchor labels**: 3D box in RAD-space rasterized onto a 16×16×4 anchor grid over (range,angle,Doppler), 6 anchors/cell | 6 (person, bicycle, car, motorcycle, bus, truck) | per-frame |
| **UWCR** | ADC (range: 128, Doppler:255, Tx:2, Rx:4, complex); image. | 2D box in RA plane. | 6 (ped, car, motorbike, bus, truck, cyclist) | ✓ (win_size) |
| **Astyx** (2019) | radarPC; lidarPC; image. | **3D boxes** (center3d, dim w/l/h, quaternion) → 8-corner in radar/lidar frame or 2D image projection. | JSON-defined (not enumerated) | per-frame |
| **ViewOfDelft** | radarPC **`Nx7 [x,y,z,RCS,v_r,v_r_comp,time]`** (+ `radar_3frames`/`radar_5frames` accumulation); lidarPC `Nx4 [x,y,z,reflectance]`; image. RAD/RD/RA = none. | **3D boxes**, KITTI label → radar-frame **`[x,y,z_center,l,w,h,yaw]`** (pillarized per frame). | 3 (Car, Pedestrian, Cyclist) | accumulation, not windowed |
| **NuScenes** | radarPC **18-dim** (multi-sweep, 5 radars); lidarPC `Nx4 [x,y,z,intensity]`; image `1600×900`. RAD/RD/RA = none. | **3D boxes** (translation, size, quaternion) → 8-corner in sensor frame or 2D image projection. | nuScenes taxonomy (open list) | ✓ (scene-based) |
| **K-Radar** | 4D RADE tensor (Doppler:64, Range:256, Azimuth:107, Elevation:37, **real power**); 3D tensor on a Cartesian grid(z:150, y:400, x:250, **real** Cartesian power); lidarPC: N×9 `[x,y,z,intensity,t,reflectivity,ring,ambient,range]`; image (4 cams). | **3D boxes** in **Cartesian** `[x,y,z,l,w,h,yaw]` (+track IDs) | 5 (Car, Bus/Truck, Motorcycle, Bicycle, Pedestrian) | ✓ (ordered frames) |


## 3. Networks and their training-data requirements

Legend: **Seq(n)** = sequence model needing an ordered
`n`-frame window. GT representation is what the network's `get_loss` consumes.
"Wired" = the `(model, dataset)` pairs currently registered in implementation.

 
| **FFTRadNet** | Requirements |
|---|---|
|**Input** | RD in `32×512×256` (32 = 2·Rx, Rx=16, 2 = real+imag concatenated `[real×16 ‖ imag×16]`; 512 = range, 256 = Doppler) |
| **Frames** | Single |
| **Label** | Objectness + range/angle **regression grid** `(3,128,224)` (ch0:objectness, ch1–2: range/angle sub-cell offset) + freespace **seg mask** `(256×224)` |
| **Preprocessing** | RD Normalization; Encode label grid; freespace PNG crop/resize | 
| **Wired dataset(s)** | RADIal |
| **Adjustable configs** | `mimo_layer`: default 192, can vary from 24 to 384; MIMO geometry 12Tx+16Rx is hardcoded |


| **RODNet** | Requirements |
|---|---|
|**Input** | RA sequence in `(C, win, H, W)`. Two forms by variant:<br>• **CDC / HG / HGwI** (single chirp): `2×win×H×W` — **2 = real+imag** of one RA map.<br>• **CDCv2 / HGv2 / HGwIv2** (MNet): `2×win×4×H×W` — **2 = real+imag**, **4 = chirps** , real+imag and chirps are **separate axes** (not `4×2=8`). |
| **Frames** | Seq(win) |
| **Label** | Gaussian confidence maps in `(3,win,H,W)` (3 is class-count: ped/cyclist/car) |
| **Preprocessing** | `generate_confmap` render + normalize | 
| **Wired dataset(s)** | CRUW |
| **Adjustable configs** | H, W: divisible by 8, win: divisible by 4, class-count: configurable. v2 variants only: `C_m` in  `mnet_cfg = [in_chirps, C_m]`, but `in_chirps` must equal `in_channels`, so the two move together. Note `in_channels` means real+imag (2) for CDC/HG/HGwI but chirp count (4) for the v2 variants. |

RECORD is a task-agnostic encoder–decoder, the dataset's annotation decides whether it's detection or segmentation. 
| **RECORD** | Requirements |
|---|---|
|**Input** | RD or RA map sequence in `(C, win, H, W)`, C is set by the data's value semantics: CARRADA (real magnitude) → `1×win×H×W` (single channel); CRUW (complex) → `2×win×H×W` (2 = real+imag). |
| **Frames** | Seq(win)|
| **Label** | Detection task: Gaussian ConfMap rendered from object center points `(3,H,W)`; Segmentation task: Dense one-hot mask `(4,H,W)`, per-pixel class |
| **Preprocessing** | CARRADA: per-frame `.npy` + **normalize** (`norm_type`); masks pre-baked on disk. CRUW: window → ConfMap (from object centers) → keep **last frame**; input used **raw (no normalize)** | 
| **Wired dataset(s)** | CRUW, CARRADA|
| **Adjustable configs** | `win` any length (LSTM, no divisibility); H and W flexible **but each must divide the encoder downsample** (3 stride-2 stages) — CRUW: H ÷8 and W ÷8; CARRADA: H (range) ÷8 and W (Doppler) ÷2; class-count configurable|


| **RECORDNoLstm** | Requirements |
|---|---|
|**Input** | RD or RA `C×H×W`, single frame (C = 1 CARRADA real / 2 CRUW complex)|
| **Frames** | single |
| **Label** | same as RECORD|
| **Preprocessing** | same as RECORD| 
| **Wired dataset(s)** | CRUW, CARRADA|
| **Adjustable configs** | same as RECORD (no `win`)|


| **RECORDNoLstmMulti** | Requirements |
|---|---|
|**Input** | RD or RA `(C, H, W)`(the `win` frames are folded into channels: C=C_frame·win) |
| **Frames** | Seq(win) |
| **Label** | same as RECORD|
| **Preprocessing** | same as RECORD| 
| **Wired dataset(s)** | CRUW, CARRADA|
| **Adjustable configs** | `win` ; class-count; `C` is **derived** as C_frame·win; H and W flexible **but each must divide the encoder downsample** (3 stride-2 stages) — CRUW: H ÷8 and W ÷8, CARRADA: H (range) ÷8 and W (Doppler) ÷2. Default: CRUW 8·win(12)= 96, CARRADA 1·win(5)=5.|


| **MVRECORD** | Requirements |
|---|---|
|**Input** | RD+RA+AD `1×win×H×W` each (1 = C real per view; win = temporal T)|
| **Frames** | Seq(win) |
| **Label** | **Two one-hot seg masks** (RD + RA), `(4,H,W)` (4 incl. bg) |
| **Preprocessing** | 3-view load + per-view normalize over win-frame window| 
| **Wired dataset(s)** | CARRADA|
| **Adjustable configs** | `win` (default 5);class-count; H, W each must divide the encoder downsample (3 stride-2 stages), **per view**: RA H÷8 & W÷8; RD range÷8 & Doppler÷2; AD angle÷8 & Doppler÷2. |


| **RADDet** | Requirements |
|---|---|
|**Input** | RAD cube `64×256×256`(Doppler×Range×Azimuth) |
| **Frames** | single |
| **Label** | **Dense YOLO 3D anchor labels** `(16,16,4,n_anchors,n_class+7)` +  actual GT boxes (30,7) (7:`[x,y,z,w,h,d,cls]`) |
| **Preprocessing** | complex→log-power `log10(\|z\|²+1)`; global affine norm (÷ log-variance); channels = 64 Doppler bins; anchor assignment by 3D IoU| 
| **Wired dataset(s)** | RADDetDataset|
| **Adjustable configs** | n_anchors(default 6, corresponds to "anchor_boxes" in model config file); n_class; the **range and azimuth** axes of `input_size` (each ÷16). Pinned despite also being entries in the model config json: `input_channels`, and the **Doppler** axis of `input_size` — both fixed by the literal arguments passed to `YoloHead`. |

| **TransRAD** | Requirements |
|---|---|
|**Input** | RAD cube with Doppler as channels, `256×256×256` (Doppler×Range×Azimuth) — the on-disk `256×256×64` cube resized along Doppler by nearest neighbour |
| **Frames** | single |
| **Label** | Per-frame box list `(N, 7)` `[cls, x, y, z, w, h, d]` normalised to the cube (x,w ÷256; y,h ÷256; z,d ÷64); anchor-free, assigned at train time (task-aligned assignment) |
| **Preprocessing** | complex→log-power `log10(\|z\|²+1)`; global affine norm (same constants as RADDet); edge-crossing boxes clamped to their larger part (not split); Doppler resize 64→256 (`scipy.ndimage.zoom`, order 0) |
| **Wired dataset(s)** | RADDetDataset |
| **Adjustable configs** | `n_class`; `embed_dims`/`depths`/`num_heads` (RMT); `phi` (neck width); `reg_max` (DFL bins). The range and azimuth axes of `input_size` must be ÷32; the Doppler axis is pinned at 256 by the adapter's resize, and `eval_input_size` in the train config names the on-disk 64. |


| **T_RODNet** | Requirements |
|---|---|
|**Input** | RA sequence `2×16×128×128` — **2 = real+imag** of one chirp per window (random chirp in training, chirp 0 at test, as the reference's loader and test script), **16 frames** fixed by the 3D patch embedding |
| **Frames** | Seq(win=16), train stride 4, test stride 8 |
| **Label** | Gaussian confidence maps `(3,16,128,128)` — the same RODNet maps the dataset pre-renders; MSE on the raw (un-sigmoided) output, as the reference trains |
| **Preprocessing** | none beyond windowing (no normalisation, as RODNet); built by `TRODNetOnCRUW` in `datasets/adapters/cruw.py` |
| **Wired dataset(s)** | CRUW |
| **Adjustable configs** | `n_class`; `embed_dim` is pinned at 64 by hard-coded channel widths in the reference's patch embedding and merging blocks; `win_size` (attention window, 4) must divide the 8×64×64 token grid; the 16-frame window and 128×128 maps are pinned by the patch embedding. |


| **DAROD** | Requirements |
|---|---|
|**Input** | RD `1×256×64`|
| **Frames** | single|
| **Label** | **2D boxes** `[y1,x1,y2,x2]` + class ID; ROI/anchor assignment inside loss|
| **Preprocessing** | RD normalize; box padding; empty-label samples dropped| 
| **Wired dataset(s)** | CARRADA, RADDetDataset|
| **Adjustable configs** | class-count (6 RADDetDataset / 3 CARRADA, +bg); RPN/FastRCNN hyperparameters. `input_size` is adjustable only in lockstep with `feature_map_shape` (no derivation between them). Channel count fixed at 1.|


| **RAMP_CNN** | Requirements |
|---|---|
|**Input** | RA `2×D×R×A` (**2 = real+imag**) + RD `1×D×R×V` + AD `1×D×A×V` (**1 = magnitude**). `D = 2·win` is a network constraint: the encoder strides depth ÷8 and the decoder upsamples ×4 (net ÷2), so the input must be twice as deep as the `win`-deep label. |
| **Frames** | Seq(win), default 16|
| **Label** | **Gaussian confidence maps** `(n_class, win, R, A)`|
| **Preprocessing** | *(UWCR dataset specific)* ADC `(128 samp,255 chirp,4 Rx,2 Tx)` → 8 virtual antennas → range FFT → **chirp split**: a 128-point Doppler FFT cannot consume 255 chirps at once, so it runs over two overlapping windows (`[0:128]`, `[127:255]`) and RA takes chirps 0 and 128 — this is what supplies the 2 samples/frame that fill `D=2·win`. Then RA (angle FFT, complex kept) / RD (Doppler FFT, \|·\| meaned over antennas) / AD (Doppler+angle FFT, \|·\| meaned over range); stack over win frames × 2 samples. **Inputs are not normalized**; `generate_confmap` → `normalize_confmap` applies to labels only. | 
| **Wired dataset(s)** | UWCR|
| **Adjustable configs** | `win` (÷8:3 stride-2 depth stages), R and A each ÷8. `n_class` is a config value the network honours, but **effectively pinned to 3 by the data pipeline** (see note below). |


| **RadarCrossAttention** | Requirements |
|---|---|
|**Input** | RA `win×256×256` + RD `win×256×64` + AD `win×64×256`. |
| **Frames** | Seq(win), default 5 |
| **Label** | **Gaussian center heatmap** `(3,256,256)` + optional center-offset `(2,256,256)` + orientation `(2,64,64)`|
| **Preprocessing** | RA used as-is; RD flipped on both axes with DC bins `31:34` zeroed; AD derived from RAD (ifftshift → ifft → \|·\|² → sum → `10log10`) then transposed, DC bins zeroed. **Inputs are not normalized** — a literal `# TODO: normalize` sits at `dataset.py:2122`, unlike the RECORD/CARRADA path which does. On-the-fly heatmap/center/orient render.| 
| **Wired dataset(s)** | CARRADA|
| **Adjustable configs** | `win_size`; the `center_offset` / `orientation` head toggles. Pinned: all three spatial sizes — hardcoded in the loader (`dataset.py:2098-2100`) and enforced by `nn.LayerNorm([256,32,32])` (`radarcrossattention.py:57`) plus the RD×AD matmul, which needs RD's and AD's encoded axes to agree. `n_class` is a config entry the head honours, but the loss weight is `torch.zeros([3,256,256])` (`network.py:1006`), so it is effectively 3.|


| **PointPillar** | Requirements |
|---|---|
|**Input** | radarPC `N×7`|
| **Frames** | single|
| **Label** | **3D boxes** `[x,y,z,l,w,h,yaw]` + per-class anchors (3: Car/Ped/Cyclist)|
| **Preprocessing** | FOV filter; **pillarize** (P,K,D); anchor target assignment in loss; sensor-agnostic (lidarPC valid once VoD-lidar path lands)| 
| **Wired dataset(s)** | ViewOfDelft|
| **Adjustable configs** | point-cloud range, voxel size, max pillars/points, backbone channels, anchors & classes **all in model config**; sensor-agnostic|


| **RadarPillar** | Requirements |
|---|---|
|**Input** | radarPC `N×7` |
| **Frames** | single|
| **Label** | same 3D boxes as PointPillar (3 classes)|
| **Preprocessing** | as PointPillar **+ requires radial-velocity features** (`v_r`,`v_r_comp`) for velocity decomposition; uniform C=32 + PillarAttention| 
| **Wired dataset(s)** | ViewOfDelft|
| **Adjustable configs** | grid/anchors/classes in config; extras (`C`, PillarAttention, velocity decomp.) are config switches; **needs per-point Doppler**|


**RAMP_CNN `n_class` is configurable but pinned to 3 in practice.** The network is fully
parameterized (`RAMP_CNN` → `Fuse_fea_new_rep` → `convt4 out_channels=n_class`; no literal
class count in `ramp_cnn.py`), but the data pipeline hardcodes 3 in three places:
`self.n_class = 3` (`dataset.py:2826`), `classes = ["pedestrian","cyclist","car"]`
(`utils.py:500`) and `n_class = 3` (`utils.py:549`) — the latter two still carry
`# dataset.object_cfg.*` comments from upstream. So UWCR's `label_map` lists **6**
categories but `generate_confmap` drops motorbike/bus/truck with a log warning
(`utils.py:518`), and the `truck→car` remap in `class_ids` (`dataset.py:2827`) is dead code
(its consumer is commented out at `:2879`). Raising `n_class` in the model config alone
yields output channels nothing supervises; a real multi-class run needs those three
literals changed too.

**RAMP_CNN `n_vel` is hardcoded twice over.** The Doppler FFT length `n_vel=128`
(`dataset.py:2648`, alongside `n_angle`/`n_range`/`n_chirp`/`n_rx`/`n_sample`) is set in the
loader and not read from the radar JSON loaded two lines earlier, while the shape assert
checks `ramap_vsize` from that JSON — the two must agree or `__getitem__` trips. `n_chirp=255`
is similarly load-bearing: the chirp split index `n_vel - 1 = 127` tiles a frame exactly only
because 255 = 2×128 − 1.

## 4. Compatibility table

Every network (rows) × every **labelled** dataset (columns), scored against C1–C5.
**RaDICaL** and **RADIalRaw** are omitted: they carry no annotations, so they cannot
train any network (C2/C5).

**Legend:**
- ✅ **Supported now** — wired in code; trains and evaluates today.
- ⚠️ **Possible with work** — **not wired in code**; modality **and** label *type* are
  compatible in principle, but the pair needs preprocessing / GT conversion /
  calibration / a training path. The reason codes say which. For ⚠️`gt` the GT
  conversion was judged feasible, not verified end-to-end.
- ❌ **Blocked** — a checkpoint cannot be met.

**Reason codes** (on ⚠️/❌):
- `mod` missing modality (C1)
- `gt` GT-representation mismatch (C2)
- `np` no training path for this pair — the dataset's `__getitem__` has no branch for
  this model, or returns `None` (C3)
- `seq` sequence model but dataset frames are independent (C4)
- `geo` model geometry hardcoded to one dataset (C1)
- `feat` per-point attributes in the point cloud differ (C1)

| Network ↓ / Dataset → | RADIal | CRUW | CARRADA | RADDetDataset | UWCR | ViewOfDelft | Astyx | NuScenes |
|---|---|---|---|---|---|---|---|---|
| **FFTRadNet** | ✅ | ❌mod | ❌geo,gt | ❌geo,gt | ❌geo | ❌mod | ❌mod | ❌mod |
| **RODNet** (×6 variants) | ❌mod | ✅ | ⚠️gt | ❌seq | ⚠️np | ❌mod | ❌mod | ❌mod |
| **RECORD** | ⚠️gt | ✅ | ✅ | ❌seq | ⚠️np | ❌mod | ❌mod | ❌mod |
| **RECORDNoLstm** | ⚠️gt | ✅ | ✅ | ⚠️gt | ⚠️np | ❌mod | ❌mod | ❌mod |
| **RECORDNoLstmMulti** | ⚠️gt | ✅ | ✅ | ❌seq | ⚠️np | ❌mod | ❌mod | ❌mod |
| **MVRECORD** | ❌mod | ❌mod | ✅ | ❌mod,gt | ❌gt | ❌mod | ❌mod | ❌mod |
| **RADDet** | ❌mod | ❌mod | ⚠️gt | ✅ | ❌mod | ❌mod | ❌mod | ❌mod |
| **TransRAD** | ❌mod | ❌mod | ⚠️gt | ✅ | ❌mod | ❌mod | ❌mod | ❌mod |
| **T_RODNet** | ❌mod | ✅ | ⚠️gt | ❌seq | ⚠️np | ❌mod | ❌mod | ❌mod |
| **DAROD** | ❌gt | ❌mod | ✅ | ✅ | ❌gt | ❌mod | ❌mod | ❌mod |
| **RAMP_CNN** | ❌mod | ❌mod | ⚠️gt | ❌mod,seq | ✅ | ❌mod | ❌mod | ❌mod |
| **RadarCrossAttention** | ❌mod | ❌mod | ✅ | ❌mod,seq | ❌geo,gt | ❌mod | ❌mod | ❌mod |
| **PointPillar** | ❌gt | ❌mod | ❌mod | ❌mod | ❌mod | ✅ | ⚠️np,feat | ⚠️np,feat |
| **RadarPillar** | ❌gt | ❌mod | ❌mod | ❌mod | ❌mod | ✅ | ⚠️np,feat | ⚠️np,feat |

### Notes on selected ⚠️/❌ cells (what the work actually is)

- **RODNet / RAMP_CNN on CARRADA:** these share the confidence-map / heatmap task
  family, and CARRADA provides the RA/RD/AD modalities and object centers. CARRADA's
  loader already dispatches on `model_type` (`dataset.py:1890, 1985, 1988, 2095`), so
  the gap is rendering ConfMaps from its box centers (`gt`) plus registration. The rest
  of the RECORD family and RadarCrossAttention are already wired on CARRADA (✅).
  Class-count differences (3 vs 4) are a head/config change, not a blocker.
- **RODNet/RECORD family and DAROD on UWCR (`np`):** modality and labels are fine — UWCR
  derives RA/RD/AD from ADC and already renders ConfMaps via `generate_confmap`. But the
  UWCR class (`dataset.py:2590`–EOF) contains **no `model_type` branching at all**: its
  `__getitem__` unconditionally emits the RAMP_CNN sample, whose RA is
  `(2, 2·win, 128, 128)` — twice the temporal depth RODNet/RECORD expect — and
  `prepare_for_train` never reads `model_cfg`. So these pairs need a **new loader
  branch**, not just a registration line; wiring them up alone fails at the first batch.
- **RadarCrossAttention on UWCR (`geo`):** blocked more deeply than the loader. The model
  is pinned to CARRADA's geometry — `nn.LayerNorm([256,32,32])`
  (`radarcrossattention.py:57`) requires RA to be exactly `256×256`, and UWCR's RA is
  `128×128`, which encodes to `16×16`. The RD×AD matmul imposes the same rigidity. Fixing
  it means editing the model, not the wiring — hence ❌ rather than ⚠️, as with FFTRadNet.
- **DAROD on RADIal / UWCR (`gt`):** RD is present, but DAROD's target is a box in the
  **RD plane** — it needs a **range extent and a Doppler extent**, and neither dataset
  carries a Doppler extent for an object. RADIal has a camera-image box
  (`x1,y1,x2,y2`), a lidar point and a radar centre (`radar_R_m/A_deg/D_mps`): back-
  projecting the image box yields azimuth and elevation extent, not range or Doppler.
  UWCR's label is `[uid, class, px, py, wid, len]` — extents, but in the Cartesian/RA
  plane (and the loader reads only `px,py`). Clustering RADIal's `laser_PCL` could
  recover a range extent, yet lidar carries no velocity, so the Doppler axis has no
  annotation source at all. By C3 ("if a step needs information the dataset does not
  carry, the pair fails here") this is ❌, not ⚠️.
- **RADDet on CARRADA:** CARRADA ships the RAD cube, but RADDet needs a **3D RAD box**
  `[x,y,z,w,h,d]`; CARRADA annotates 2D boxes per view (RA and RD separately). A 3D
  box can in principle be reconstructed by combining the two views — feasible but
  non-trivial, hence ⚠️ not ✅.
- **TransRAD on CARRADA:** the same situation as RADDet — the cube exists, the 3D RAD
  box does not.
- **Pillar models on Astyx / NuScenes:** correct modality (radar point cloud) and GT
  (3D boxes), but `__getitem__` returns `None` (`np`) — no training path — and the
  per-point attributes differ (Astyx has no per-point Doppler for RadarPillar's
  velocity decomposition; NuScenes points are 18-dim, VoD config expects `N×7`), so
  a per-point attribute mapping (`feat`) is also needed.
- **Pillar models on RADIal (`gt`):** the pillar models need 3D boxes
  `[x,y,z,l,w,h,yaw]`, and RADIal annotates only a **centre** — `laser_X/Y/Z_m` plus
  `radar_X_m/Y_m`, with no `l,w,h` and no `yaw` anywhere in `labels.csv`. The extents
  are physically present in `laser_PCL` and could be clustered out around each centre,
  but that is generating new annotations, not converting existing ones; by the strict
  reading of C3 the dataset does not carry them, so ❌ — the same rule applied to DAROD
  above. Two lesser issues that would remain even with boxes: RADIal has a **single
  class** against the models' three per-class anchors (a head/anchor reconfiguration),
  and the loader keeps only `x,y,z` from a `N×9` point cloud that does contain Doppler
  and `v` (`dataset.py:651`) — a loader fix, not missing data.

### Summary

- **Supported today (✅):** exactly the 18 wired pairs below (23 entries in
  `eval_func_dict`, counting RODNet's six subtypes individually) — FFTRadNet/RADIal;
  RODNet×6/CRUW; T_RODNet/CRUW; RECORD·RECORDNoLstm·RECORDNoLstmMulti /CRUW & /CARRADA;
  MVRECORD/CARRADA; RADDet/RADDetDataset; TransRAD/RADDetDataset; DAROD/CARRADA & /RADDetDataset;
  RAMP_CNN/UWCR; RadarCrossAttention/CARRADA; PointPillar/ViewOfDelft;
  RadarPillar/ViewOfDelft.
- **Adapter-served datasets:** RADDetDataset, CRUW and CARRADA no longer build samples in
  their dataset class; every model that reads them goes through a registered sample
  adapter (`datasets/adapters/raddet.py`, `cruw.py`, `carrada.py`), so adding a network
  to any of them means adding an adapter, not a branch in `dataset.py`.
- **Astyx and NuScenes** carry the right modality and GT for the **pillar models**, and
  there they are blocked only by a missing training path (+ per-point attribute
  mapping) — enabling `__getitem__` would make them candidates. Every other network is
  ❌mod on them.
- The recurring "possible with work" theme is the **RA/RD/AD confidence-map family**
  (RODNet / RECORD / RAMP_CNN across CARRADA, UWCR, RADIal and RADDetDataset), where
  what is missing is GT rendering (`gt`) or a loader branch (`np`), plus the **pillar
  family** on Astyx / NuScenes (`np`, `feat`). Box-target models fare worst: DAROD and
  the pillar models are ❌ on every dataset that annotates only object **centres**, since
  no amount of wiring invents an extent that was never labelled. CRUW has no ⚠️ cells —
  every CRUW pair is already ✅ or ❌mod.
