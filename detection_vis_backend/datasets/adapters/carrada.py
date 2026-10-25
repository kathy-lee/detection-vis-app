"""Sample adapters for CARRADA: how the RECORD family, MVRECORD, DAROD,
RadarCrossAttention and T-RODNet each read a sequence.

Moved from CARRADA.__getitem__ / prepare_for_train. The parser keeps parse, the raw
views (processed_map, dense_mask, annotation, rca_annotations, get_RA/get_RD and the
cached AD maps), the grids and get_label; the windowing, flips, normalisation and label
encoding are here. Flips and noise draw from numpy's global RNG in the same order as the
old code, so a seeded run reproduces it sample for sample.
"""
import numpy as np
from loguru import logger

from detection_vis_backend.datasets.adapters import SampleAdapter, register
from detection_vis_backend.datasets.utils import (
    get_transformations, VFlip, HFlip, normalize,
    bi_var_gauss, plain_gauss, get_center_map, get_co_vec, get_orent_map,
)

VIEW = {'RD': 'range_doppler', 'RA': 'range_angle', 'AD': 'angle_doppler'}


class CARRADAWindowAdapter(SampleAdapter):
    """win_size consecutive frames ending at frame index + win_size - 1, with the
    vflip/hflip augmentation (train split only) applied identically to every view."""
    DATASET = "CARRADA"

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        len(parser)   # a fresh request copy binds its grids and annotation folder on first use
        self.win_frames = train_cfg['win_size']
        # get_transformations only appends the flips when split == 'train'; splittype None
        # is the 'random' splitmode, where one instance backs every split.
        self.transformations = get_transformations(transform_names=train_cfg['transformations'],
                                                   split=splittype or 'train')
        self.norm_type = train_cfg['norm_type']
        self.add_temp = True

    def __len__(self):
        return self.parser.frame_sync - self.win_frames + 1

    def transform(self, frame, is_vflip=False, is_hflip=False):
        if self.transformations is not None:
            for function in self.transformations:
                if isinstance(function, VFlip):
                    if is_vflip:
                        frame = function(frame)
                    else:
                        continue
                if isinstance(function, HFlip):
                    if is_hflip:
                        frame = function(frame)
                    else:
                        continue
                if not isinstance(function, VFlip) and not isinstance(function, HFlip):
                    frame = function(frame)
        return frame

    def _frame_ids(self, index):
        frame_id = index + self.win_frames - 1
        return frame_id, list(range(frame_id - self.win_frames + 1, frame_id + 1))

    def _stack(self, view, frames):
        matrix = np.dstack([self.parser.processed_map(view, f) for f in frames])
        return np.rollaxis(matrix, axis=-1)

    @staticmethod
    def _draw_flips():
        is_vflip = np.random.uniform(0, 1) > 0.5
        is_hflip = np.random.uniform(0, 1) > 0.5
        return is_vflip, is_hflip

    def _expand(self, matrix):
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                return np.expand_dims(matrix, axis=0)
            assert isinstance(self.add_temp, int)
            return np.expand_dims(matrix, axis=self.add_temp)
        return matrix


@register
class RECORDOnCARRADA(CARRADAWindowAdapter):
    """One view, (1, win, H, W) normalised, dense mask of the window's last frame;
    NoLstm folds the window into channels."""
    MODELS = ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti")

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        assert len(features) == 1 and features[0] in ('RD', 'RA'), f"{model_cfg['class']} reads one of RD/RA, got {features}"
        self.model_type = model_cfg['class']

    def __getitem__(self, index):
        frame_id, frames = self._frame_ids(index)
        featurestr = VIEW[self.features[0]]
        mask = self.parser.dense_mask(featurestr, frame_id)
        feature_frame = {'matrix': self._stack(featurestr, frames), 'gt_mask': mask}
        is_vflip, is_hflip = self._draw_flips()
        feature_frame = self.transform(feature_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        feature_frame['matrix'] = self._expand(feature_frame['matrix'])
        feature_frame['matrix'] = normalize(feature_frame['matrix'], featurestr, norm_type=self.norm_type)
        if self.model_type == 'RECORDNoLstmMulti':
            c, t, h, w = feature_frame['matrix'].shape
            feature_frame['matrix'] = feature_frame['matrix'].reshape(c * t, h, w)
        elif self.model_type == 'RECORDNoLstm':
            c, t, h, w = feature_frame['matrix'].shape
            assert t == 1
            feature_frame['matrix'] = feature_frame['matrix'].reshape(c, h, w)
        return {self.features[0]: feature_frame['matrix'], 'gt_mask': feature_frame['gt_mask']}


@register
class T_RODNetOnCARRADA(CARRADAWindowAdapter):
    """RA window (1, win, H, W) normalised, with the RA dense mask of every frame in the
    window stacked as (n_class, win, H, W): the 3D body predicts a mask per frame, and the
    CRUW recipe it follows supervises every frame of its window."""
    MODELS = ("T_RODNet",)

    def __getitem__(self, index):
        _, frames = self._frame_ids(index)
        mask = np.stack([self.parser.dense_mask('range_angle', f) for f in frames], axis=1)
        feature_frame = {'matrix': self._stack('range_angle', frames), 'gt_mask': mask}
        # Drawn on every split like the sibling adapters, so a seeded run keeps one RNG
        # sequence whatever the configured flips are.
        is_vflip, is_hflip = self._draw_flips()
        feature_frame = self.transform(feature_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        matrix = normalize(self._expand(feature_frame['matrix']), 'range_angle', norm_type=self.norm_type)
        return {'RA': np.asarray(matrix, dtype=np.float32), 'gt_mask': feature_frame['gt_mask']}


@register
class MVRECORDOnCARRADA(CARRADAWindowAdapter):
    """RD, RA and AD windows flipped together and normalised per view; the RD and RA
    dense masks of the window's last frame."""
    MODELS = ("MVRECORD",)

    def __getitem__(self, index):
        frame_id, frames = self._frame_ids(index)
        rd_mask = self.parser.dense_mask('range_doppler', frame_id)
        ra_mask = self.parser.dense_mask('range_angle', frame_id)
        rd_matrices, ra_matrices, ad_matrices = [], [], []
        for f in frames:
            rd_matrices.append(self.parser.processed_map('range_doppler', f))
            ra_matrices.append(self.parser.processed_map('range_angle', f))
            ad_matrices.append(self.parser.processed_map('angle_doppler', f))
        is_vflip, is_hflip = self._draw_flips()

        rd_frame = {'matrix': np.rollaxis(np.dstack(rd_matrices), axis=-1), 'gt_mask': rd_mask}
        rd_frame = self.transform(rd_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        rd_frame['matrix'] = self._expand(rd_frame['matrix'])

        ra_frame = {'matrix': np.rollaxis(np.dstack(ra_matrices), axis=-1), 'gt_mask': ra_mask}
        ra_frame = self.transform(ra_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        ra_frame['matrix'] = self._expand(ra_frame['matrix'])

        # AD has no mask of its own; the RD mask rides along only so the flips can apply.
        ad_frame = {'matrix': np.rollaxis(np.dstack(ad_matrices), axis=-1), 'gt_mask': rd_mask.copy()}
        ad_frame = self.transform(ad_frame, is_vflip=is_vflip, is_hflip=is_hflip)
        ad_frame['matrix'] = self._expand(ad_frame['matrix'])

        rd_frame['matrix'] = normalize(rd_frame['matrix'], 'range_doppler', norm_type=self.norm_type)
        ra_frame['matrix'] = normalize(ra_frame['matrix'], 'range_angle', norm_type=self.norm_type)
        ad_frame['matrix'] = normalize(ad_frame['matrix'], 'angle_doppler', norm_type=self.norm_type)
        return {'RD': rd_frame['matrix'], 'rd_mask': rd_frame['gt_mask'],
                'RA': ra_frame['matrix'], 'ra_mask': ra_frame['gt_mask'],
                'AD': ad_frame['matrix']}


@register
class DARODOnCARRADA(CARRADAWindowAdapter):
    """One view standardised with the train config's data_mean/data_std; boxes are the
    extent of each object's DENSE annotation, normalised to [0, 1] and mirrored with the
    flips; labels are the CARRADA class ids."""
    MODELS = ("DAROD",)

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        assert len(features) == 1 and features[0] in ('RD', 'RA'), f"DAROD reads one of RD/RA, got {features}"
        self.data_mean = train_cfg.get("data_mean")
        self.data_std = train_cfg.get("data_std")

    def __getitem__(self, index):
        frame_id, frames = self._frame_ids(index)
        featurestr = VIEW[self.features[0]]
        mask = self.parser.dense_mask(featurestr, frame_id)
        feature_frame = {'matrix': self._stack(featurestr, frames), 'gt_mask': mask}
        is_vflip, is_hflip = self._draw_flips()
        feature_frame = self.transform(feature_frame, is_vflip=is_vflip, is_hflip=is_hflip)

        gt_boxes, gt_labels = [], []
        h_dim, w_dim = feature_frame['matrix'].shape[-2], feature_frame['matrix'].shape[-1]
        objects = self.parser.annotation(frame_id)
        if objects:
            for obj_anno in objects.values():
                # Box = extent of the DENSE annotation, as the reference builds it
                # (colindecourt/darod, datasets/carrada_builder/carrada.py: _build_bbox);
                # the annotation's own 'box' field is the extent of the SPARSE seeds.
                dense = np.asarray(obj_anno[featurestr]['dense'])
                y1, x1 = dense[:, 0].min(), dense[:, 1].min()
                y2, x2 = dense[:, 0].max(), dense[:, 1].max()
                gt_boxes.append([y1 / h_dim, x1 / w_dim, y2 / h_dim, x2 / w_dim])
                gt_labels.append(obj_anno[featurestr]['label'])
        # Mirror onto the boxes the flips transform() applied to the matrix. For the
        # (win, H, W) matrix HFlip flips axis 1 (range, the y coord) and VFlip axis 2
        # (Doppler, the x coord); a flip applies only when its class is configured.
        for _t in (self.transformations or []):
            if isinstance(_t, HFlip) and is_hflip:
                gt_boxes = [[1.0 - y2, x1, 1.0 - y1, x2] for (y1, x1, y2, x2) in gt_boxes]
            elif isinstance(_t, VFlip) and is_vflip:
                gt_boxes = [[y1, 1.0 - x2, y2, 1.0 - x1] for (y1, x1, y2, x2) in gt_boxes]
        matrix = feature_frame['matrix']
        if self.data_mean is not None and self.data_std is not None:
            matrix = (matrix - self.data_mean) / self.data_std
        return {self.features[0]: matrix, 'label': gt_labels, 'boxes': gt_boxes}


@register
class RadarCrossAttentionOnCARRADA(SampleAdapter):
    """RadarCrossAttention: RA, RD and AD stacks over the frames covered by the
    annotation tracks, per-view standardisation, train-only Gaussian noise, and the
    Gaussian / centre-offset / orientation maps from new_gt_anno.json."""
    MODELS, DATASET = ("RadarCrossAttention",), "CARRADA"

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        len(parser)
        self.win_frames = train_cfg['win_size']
        tracks = parser.rca_annotations()['tracks']
        # Data frame intervals in the sequence, from the annotation tracks.
        intervals = []
        for i, track in enumerate(tracks):
            if i == 0:
                intervals.append(track[0])
            elif i == len(tracks) - 1:
                intervals.append(track[-1])
            else:
                last_frame = tracks[i - 1][0]
                new_frame = track[0]
                if new_frame != last_frame + 1:
                    intervals.append(last_frame + 4)
                    intervals.append(new_frame)
        self.indexmapping = []
        for i in range(int(len(intervals) / 2)):
            self.indexmapping.extend(list(range(intervals[2 * i], intervals[2 * i + 1] + 2 - self.win_frames)))
        self.gauss_type = train_cfg['gauss_type']
        self.center_offset = model_cfg['center_offset']
        self.orientation = model_cfg['orientation']
        # One standardisation constant pair per view (the reference's torchvision
        # Normalize broadcasts over the stack); train-only noise N(0, gauss_noise_std)
        # with probability 1/2 per view.
        self.input_norm = train_cfg.get('input_norm')
        if not self.input_norm:
            logger.warning("RadarCrossAttention: no 'input_norm' in train config -- inputs "
                           "will be fed unnormalized, which the reference never does.")
        self.noise_std = (float(train_cfg.get('gauss_noise_std') or 0.0)
                          if splittype in (None, 'train') else 0.0)

    def __len__(self):
        return len(self.indexmapping)

    def __getitem__(self, index):
        p = self.parser
        raw_index = self.indexmapping[index]
        frame = {'image_path': p.image_filenames[raw_index]}
        ra_map = np.zeros((self.win_frames, 256, 256))
        rd_map = np.zeros((self.win_frames, 256, 64))
        ad_map = np.zeros((self.win_frames, 64, 256))
        for i in range(self.win_frames):
            frame_id = raw_index + i
            ra_map[i, ::] = p.get_RA(frame_id)
            rd_map_temp = p.get_RD(frame_id)
            # flip the axis in the order of increasing velocity (L->R)
            rd_map_temp = np.flip(rd_map_temp, 0)
            rd_map_temp = np.flip(rd_map_temp, 1)
            rd_map_temp[:, 31:34] = 0  # removing dc component
            rd_map[i, ::] = np.float32(rd_map_temp)
            ad_map[i, ::] = p._get_ad_cached(frame_id)
        ra_map = ra_map.astype(np.float32)
        rd_map = rd_map.astype(np.float32)
        ad_map = ad_map.astype(np.float32)
        if self.input_norm:
            for arr, key in ((ra_map, 'ra'), (rd_map, 'rd'), (ad_map, 'ad')):
                arr -= np.float32(self.input_norm[key]['mean'])
                arr /= np.float32(self.input_norm[key]['std'])
        if self.noise_std > 0:
            for arr in (ra_map, rd_map, ad_map):
                if np.random.randint(2):
                    arr += np.random.normal(0.0, self.noise_std, arr.shape).astype(np.float32)
        anno = p.rca_annotations()[str(raw_index).zfill(6)]
        gauss_map = bi_var_gauss(anno) if self.gauss_type == "Bivar" else plain_gauss(anno, s_r=15, s_a=15)
        frame.update({'RD': rd_map, 'RA': ra_map, 'AD': ad_map, 'gt_mask': gauss_map})
        if self.center_offset:
            frame.update({'gt_center_map': get_center_map(anno, vect=get_co_vec())})
        if self.orientation:
            frame.update({'gt_orent_map': get_orent_map(anno)})
        return frame
