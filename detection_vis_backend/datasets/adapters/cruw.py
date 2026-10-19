"""Sample adapters for CRUW (ROD2021): how the RODNet, RECORD and T-RODNet families window
the RA chirps and confidence maps of one sequence.

Moved from CRUW.__getitem__ / prepare_for_train. The parser keeps parse, the raw views
(annotation, confmap, radar_chirp) and the visualisation getters; the window arithmetic,
chirp choice and tensor layout are here.
"""
import os
import random
import time

import numpy as np
import torch

from detection_vis_backend.datasets.adapters import SampleAdapter, register

RODNET_VARIANTS = ("RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2")


class CRUWWindowAdapter(SampleAdapter):
    """One window of win_size frames, step frames apart, stride frames between windows.

    Subclasses set the chirp count and the layout of the radar window and labels.
    """
    DATASET = "CRUW"

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        len(parser)   # a fresh request copy binds its parsed sequence on first use
        self.win_size = train_cfg['win_size']
        if splittype in ('train', 'val') or splittype is None:
            self.step, self.stride = train_cfg['train_step'], train_cfg['train_stride']
        else:
            self.step, self.stride = train_cfg['test_step'], train_cfg['test_stride']
        self.n_class = parser.n_class
        self.chirp_ids = parser.chirp_ids
        self.n_chirps = self._n_chirps(model_cfg)

    def _n_chirps(self, model_cfg):
        return 1

    def __len__(self):
        span = self.win_size * self.step - 1
        n_frame = self.parser.n_frame
        if n_frame <= span:
            return 0
        return (n_frame - span) // self.stride + (1 if (n_frame - span) % self.stride > 0 else 0)

    def _chirp(self):
        """A random chirp per window on the single-chirp paths, every chirp otherwise."""
        if self.n_chirps > 1:
            return list(range(len(self.chirp_ids)))
        return random.randint(0, len(self.chirp_ids) - 1)

    def _frames(self, index):
        data_id = index * self.stride
        return list(range(data_id, data_id + self.win_size * self.step, self.step))

    def _window(self, frames, chirp):
        """(win, R, A, 2) for one chirp, (win, n_chirps, R, A, 2) for a chirp list; float32."""
        if isinstance(chirp, int):
            return np.stack([self.parser.radar_chirp(f, chirp) for f in frames]).astype(np.float32, copy=False)
        return np.stack([np.stack([self.parser.radar_chirp(f, c) for c in chirp]) for f in frames]
                        ).astype(np.float32, copy=False)

    def _labels(self, frames):
        obj_info = [self.parser.annotation(f) for f in frames]
        confmap_gt = np.transpose(np.stack([self.parser.confmap(f) for f in frames]), (1, 0, 2, 3))
        return obj_info, confmap_gt

    def _radar_layout(self, radar_npy_win):
        raise NotImplementedError

    def _gt_layout(self, confmap_gt):
        return confmap_gt

    def __getitem__(self, index):
        data_dict = dict(status=True, seq_name=self.parser.seq_name, image_paths=[])
        chirp = self._chirp()
        frames = self._frames(index)
        data_dict['start_frame'] = frames[0]
        data_dict['end_frame'] = frames[-1]
        try:
            radar_npy_win = self._window(frames, chirp)
            data_dict['image_paths'] = [self.parser.image_paths[f] for f in frames]
        except Exception:
            # in case load npy fail
            data_dict['status'] = False
            log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
            with open(os.path.join(self.parser.feature_path, log_name), 'w') as f_log:
                f_log.write('seq: %s chirp: %s\nframe indices: %d:%d:%d'
                            % (self.parser.seq_name, chirp, frames[0], frames[-1] + self.step, self.step))
            return data_dict

        data_dict['RA'] = self._radar_layout(radar_npy_win)
        if len(self.parser.confmaps) != 0:
            obj_info, confmap_gt = self._labels(frames)
            data_dict.update({'obj_infos': obj_info, 'gt_mask': self._gt_layout(confmap_gt)})
        return data_dict


@register
class RODNetOnCRUW(CRUWWindowAdapter):
    """Numpy (2, win, R, A), or (2, win, n_chirps, R, A) for the M-Net variants."""
    MODELS = RODNET_VARIANTS + ("RadarFormer_hrformer2d",)

    def _n_chirps(self, model_cfg):
        return model_cfg['mnet_cfg'][0] if 'mnet_cfg' in model_cfg else 1

    def _radar_layout(self, radar_npy_win):
        if self.n_chirps > 1:
            return np.transpose(radar_npy_win, (4, 0, 1, 2, 3))
        return np.transpose(radar_npy_win, (3, 0, 1, 2))


@register
class RECORDOnCRUW(CRUWWindowAdapter):
    """Torch (2·4, win, R, A) with the four chirps folded into channels; NoLstm variants
    fold the window too; the label is the window's last frame."""
    MODELS = ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti")

    def _n_chirps(self, model_cfg):
        return 4

    def _radar_layout(self, radar_npy_win):
        win = torch.from_numpy(np.ascontiguousarray(np.transpose(radar_npy_win, (0, 1, 4, 2, 3))))
        w, n, two, h, wd = win.shape
        win = win.reshape(w, n * two, h, wd).transpose(1, 0)
        if self.model_cfg_class == 'RECORDNoLstmMulti':
            c, t, h, w_ = win.shape
            return win.reshape(c * t, h, w_)
        if self.model_cfg_class == 'RECORDNoLstm':
            c, t, h, w_ = win.shape
            assert t == 1
            return win.reshape(c, h, w_)
        return win

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        self.model_cfg_class = model_cfg['class']

    def _gt_layout(self, confmap_gt):
        return confmap_gt[:, -1]


@register
class TRODNetOnCRUW(CRUWWindowAdapter):
    """Numpy (2, 16, 128, 128), single chirp: random in training, chirp 0 on val and test,
    as the reference trains with is_random_chirp=True and tests with False."""
    MODELS = ("T_RODNet",)

    def _chirp(self):
        if self.splittype in ('val', 'test'):
            return 0
        return random.randint(0, len(self.chirp_ids) - 1)

    def _radar_layout(self, radar_npy_win):
        ra = np.transpose(radar_npy_win, (3, 0, 1, 2))
        assert ra.shape == (2, 16, 128, 128), f"T-RODNet reads (2, 16, 128, 128) windows, got {ra.shape}"
        return ra
