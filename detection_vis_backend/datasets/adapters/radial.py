"""Sample adapters for RADIal (the ready-to-use release): how FFTRadNet (RD spectrum) and
SSMRadNet (raw ADC) read it.

Moved from RADIal.__getitem__ / encode. The parser keeps the file lists, labels.csv
(labels, label_dict, sample_keys), get_RD and the freespace crop/resize; everything
model-shaped is here.
"""
import os

import numpy as np
from loguru import logger
from PIL import Image

from detection_vis_backend.datasets.adapters import SampleAdapter, register

# Per-channel mean and std of the 32 RD channels (16 Rx real, then imag), from valeo's
# FFTRadNet release.
RD_INPUT_MEAN = [-2.6244e-03, -2.1335e-01,  1.8789e-02, -1.4427e+00, -3.7618e-01,
                 1.3594e+00, -2.2987e-01,  1.2244e-01,  1.7359e+00, -6.5345e-01,
                 3.7976e-01,  5.5521e+00,  7.7462e-01, -1.5589e+00, -7.2473e-01,
                 1.5182e+00, -3.7189e-01, -8.8332e-02, -1.6194e-01,  1.0984e+00,
                 9.9929e-01, -1.0495e+00,  1.9972e+00,  9.2869e-01,  1.8991e+00,
                 -2.3772e-01,  2.0000e+00,  7.7737e-01,  1.3239e+00,  1.1817e+00,
                 -6.9696e-01,  4.4288e-01]
RD_INPUT_STD = [20775.3809, 23085.5000, 23017.6387, 14548.6357, 32133.5547, 28838.8047,
                27195.8945, 33103.7148, 32181.5273, 35022.1797, 31259.1895, 36684.6133,
                33552.9258, 25958.7539, 29532.6230, 32646.8984, 20728.3320, 23160.8828,
                23069.0449, 14915.9053, 32149.6172, 28958.5840, 27210.8652, 33005.6602,
                31905.9336, 35124.9180, 31258.4316, 31086.0273, 33628.5352, 25950.2363,
                29445.2598, 32885.7422]
# Mean and std of the range / angle offsets inside a detection cell.
REG_MEAN = [0.4048094369863972, 0.3997392847799934]
REG_STD = [0.6968599580482511, 0.6942950877813826]

# valeo's sequence split ("original" splitmode): 4 val and 4 test recordings, the rest train.
VALEO_SEQUENCES = {'val': ['RECORD@2020-11-22_12.49.56', 'RECORD@2020-11-22_12.11.49',
                           'RECORD@2020-11-22_12.28.47', 'RECORD@2020-11-21_14.25.06'],
                   'test': ['RECORD@2020-11-22_12.45.05', 'RECORD@2020-11-22_12.25.47',
                            'RECORD@2020-11-22_12.03.47', 'RECORD@2020-11-22_12.54.38']}


def radial_sequence_split(parser):
    """{'train', 'val', 'test'}: parser frame indices of valeo's sequence split."""
    labels = parser.labels
    dict_index_to_keys = {s: i for i, s in enumerate(parser.sample_keys)}
    split = {}
    for name in ('val', 'test'):
        indexes = []
        for seq in VALEO_SEQUENCES[name]:
            idx = np.where(labels[:, 14] == seq)[0]
            indexes.append(labels[idx, 0])
        split[name] = [dict_index_to_keys[k] for k in np.unique(np.concatenate(indexes))]
    split['train'] = np.setdiff1d(np.arange(len(parser)), np.concatenate([split['val'], split['test']]))
    return split


def radial_box_labels(rows):
    """labels.csv rows of one frame -> float32 (N, 10).

    labels.csv columns, in order:
    0:numSample 1:x1_pix 2:y1_pix 3:x2_pix 4:y2_pix 5:laser_X_m 6:laser_Y_m 7:laser_Z_m
    8:radar_X_m 9:radar_Y_m 10:radar_R_m 11:radar_A_deg 12:radar_D_mps 13:radar_P_db
    14:dataset 15:index 16:Difficult
    Selected as [Range, Angle, Doppler, laser_X_m, laser_Y_m, laser_Z_m, x1_pix, y1_pix, x2_pix, y2_pix].
    """
    return rows[:, [10, 11, 12, 5, 6, 7, 1, 2, 3, 4]].astype(np.float32)


def radial_freespace(parser, index):
    """Freespace label of frame index: bool (256, 224)."""
    sample_id = parser.sample_keys[index]
    segmap_name = os.path.join(parser.root_dir, 'radar_Freespace', "freespace_{:06d}.png".format(sample_id))
    segmap = Image.open(segmap_name)  # [512,900]
    # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
    # We crop the fov to 89.6deg
    segmap = parser.crop(segmap)
    # and we resize to half of its size
    return np.asarray(parser.resize(segmap)) == 255


def encode_radial_label(labels):
    """Box labels (N, 10) -> FFTRadNet's detection target (3, 128, 224): [cls, range, angle]."""
    geometry = {
        "ranges": [512, 896, 1],
        "resolution": [0.201171875, 0.2],
        "size": 3
    }
    regression_layer = 2

    INPUT_DIM = (geometry['ranges'][0], geometry['ranges'][1], geometry['ranges'][2])
    OUTPUT_DIM = (regression_layer + 1, INPUT_DIM[0] // 4, INPUT_DIM[1] // 4)

    map = np.zeros(OUTPUT_DIM)

    for lab in labels:
        # [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]

        if(lab[0] == -1):
            continue

        range_bin = int(np.clip(lab[0]/geometry['resolution'][0]/4, 0, OUTPUT_DIM[1]))
        range_mod = lab[0] - range_bin*geometry['resolution'][0]*4

        # ANgle and deg
        angle_bin = int(np.clip(np.floor(lab[1]/geometry['resolution'][1]/4 + OUTPUT_DIM[2]/2), 0, OUTPUT_DIM[2]))
        angle_mod = lab[1] - (angle_bin - OUTPUT_DIM[2]/2)*geometry['resolution'][1]*4

        if(geometry['size'] == 1):
            map[0, range_bin, angle_bin] = 1
            map[1, range_bin, angle_bin] = (range_mod - REG_MEAN[0]) / REG_STD[0]
            map[2, range_bin, angle_bin] = (angle_mod - REG_MEAN[1]) / REG_STD[1]
        else:

            s = int((geometry['size']-1)/2)
            r_lin = np.linspace(geometry['resolution'][0]*s, -geometry['resolution'][0]*s,
                                geometry['size'])*4
            a_lin = np.linspace(geometry['resolution'][1]*s, -geometry['resolution'][1]*s,
                                geometry['size'])*4

            px_a, px_r = np.meshgrid(a_lin, r_lin)

            if(angle_bin >= s and angle_bin < (OUTPUT_DIM[2]-s)):
                map[0, range_bin-s:range_bin+(s+1), angle_bin-s:angle_bin+(s+1)] = 1
                map[1, range_bin-s:range_bin+(s+1), angle_bin-s:angle_bin+(s+1)] = ((px_r+range_mod) - REG_MEAN[0]) / REG_STD[0]
                map[2, range_bin-s:range_bin+(s+1), angle_bin-s:angle_bin+(s+1)] = ((px_a + angle_mod) - REG_MEAN[1]) / REG_STD[1]
            elif(angle_bin < s):
                map[0, range_bin-s:range_bin+(s+1), 0:angle_bin+(s+1)] = 1
                map[1, range_bin-s:range_bin+(s+1), 0:angle_bin+(s+1)] = ((px_r[:, s-angle_bin:] + range_mod) - REG_MEAN[0]) / REG_STD[0]
                map[2, range_bin-s:range_bin+(s+1), 0:angle_bin+(s+1)] = ((px_a[:, s-angle_bin:] + angle_mod) - REG_MEAN[1]) / REG_STD[1]

            elif(angle_bin >= OUTPUT_DIM[2]):
                end = s+(OUTPUT_DIM[2]-angle_bin)
                map[0, range_bin-s:range_bin+(s+1), angle_bin-s:] = 1
                map[1, range_bin-s:range_bin+(s+1), angle_bin-s:] = ((px_r[:, :end] + range_mod) - REG_MEAN[0]) / REG_STD[0]
                map[2, range_bin-s:range_bin+(s+1), angle_bin-s:] = ((px_a[:, :end] + angle_mod) - REG_MEAN[1]) / REG_STD[1]

    return map


@register
class FFTRadNetOnRADIal(SampleAdapter):
    """Normalized RD spectrum (32, 512, 256), the encoded detection map, the box labels and
    optionally the freespace map."""
    MODELS, DATASET = ("FFTRadNet",), "RADIal"

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        self.segmentation_head = True if model_cfg['segmentation_head'] else False

    def __getitem__(self, index):
        box_labels = radial_box_labels(self.parser.annotation(index))
        encoded_label = encode_radial_label(box_labels).copy()

        radar_FFT = self.parser.get_RD(index)
        for i in range(len(RD_INPUT_MEAN)):
            radar_FFT[..., i] -= RD_INPUT_MEAN[i]
            radar_FFT[..., i] /= RD_INPUT_STD[i]
        radar_FFT = np.transpose(radar_FFT, axes=(2, 0, 1))
        data_dict = {'RD': radar_FFT, 'encoded_label': encoded_label, 'box_label': box_labels}

        if self.segmentation_head:
            data_dict.update({'seg_label': radial_freespace(self.parser, index)})
        return data_dict


def decode_radial_adc(raw):
    """(1048576, 4) int16 chip streams -> float32 (512 samples, 256 chirps, 32): the 16 Rx
    (chips 3, 0, 1, 2, each I/Q interleaved) real parts, then imaginary parts. As the
    SSMRadNet authors' convert_radar_cube: no DC removal, no window, no scaling."""
    frames = [np.reshape(raw[0::2, c] + 1j * raw[1::2, c], (512, 4, 256), order='F').transpose((0, 2, 1))
              for c in range(4)]
    cube = np.concatenate([frames[3], frames[0], frames[1], frames[2]], axis=2)
    return np.concatenate([cube.real, cube.imag], axis=2).astype(np.float32)


@register
class SSMRadNetOnRADIal(SampleAdapter):
    """Raw ADC (512, 256, 32), with FFTRadNet's encoded detection map, box labels and
    freespace map. Serves only the frames whose radar_ADC/adc_{numSample:06d}.npy exists:
    the ADC is extracted from the raw recordings separately and may cover part of the set."""
    MODELS, DATASET = ("SSMRadNet",), "RADIal"

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        self.segmentation_head = True if model_cfg['segmentation_head'] else False
        self.adc_dir = os.path.join(parser.root_dir, 'radar_ADC')
        if not os.path.isdir(self.adc_dir):
            raise FileNotFoundError(f"{self.adc_dir} not found: SSMRadNet reads RADIal's raw ADC "
                                    "(written by scripts/extract_radial_adc.py)")
        present = set(os.listdir(self.adc_dir))
        self.frames = [i for i, key in enumerate(parser.sample_keys) if f"adc_{key:06d}.npy" in present]
        if not self.frames:
            raise FileNotFoundError(f"{self.adc_dir} holds no adc file of the {len(parser)} labelled frames")
        logger.info(f"RADIal ADC: kept {len(self.frames)} of {len(parser)} frames")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        i = self.frames[index]
        box_labels = radial_box_labels(self.parser.annotation(i))
        encoded_label = encode_radial_label(box_labels).copy()
        adc = decode_radial_adc(np.load(os.path.join(self.adc_dir, f"adc_{self.parser.sample_keys[i]:06d}.npy")))
        data_dict = {'ADC': adc, 'encoded_label': encoded_label, 'box_label': box_labels}
        if self.segmentation_head:
            data_dict.update({'seg_label': radial_freespace(self.parser, i)})
        return data_dict
