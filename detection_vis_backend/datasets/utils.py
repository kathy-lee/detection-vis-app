import numpy as np
import logging
import math
import scipy
import matplotlib.pyplot as plt


from skimage import transform



logger = logging.getLogger()

def reshape_frame(data, samples_per_chirp, n_receivers, n_tdm, n_chirps_per_frame):
  _data = data.reshape(-1, 8)
  _data = _data[:, :4] + 1j * _data[:, 4:]
  _data = _data.reshape(n_chirps_per_frame, samples_per_chirp, n_receivers)

  #deinterleve if theres TDM
  if n_tdm > 1:
    _data_i = [_data[i::n_tdm, :, :] for i in range(n_tdm)]
    _data = np.concatenate(_data_i, axis=-1)

  return _data


def read_radar_params(filename):
    """Reads a text file containing serial commands and returns parsed config as a dictionary"""
    with open(filename) as cfg:
        iwr_cmds = cfg.readlines()
        iwr_cmds = [x.strip() for x in iwr_cmds]
        radar_cfg = parse_commands(iwr_cmds)

    logger.debug(radar_cfg)
    return radar_cfg


def parse_commands(commands):
    """Calls the corresponding parser for each command in commands list"""
    cfg = None
    for line in commands:
        try:
            cmd = line.split()[0]
            args = line.split()[1:]
            cfg = command_handlers[cmd](args, cfg)
        except KeyError:
            logger.debug(f'{cmd} is not handled')
        except IndexError:
            logger.debug(f'line is empty "{line}"')
    return cfg


def dict_to_list(cfg):
    """Generates commands from config dictionary"""
    cfg_list = ['flushCfg','dfeDataOutputMode 1']

    # rx antennas/lanes for channel config
    rx_bool = [cfg['rx4'], cfg['rx3'], cfg['rx2'], cfg['rx1']]
    rx_mask = sum(2 ** i for i, v in enumerate(reversed(rx_bool)) if v)
    # number of tx antennas for channel config
    tx_bool = [cfg['tx3'], cfg['tx2'], cfg['tx1']]
    tx_mask = sum(2 ** i for i, v in enumerate(reversed(tx_bool)) if v)
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) is False else 0
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is False and (cfg['tx1'] or cfg['tx3']) is True else 0
    #print('[NOTE] Elevation and Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) else 0
    cfg_list.append('channelCfg %s %s 0' % (rx_mask, tx_mask))  # rx and tx mask

    # adc config
    if cfg['isComplex'] and cfg['image_band']:
        outputFmt = 2
        #print('[NOTE] Complex 2x mode, both Imaginary and Real IF spectrum is filtered and sent to ADC, so\n'
        #      '       if Sampling rate is X, ADC data would include frequency spectrum from -X/2 to X/2.')
    elif cfg['isComplex'] and not cfg['image_band'] == True:
        outputFmt = 1
        #print('[NOTE] Complex 1x mode, Only Real IF Spectrum is filtered and sent to ADC, so if Sampling rate\n'
        #      '       is X, ADC data would include frequency spectrum from 0 to X.')
    else: raise ValueError("Real Data Type Not Supported")
    cfg_list.append('adcCfg 2 %s' % outputFmt)  # 16 bits (mandatory), complex 1x or 2x

    # adc power
    if cfg['adcPower'] =='low':
        power_mode = 1
        #print('[NOTE] The Low power ADC mode limits the sampling rate to half the max value.')
    elif cfg['adcPower'] =='regular': power_mode = 0
    else: raise ValueError("ADC power level Not Supported")
    cfg_list.append('lowPower 0 %s' % power_mode)  # power mode

    # profile configs
    for profile_ii in cfg['profiles']:
        cfg_list.append('profileCfg %s %s %s %s %s %s %s %s %s %s %s %s %s %s'
                % (profile_ii['id'],
                float(profile_ii['start_frequency']/1e9),
                float(profile_ii['idle']/1e-6),
                float(profile_ii['adcStartTime']/1e-6),
                float(profile_ii['rampEndTime']/1e-6),
                int(profile_ii['txPower']),
                int(profile_ii['txPhaseShift']),
                float(profile_ii['freqSlopeConst']/1e12),
                float(profile_ii['txStartTime']/1e-6),
                int(profile_ii['adcSamples']),
                int(profile_ii['adcSampleRate']/1e3),
                int(profile_ii['hpfCornerFreq1']),
                int(profile_ii['hpfCornerFreq2']),
                int(profile_ii['rxGain'])))

    # chirp configs
    for chirp_ii in cfg['chirps']:

        # Check if chirp is referring to valid profile config
        profile_valid = False
        for profile_ii in cfg['profiles']:
            if chirp_ii['profileID'] == profile_ii['id']: profile_valid = True
        if profile_valid is False: raise ValueError("The following profile id used in chirp "
                                                    "is invalid: %i" % chirp_ii['profileID'])
        ###############################################################################################################
        '''
        # check if tx values are valid
        if hamming([chirp_ii['chirptx3'],chirp_ii['chirptx2'],chirp_ii['chirptx1']],
            [cfg['tx3'], cfg['tx2'], cfg['tx1']])*3 > 1:
            raise ValueError("Chirp should have at most one different Tx than channel cfg")
        '''
        ###############################################################################################################
        if chirp_ii['chirpStartIndex'] > chirp_ii['chirpStopIndex']: raise ValueError("Particular chirp start index after chirp stop index")
        tx_bool = [chirp_ii['chirptx3'],chirp_ii['chirptx2'],chirp_ii['chirptx1']]
        tx_mask = sum(2 ** i for i, v in enumerate(reversed(tx_bool)) if v)
        cfg_list.append('chirpCfg %s %s %s %s %s %s %s %s'
                % (chirp_ii['chirpStartIndex'],
                   chirp_ii['chirpStopIndex'],
                   chirp_ii['profileID'],
                   chirp_ii['startFreqVariation'],
                   chirp_ii['slopeVariation'],
                   chirp_ii['idleVariation'],
                   chirp_ii['adcStartVariation'],
                   tx_mask))

    # frame config
    chirpStop = 0
    chirpStart = 511  # max value for chirp start index
    for chirp_ii in cfg['chirps']:
        chirpStop = max(chirpStop, chirp_ii['chirpStopIndex'])
        chirpStart = min(chirpStart,chirp_ii['chirpStartIndex'])
    chirps_len  = chirpStop + 1

    numLoops = cfg['numChirps']/chirps_len
    if chirpStart > chirpStop: raise ValueError("Chirp(s) start index is after chirp stop index")
    if numLoops % 1 != 0: raise ValueError("Number of loops is not integer")
    if numLoops > 255 or numLoops < 1: raise ValueError("Number of loops must be int in [1,255]")

    numFrames = cfg['numFrames'] if 'numFrames' in cfg.keys() else 0  # if zero => inf

    cfg_list.append('frameCfg %s %s %s %s %s 1 0'
            % (chirpStart, chirpStop, int(numLoops), numFrames, 1000/cfg['fps']))

    cfg_list.append('testFmkCfg 0 0 0 1')
    cfg_list.append('setProfileCfg disable ADC disable')
    return cfg_list


def channelStr_to_dict(args, curr_cfg=None):
    """Handler for `channelcfg`"""

    if curr_cfg:
        cfg = curr_cfg
    else:
        cfg = {}

    # This is the number of receivers which is equivalent to the number of lanes in the source code
    # Later, may include the result from the number of transmitters
    rx_bin = bin(int(args[0]))[2:].zfill(4)
    cfg['numLanes'] = len([ones for ones in rx_bin if ones == '1'])
    (cfg['rx4'],cfg['rx3'],cfg['rx2'],cfg['rx1']) = [bool(int(ones)) for ones in rx_bin]

    # This is the number of transmitters
    tx_bin = bin(int(args[1]))[2:].zfill(3)
    cfg['numTx'] = len([ones for ones in tx_bin if ones == '1'])
    (cfg['tx3'], cfg['tx2'], cfg['tx1']) = [bool(int(ones)) for ones in tx_bin]
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) is False else 0
    #print('[NOTE] Azimuth angle can be determined from channel config.') if cfg['tx2'] is False and (cfg['tx1'] or cfg['tx3']) is True else 0
    #print('[NOTE] Elevation and Azimuth angle can be determined from channel config.') if cfg['tx2'] is True and (cfg['tx1'] or cfg['tx3']) else 0


    return cfg


def profileStr_to_dict(args, curr_cfg=None):
    """Handler for `profileCfg`"""
    normalizer = [None, 1e9, 1e-6, 1e-6, 1e-6, None, None, 1e12, 1e-6, None, 1e3, None, None, None]
    dtype = [int, float, float, float, float, float, float, float, float, int, float, int, int, float]
    keys = ['id',
            'start_frequency',
            'idle',
            'adcStartTime',
            'rampEndTime',
            'txPower',
            'txPhaseShift',
            'freqSlopeConst',
            'txStartTime',
            'adcSamples',
            'adcSampleRate',
            'hpfCornerFreq1',
            'hpfCornerFreq2',
            'rxGain',
            ]
    # Check if the main dictionary exists
    if curr_cfg:
        cfg = curr_cfg
        if 'profiles' not in cfg.keys():
            cfg['profiles']=[]
    else:
        cfg = {'profiles': []}

    profile_dict = {}
    for k, v, n, d in zip(keys, args, normalizer, dtype):
        profile_dict[k] = d(float(v) * n if n else v)

    cfg['profiles'].append(profile_dict)
    return cfg


def chirp_to_dict(args,curr_cfg=None):
    """Handler for `chirpCfg`"""
    if curr_cfg:
        cfg = curr_cfg
        if 'chirps' not in cfg.keys():
            cfg['chirps'] = []
    else:
        cfg = {'chirps': []}

    chirp_dict = {}
    chirp_dict['chirpStartIndex'] = int(args[0])
    chirp_dict['chirpStopIndex'] = int(args[1])
    chirp_dict['profileID'] = int(args[2])
    chirp_dict['startFreqVariation'] = float(args[3])
    chirp_dict['slopeVariation'] = float(args[4])
    chirp_dict['idleVariation'] = float(args[5])
    chirp_dict['adcStartVariation'] = float(args[6])

    tx_bin = bin(int(args[7]))[2:].zfill(3)
    (chirp_dict['chirptx3'], chirp_dict['chirptx2'], chirp_dict['chirptx1']) = [bool(int(ones)) for ones in tx_bin]

    cfg['chirps'].append(chirp_dict)
    return cfg


def power_to_dict(args,curr_cfg=None):
    """handler for `lowPower`"""
    if curr_cfg:
        cfg = curr_cfg
    else:
        cfg = {}
    if int(args[1]) ==1:
        cfg['adcPower'] = 'low'
        #print('[NOTE] The Low power ADC mode limits the sampling rate to half the max value.')
    elif int(args[1]) ==0:
        cfg['adcPower'] = 'regular'
    else:
        raise ValueError ("Invalid Power Level")
    return cfg


def frameStr_to_dict(args, cfg):
    """Handler for `frameCfg`"""

    # Number of chirps
    if 'chirps' not in cfg.keys():
        raise ValueError("Need to define chirps before frame")

    chirpStop =0
    for ii in range(len(cfg['chirps'])):
        chirpStop = max(chirpStop,cfg['chirps'][ii]['chirpStopIndex'])
    chirps_len = chirpStop + 1

    cfg['numChirps'] = int(args[2]) * chirps_len  # num loops * len(chirps)
    if int(args[3]) != 0: cfg['numFrames'] = int(args[3])

    # args[4] is the time in milliseconds of each frame
    cfg['fps'] = 1000/float(args[4])


    return cfg


def adcStr_to_dict(args, curr_cfg=None):
    """Handler for `adcCfg`"""
    if curr_cfg:
        cfg = curr_cfg
    else:
        cfg = {}

    if int(args[1]) == 1:
        cfg['isComplex'] = True
        cfg['image_band'] = False
        #print('[NOTE] Complex 1x mode, Only Real IF Spectrum is filtered and sent to ADC, so if Sampling rate\n'
        #      '       is X, ADC data would include frequency spectrum from 0 to X.')
    elif int(args[1]) == 2:
        cfg['isComplex'] = True
        cfg['image_band'] = True
        #print('[NOTE] Complex 2x mode, both Imaginary and Real IF spectrum is filtered and sent to ADC, so\n'
        #      '       if Sampling rate is X, ADC data would include frequency spectrum from -X/2 to X/2.')
    else:
        raise ValueError("Real Data Type Not Supported")

    return cfg


#Mapping of serial command to command handler
command_handlers = {
    'channelCfg': channelStr_to_dict,
    'profileCfg': profileStr_to_dict,
    'chirpCfg': chirp_to_dict,
    'frameCfg': frameStr_to_dict,
    'adcCfg': adcStr_to_dict,
    'lowPower': power_to_dict,
}


def gen_steering_vec(ang_est_range, ang_est_resolution, num_ant):
    """Generate a steering vector for AOA estimation given the theta range, theta resolution, and number of antennas

    Defines a method for generating steering vector data input --Python optimized Matrix format
    The generated steering vector will span from -angEstRange to angEstRange with increments of ang_est_resolution
    The generated steering vector should be used for all further AOA estimations (bartlett/capon)

    Args:
        ang_est_range (int): The desired span of thetas for the angle spectrum.
        ang_est_resolution (float): The desired resolution in terms of theta
        num_ant (int): The number of Vrx antenna signals captured in the RDC

    Returns:
        num_vec (int): Number of vectors generated (integer divide angEstRange/ang_est_resolution)
        steering_vectors (ndarray): The generated 2D-array steering vector of size (num_vec,num_ant)

    Example:
        >>> #This will generate a numpy array containing the steering vector with 
        >>> #angular span from -90 to 90 in increments of 1 degree for a 4 Vrx platform
        >>> _, steering_vec = gen_steering_vec(90,1,4)

    """
    num_vec = (2 * ang_est_range / ang_est_resolution + 1)
    num_vec = int(round(num_vec))
    steering_vectors = np.zeros((num_vec, num_ant), dtype='complex64')
    for kk in range(num_vec):
        for jj in range(num_ant):
            mag = -1 * np.pi * jj * np.sin((-ang_est_range + kk * ang_est_resolution) * np.pi / 180)
            real = np.cos(mag)
            imag = np.sin(mag)

            steering_vectors[kk, jj] = complex(real, imag)

    return [num_vec, steering_vectors]


def peak_search_full_variance(doa_spectrum, steering_vec_size, sidelobe_level=0.251188643150958, gamma=1.2):
    """ Performs peak search (TI's full search) will retaining details about each peak including
    each peak's width, location, and value.

    Args:
        doa_spectrum (ndarray): a 1D numpy array containing the power spectrum generated via some aoa method (naive,
        bartlett, or capon)
        steering_vec_size (int): Size of the steering vector in terms of number of theta bins
        sidelobe_level (float): A low value threshold used to avoid sidelobe detections as peaks
        gamma (float): Weight to determine when a peak will pass as a true peak

    Returns:
        peak_data (ndarray): A 1D numpy array of custom data types with length numberOfPeaksDetected.
        Each detected peak is organized as [peak_location, peak_value, peak_width]
        total_power (float): The total power of the spectrum. Used for variance calculations
    """
    peak_threshold = max(doa_spectrum) * sidelobe_level

    # Multiple Peak Search
    running_index = 0
    num_max = 0
    extend_loc = 0
    init_stage = True
    max_val = 0
    total_power = 0
    max_loc = 0
    max_loc_r = 0
    min_val = np.inf
    locate_max = False

    peak_data = []

    while running_index < (steering_vec_size + extend_loc):
        if running_index >= steering_vec_size:
            local_index = running_index - steering_vec_size
        else:
            local_index = running_index

        # Pull local_index values
        current_val = doa_spectrum[local_index]
        # Record Min & Max locations
        if current_val > max_val:
            max_val = current_val
            max_loc = local_index
            max_loc_r = running_index

        if current_val < min_val:
            min_val = current_val

        if locate_max:
            if current_val < max_val / gamma:
                if max_val > peak_threshold:
                    bandwidth = running_index - max_loc_r
                    obj = dict.fromkeys(['peakLoc', 'peakVal', 'peakWid'])
                    obj['peakLoc'] = max_loc
                    obj['peakVal'] = max_val
                    obj['peakWid'] = bandwidth
                    peak_data.append(obj)
                    total_power += max_val
                    num_max += 1
                min_val = current_val
                locate_max = False
        else:
            if current_val > min_val * gamma:
                locate_max = True
                max_val = current_val
                if init_stage:
                    extend_loc = running_index
                    init_stage = False

        running_index += 1

    peak_data = np.array(peak_data)
    return peak_data, total_power


def load_anno_txt(txt_path, n_frame, radar_cfg):
    folder_name_dict = dict(
        cam_0='IMAGES_0',
        rad_h='RADAR_RA_H'
    )
    anno_dict = init_meta_json(n_frame, folder_name_dict)

    range_grid = confmap2ra(radar_cfg, name='range')
    angle_grid = confmap2ra(radar_cfg, name='angle')
    
    with open(txt_path, 'r') as f:
        data = f.readlines()
    for line in data:
        frame_id, r, a, class_name = line.rstrip().split()
        frame_id = int(frame_id)
        r = float(r)
        a = float(a)
        rid, aid = ra2idx(r, a, range_grid, angle_grid)
        anno_dict[frame_id]['rad_h']['n_objects'] += 1
        anno_dict[frame_id]['rad_h']['obj_info']['categories'].append(class_name)
        anno_dict[frame_id]['rad_h']['obj_info']['centers'].append([r, a])
        anno_dict[frame_id]['rad_h']['obj_info']['center_ids'].append([rid, aid])
        anno_dict[frame_id]['rad_h']['obj_info']['scores'].append(1.0)

    return anno_dict


def get_class_id(class_str, classes):
    if class_str in classes:
        class_id = classes.index(class_str)
    else:
        if class_str == '':
            raise ValueError("No class name found")
        else:
            class_id = -1000
    return class_id


def generate_confmap(n_obj, obj_info, radar_configs, gaussian_thres=36):
    """
    Generate confidence map a radar frame.
    :param n_obj: number of objects in this frame
    :param obj_info: obj_info includes metadata information
    :param dataset: dataset object
    :param config_dict: rodnet configurations
    :param gaussian_thres: threshold for gaussian distribution in confmaps
    :return: generated confmap
    """

    confmap_cfg = dict(
        confmap_sigmas={
            'pedestrian': 15,
            'cyclist': 20,
            'car': 30,
            # 'van': 40,
            # 'truck': 50,
        },
        confmap_sigmas_interval={
            'pedestrian': [5, 15],
            'cyclist': [8, 20],
            'car': [10, 30],
            # 'van': [15, 40],
            # 'truck': [20, 50],
        },
        confmap_length={
            'pedestrian': 1,
            'cyclist': 2,
            'car': 3,
            # 'van': 4,
            # 'truck': 5,
        }
    )
    n_class = 3 # dataset.object_cfg.n_class
    classes = ["pedestrian", "cyclist", "car"] # dataset.object_cfg.classes
    confmap_sigmas = confmap_cfg['confmap_sigmas']
    confmap_sigmas_interval = confmap_cfg['confmap_sigmas_interval']
    confmap_length = confmap_cfg['confmap_length']

    range_grid = confmap2ra(radar_configs, name='range')
    angle_grid = confmap2ra(radar_configs, name='angle')

    confmap = np.zeros((n_class, radar_configs['ramap_rsize'], radar_configs['ramap_asize']), dtype=float)
    for objid in range(n_obj):
        rng_idx = obj_info['center_ids'][objid][0]
        agl_idx = obj_info['center_ids'][objid][1]
        class_name = obj_info['categories'][objid]
        if class_name not in classes:
            # print("not recognized class: %s" % class_name)
            continue
        class_id = get_class_id(class_name, classes)
        sigma = 2 * np.arctan(confmap_length[class_name] / (2 * range_grid[rng_idx])) * confmap_sigmas[class_name]
        sigma_interval = confmap_sigmas_interval[class_name]
        if sigma > sigma_interval[1]:
            sigma = sigma_interval[1]
        if sigma < sigma_interval[0]:
            sigma = sigma_interval[0]
        for i in range(radar_configs['ramap_rsize']):
            for j in range(radar_configs['ramap_asize']):
                distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma ** 2
                if distant < gaussian_thres:  # threshold for confidence maps
                    value = np.exp(- distant / 2) / (2 * math.pi)
                    confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]

    return confmap


def normalize_confmap(confmap):
    conf_min = np.min(confmap)
    conf_max = np.max(confmap)
    if conf_max - conf_min != 0:
        confmap_norm = (confmap - conf_min) / (conf_max - conf_min)
    else:
        confmap_norm = confmap
    return confmap_norm


def add_noise_channel(confmap, radar_configs):
    n_class = 3 # dataset.object_cfg.n_class

    confmap_new = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']), dtype=float)
    confmap_new[:n_class, :, :] = confmap
    conf_max = np.max(confmap, axis=0)
    confmap_new[n_class, :, :] = 1.0 - conf_max
    return confmap_new


def visualize_confmap(confmap, pps=[]):
    if len(confmap.shape) == 2:
        plt.imshow(confmap, origin='lower', aspect='auto')
        for pp in pps:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.show()
        return
    else:
        n_channel, _, _ = confmap.shape
    if n_channel == 3:
        confmap_viz = np.transpose(confmap, (1, 2, 0))
    elif n_channel > 3:
        confmap_viz = np.transpose(confmap[:3, :, :], (1, 2, 0))
        if n_channel == 4:
            confmap_noise = confmap[3, :, :]
            plt.imshow(confmap_noise, origin='lower', aspect='auto')
            plt.show()
    else:
        print("Warning: wrong shape of confmap!")
        return
    plt.imshow(confmap_viz, origin='lower', aspect='auto')
    for pp in pps:
        plt.scatter(pp[1], pp[0], s=5, c='white')
    plt.show()


def generate_confmaps(metadata_dict, n_class, viz, radar_configs):
    confmaps = []
    for metadata_frame in metadata_dict:
        n_obj = metadata_frame['rad_h']['n_objects']
        obj_info = metadata_frame['rad_h']['obj_info']
        if n_obj == 0:
            confmap_gt = np.zeros(
                (n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']),
                dtype=float)
            confmap_gt[-1, :, :] = 1.0  # initialize noise channal
        else:
            confmap_gt = generate_confmap(n_obj, obj_info, radar_configs)
            confmap_gt = normalize_confmap(confmap_gt)
            confmap_gt = add_noise_channel(confmap_gt, radar_configs)
        assert confmap_gt.shape == (
            n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
        if viz:
            visualize_confmap(confmap_gt)
        confmaps.append(confmap_gt)
    confmaps = np.array(confmaps)
    return confmaps


def init_meta_json(n_frames, 
                   imwidth=1440, imheight=864,
                   rarange=128, raazimuth=128, n_chirps=255):
    folder_name_dict = dict(
        cam_0='IMAGES_0',
        cam_1='IMAGES_1',
        rad_h='RADAR_RA_H'
    )
    meta_all = []
    for frame_id in range(n_frames):
        meta_dict = dict(frame_id=frame_id)
        for key in folder_name_dict.keys():
            if key.startswith('cam'):
                meta_dict[key] = init_camera_json(folder_name_dict[key], imwidth, imheight)
            elif key.startswith('rad'):
                meta_dict[key] = init_radar_json(folder_name_dict[key], rarange, raazimuth, n_chirps)
            else:
                raise NotImplementedError
        meta_all.append(meta_dict)
    return meta_all


def init_camera_json(folder_name, width, height):
    return dict(
        folder_name=folder_name,
        frame_name=None,
        width=width,
        height=height,
        n_objects=0,
        obj_info=dict(
            anno_source=None,
            categories=[],
            bboxes=[],
            scores=[],
            masks=[]
        )
    )


def init_radar_json(folder_name, range, azimuth, n_chirps):
    return dict(
        folder_name=folder_name,
        frame_name=None,
        range=range,
        azimuth=azimuth,
        n_chirps=n_chirps,
        n_objects=0,
        obj_info=dict(
            anno_source=None,
            categories=[],
            centers=[],
            center_ids=[],
            scores=[]
        )
    )


def ra2idx(rng, agl, range_grid, angle_grid):
    """Mapping from absolute range (m) and azimuth (rad) to ra indices."""
    rng_id, _ = find_nearest(range_grid, rng)
    agl_id, _ = find_nearest(angle_grid, agl)
    return rng_id, agl_id


def find_nearest(array, value):
    """Find nearest value to 'value' in 'array'."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def confmap2ra(radar_configs, name, radordeg='rad'):
    """
    Map confidence map to range(m) and angle(deg): not uniformed angle
    :param radar_configs: radar configurations
    :param name: 'range' for range mapping, 'angle' for angle mapping
    :param radordeg: choose from radius or degree for angle grid
    :return: mapping grids
    """
    # TODO: add more args for different network settings
    Fs = radar_configs['sample_freq']
    sweepSlope = radar_configs['sweep_slope']
    num_crop = radar_configs['crop_num']
    fft_Rang = radar_configs['ramap_rsize'] + 2 * num_crop
    fft_Ang = radar_configs['ramap_asize']
    c = scipy.constants.speed_of_light

    if name == 'range':
        freq_res = Fs / fft_Rang
        freq_grid = np.arange(fft_Rang) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
        return rng_grid

    if name == 'angle':
        # for [-90, 90], w will be [-1, 1]
        w = np.linspace(math.sin(math.radians(radar_configs['ra_min'])),
                        math.sin(math.radians(radar_configs['ra_max'])),
                        radar_configs['ramap_asize'])
        if radordeg == 'deg':
            agl_grid = np.degrees(np.arcsin(w))  # rad to deg
        elif radordeg == 'rad':
            agl_grid = np.arcsin(w)
        else:
            raise TypeError
        return agl_grid


def is_float(str_val):
    try:
        float(str_val)
        return True
    except ValueError:
        return False

def read_pointcloudfile(filename):
    pc = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip()
            if line:
                line_str = line.split()
                if is_float(line_str[0]):
                    line_float = [float(x) for x in line_str]
                    pc.append(line_float)
    pa = np.array(pc)
    return pa

def inv_trans(T):
    rotation = np.linalg.inv(T[0:3, 0:3])  # rotation matrix

    translation = T[0:3, 3]
    translation = -1 * np.dot(rotation, translation.T)
    translation = np.reshape(translation, (3, 1))
    Q = np.hstack((rotation, translation))

    # # test if it is truly a roation matrix
    # d = np.linalg.det(rotation)
    # t = np.transpose(rotation)
    # o = np.dot(rotation, t)
    return Q

def quat_to_rotation(quat):
    m = np.sum(np.multiply(quat, quat))
    q = quat.copy()
    q = np.array(q)
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        rot_matrix = np.identity(4)
        return rot_matrix
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0]],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0]],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2]]],
        dtype=q.dtype)
    rot_matrix = np.transpose(rot_matrix)
    # # test if it is truly a rotation matrix
    # d = np.linalg.det(rotation)
    # t = np.transpose(rotation)
    # o = np.dot(rotation, t)
    return rot_matrix

def qaut_to_angle(quat):
    w=quat[0]
    x=quat[1]
    y=quat[2]
    z=quat[3]

    rol = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))#the rol is the yaw angle!
    pith = math.asin(2*(w*y-x*z))
    yaw = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))
    return rol


class Rescale:
    """Rescale the image in a sample to a given size.

    PARAMETERS
    ----------
    output_size: tuple or int
        Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, frame):
        matrix, rd_mask, ra_mask = frame['matrix'], frame['rd_mask'], frame['ra_mask']
        h, w = matrix.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # transform.resize induce a smoothing effect on the values
        # transform only the input data
        matrix = transform.resize(matrix, (matrix.shape[0], new_h, new_w))
        return {'matrix': matrix, 'rd_mask': rd_mask, 'ra_mask': ra_mask}


class Flip:
    """
    Randomly flip the matrix with a proba p
    """

    def __init__(self, proba):
        assert proba <= 1.
        self.proba = proba

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        h_flip_proba = np.random.uniform(0, 1)
        if h_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        v_flip_proba = np.random.uniform(0, 1)
        if v_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        return {'matrix': matrix, 'mask': mask}


class HFlip:
    """
    Randomly horizontal flip the matrix with a proba p
    """

    def __init__(self):
        pass

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        matrix = np.flip(matrix, axis=1).copy()
        if len(mask.shape) == 3:
            mask = np.flip(mask, axis=1).copy()
        elif len(mask.shape) == 4:
            mask = np.flip(mask, axis=2).copy()
        return {'matrix': matrix, 'mask': mask}


class VFlip:
    """
    Randomly vertical flip the matrix with a proba p
    """

    def __init__(self):
        pass

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        matrix = np.flip(matrix, axis=2).copy()
        if len(mask.shape) == 3:
            mask = np.flip(mask, axis=2).copy()
        elif len(mask.shape) == 4:
            mask = np.flip(mask, axis=3).copy()
        return {'matrix': matrix, 'mask': mask}
    

def get_transformations(transform_names, split='train', sizes=None):
    """Create a list of functions used for preprocessing
    @param transform_names: list of str, one for each transformation
    @param split: split currently used
    @param sizes: int or tuple (optional)
    @return: transformations to use for preprocessing (e.g. data augmentation)
    """
    transformations = list()
    if 'rescale' in transform_names:
        transformations.append(Rescale(sizes))
    if 'flip' in transform_names and split == 'train':
        transformations.append(Flip(0.5))
    if 'vflip' in transform_names and split == 'train':
        transformations.append(VFlip())
    if 'hflip' in transform_names and split == 'train':
        transformations.append(HFlip())
    return transformations


def normalize(data, signal_type, norm_type='local'):
    """
    Method to normalise the radar views
    @param data: radar view
    @param signal_type: signal to normalise ('range_doppler', 'range_angle' and 'angle_doppler')
    @param proj_path: path to the project to load weights
    @param norm_type: type of normalisation to apply ('local' or 'tvt')
    @return: normalised data
    """
    if norm_type in ('local'):
        min_value = np.min(data)
        max_value = np.max(data)
        norm_data = (data - min_value) / (max_value - min_value)
        return norm_data

    elif signal_type == 'range_doppler':
        if norm_type == 'tvt':
            rd_stats = {"mean": 58.47112418237332, "std": 3.748725977590863, "min_val": 37.59535773996415, "max_val": 119.08313902425246}
        min_value = float(rd_stats['min_val'])
        max_value = float(rd_stats['max_val'])

    elif signal_type == 'range_angle':
        if norm_type == 'tvt':
            ra_stats = {"mean": 56.00209075744544, "std": 5.761533706342774, "min_val": 40.40928894952408, "max_val": 103.80548746494114}
        min_value = float(ra_stats['min_val'])
        max_value = float(ra_stats['max_val'])

    elif signal_type == 'angle_doppler':
        if norm_type == 'tvt':
            ad_stats = {"mean": 59.62242439612116, "std": 4.212092368314555, "min_val": 54.42604354196056, "max_val": 105.79746676271202}
        min_value = float(ad_stats['min_val'])
        max_value = float(ad_stats['max_val'])

    else:
        raise TypeError('Signal {} is not supported.'.format(signal_type))

    norm_data = (data - min_value) / (max_value - min_value)
    return norm_data