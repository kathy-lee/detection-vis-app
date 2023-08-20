import rosbag
import os
import logging
import struct
import cv2
#import mkl_fft
import torch
import math
import numpy as np
import torchvision.transforms as transform
import pandas as pd

from pathlib import Path
from torchvision.transforms import Resize,CenterCrop
from torch.utils.data import Dataset
from PIL import Image
from mmwave import dsp
from mmwave.dsp.utils import Window

from detection_vis_backend.datasets.utils import read_radar_params, reshape_frame, gen_steering_vec, peak_search_full_variance
from detection_vis_backend.datasets.cfar import CA_CFAR


class DatasetFactory:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(DatasetFactory, cls).__new__(cls, *args, **kwargs)
            cls._instance.instance_dict = {}  # this dictionary stores the instances
        return cls._instance

    def get_instance(self, class_name, id):
        if id in self.instance_dict:
            # logging.error(f"#####################{class_name} instance created already, directly return {id}")
            return self.instance_dict[id]
        
        class_obj = globals()[class_name]()
        self.instance_dict[id] = class_obj
        # logging.error(f"#######################{class_name} instance not created yet, will create from file {id}")
        return class_obj

# class DatasetFactory:
#     _instances = {}
#     _singleton_instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._singleton_instance:
#             cls._singleton_instance = super().__new__(cls)
#         return cls._singleton_instance

#     def get_instance(self, class_name):
#         if class_name not in self._instances:
#             # Fetch the class from globals, create a singleton instance
#             cls = globals()[class_name]
#             self._instances[class_name] = cls()
#         return self._instances[class_name]


class RaDICaL(Dataset):
    name = ""
    features = []
    feature_path = ""
    config = ""

    image_count = 0
    depthimage_count = 0
    radarframe_count = 0
    frame_sync = 0


    def __init__(self, features=None):
        self.name = "RaDICaL dataset instance"
        self.features = ['image', 'depthimage', 'adc']


    def parse(self, file_id, file_path, file_name, config):
        self.config = config
        file = os.path.join(file_path, file_name)
        try:
            bag = rosbag.Bag(file)
        except rosbag.ROSBagException:
            print(f"No file found at {file}")
        topics_dict = bag.get_type_and_topic_info()[1]

        feature_path = Path(os.getenv('TMP_ROOTDIR')).joinpath(str(file_id))
        feature_path.mkdir(parents=True, exist_ok=True)
        self.feature_path = feature_path

        if "/camera/color/image_raw" in topics_dict:    
            (feature_path / "image").mkdir(parents=True, exist_ok=True)
            for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=['/camera/color/image_raw'])):
                # print(t.secs)
                # print(t.nsecs)
                # print(msg.header.stamp.secs)
                # print(msg.header.stamp.nsecs)
                # print(msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9)
                assert msg.encoding == "rgb8"
                dtype = np.dtype("uint8")  # 8-bit color image
                dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
                image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, 3)  # 3 for RGB
                # self.image.append(image)
                np.save(os.path.join(feature_path, "image", f"image_{idx}.npy"), image)
            self.image_count = idx + 1

        if "/camera/aligned_depth_to_color/image_raw" in topics_dict:
            (feature_path / "depth_image").mkdir(parents=True, exist_ok=True)
            for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw'])):
                # print(t.secs)
                # print(t.nsecs)
                # print(msg.header.stamp.secs)
                # print(msg.header.stamp.nsecs)
                # print(msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9)
                assert msg.encoding == "16UC1"
                dtype = np.dtype("uint16")  # 16-bit grayscale image
                dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
                image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
                # self.depth_image.append(image)
                np.save(os.path.join(feature_path, "depth_image", f"depth_image_{idx}.npy"), image)
            self.depthimage_count = idx + 1

        if "/radar_data" in topics_dict:
            (feature_path / "adc").mkdir(parents=True, exist_ok=True)
            for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=['/radar_data'])):
                #   print(t.secs)
                #   print(t.nsecs  print(t.secs + t.nsecs*1e-9)
                # if(count > 0):
                #   print(t.secs + t.nsecs*1e-9 - last)
                # last = t.secs + t.nsecs*1e-9
                # print("\nProcessing no.", count, "th radar msg:")
                
                arr = np.array(msg.data)
                complex_arr = reshape_frame(arr,304,4,2,64)
                adc = np.swapaxes(complex_arr, 1, 2)
                # self.ADC.append(transformed)
                np.save(os.path.join(feature_path, "adc", f"adc_{idx}.npy"), adc)
            self.radarframe_count = idx + 1
        
        bag.close()
        
        non_zero_lengths = [l for l in [self.image_count, self.depthimage_count, self.radarframe_count] if l != 0]
        if len(non_zero_lengths) == 0:
            self.frame_sync = 0
        elif len(non_zero_lengths) == 1:
            self.frame_sync = non_zero_lengths[0]
        else:
            if all(length == non_zero_lengths[0] for length in non_zero_lengths):
                self.frame_sync = non_zero_lengths[0]
            else:
                self.frame_sync = 0

        # parse radar config from config file
        self.radar_cfg = read_radar_params(self.config) 
        self.bins_processed = self.radar_cfg['profiles'][0]['adcSamples'] #radar_cube.shape[0]
        self.virt_ant = self.radar_cfg['numLanes'] * len(self.radar_cfg['chirps']) #radar_cube.shape[1]
        self.__doppler_bins = self.radar_cfg['numChirps'] // len(self.radar_cfg['chirps']) #radar_cube.shape[2]
        self.angle_res = 1
        self.angle_range = 90
        self.angle_bins = (self.angle_range * 2) // self.angle_res + 1
        self.num_vec, self.steering_vec = gen_steering_vec(self.angle_range, self.angle_res, self.virt_ant)


        self.numTxAntennas = 2
        self.numDopplerBins = 32
        self.numRangeBins = 304
        self.range_resolution, bandwidth = dsp.range_resolution(self.numRangeBins)
        self.doppler_resolution = dsp.doppler_resolution(bandwidth, start_freq_const=77, ramp_end_time=62, idle_time_const=100, 
                                                         num_loops_per_frame=32, num_tx_antennas=2)
        self.est_range=90 # est_range (int): The desired span of thetas for the angle spectrum. Used for gen_steering_vec
        self.est_resolution=1 # est_resolution (float): The desired angular resolution for gen_steering_vec
        return
        

    def get_RAD(self, idx=None):
        return None
    
    def get_ADC(self, idx=None):
        adc_file = os.path.join(self.feature_path,'adc',f"adc_{idx}.npy")
        adc = np.load(adc_file)
        return adc
    
    def get_RA(self, idx=None):
        adc = self.get_ADC(idx)
        # rf = RadarFrame(radar_config)
        # beamformed_range_azimuth = rf.compute_range_azimuth(adc) 
        range_cube = dsp.range_processing(adc, window_type_1d=Window.BLACKMAN)
        range_cube = np.swapaxes(range_cube, 0, 2)
        ra = np.zeros((self.bins_processed, self.angle_bins), dtype=complex)
        for i in range(self.bins_processed):
            ra[i,:], _ = dsp.aoa_capon(range_cube[i], self.steering_vec)
        np.flipud(np.fliplr(ra))
        ra = np.log(np.abs(ra))  
        return ra 

    def get_RD(self, idx=None):
        # rf = RadarFrame(radar_config)
        # rf.raw_cube = self.get_ADC(idx)
        # range_doppler = rf.range_doppler
        adc = self.get_ADC(idx)
        range_cube = dsp.range_processing(adc, window_type_1d=Window.BLACKMAN)
        range_doppler, _ = dsp.doppler_processing(range_cube, interleaved=False, num_tx_antennas=2, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)
        range_doppler = np.fft.fftshift(range_doppler, axes=1)
        range_doppler = np.transpose(range_doppler)
        range_doppler[np.isinf(range_doppler)] = 0  # replace Inf with zero
        # rd = np.concatenate([range_doppler.real,range_doppler.imag], axis=2)
        return range_doppler

    def get_radarpointcloud(self, idx=None):
        adc = self.get_ADC(idx)
        logging.error("#########################################")
        # 1. range fft
        radar_cube = dsp.range_processing(adc, window_type_1d=Window.BLACKMAN)
        # 2. doppler fft
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, interleaved=False, num_tx_antennas=2, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)
        # 3. 2D CFAR
        fft2d_sum = det_matrix.astype(np.int64)
        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_, axis=0, arr=fft2d_sum.T, l_bound=1.5, guard_len=4, noise_len=16)
        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_, axis=0, arr=fft2d_sum, l_bound=2.5, guard_len=4, noise_len=16)
        thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
        det_doppler_mask = (det_matrix > thresholdDoppler)
        det_range_mask = (det_matrix > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(self.numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()
        logging.error(f"detObj2DRaw:{detObj2DRaw.shape}")
        # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, self.numDopplerBins, reserve_neighbor=True)
        logging.error(f"detObj2DRaw:{detObj2DRaw.shape}")
        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, self.numDopplerBins)
        SNRThresholds2 = np.array([[2, 15], [10, 10], [35, 5]])
        peakValThresholds2 = np.array([[2, 50], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, 
                                           max_range=self.numRangeBins, min_range=0.5, range_resolution=self.range_resolution)
        logging.error(f"detObj2D:{detObj2D.shape}")
        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
        logging.error(f"azimuthInput:{azimuthInput.shape}")

        # 4. AoA
        num_vec, steering_vec = gen_steering_vec(self.est_range, self.est_resolution, 8)

        points = []
        method = 'Bartlett'
        for i, inputSignal in enumerate(azimuthInput):
            #logging.error(f"{i} loop ---")
            if method == 'Capon':
                doa_spectrum, _ = dsp.aoa_capon(np.reshape(inputSignal[:8], (8, 1)).T, steering_vec)
                doa_spectrum = np.abs(doa_spectrum)
            elif method == 'Bartlett':
                doa_spectrum = dsp.aoa_bartlett(steering_vec, np.reshape(inputSignal[:8], (8, 1)), axis=0)
                doa_spectrum = np.abs(doa_spectrum).squeeze()
            else:
                doa_spectrum = None

            # Find Max Values and Max Indices

            # num_out, max_theta, total_power = peak_search(doa_spectrum)
            obj_dict, total_power = peak_search_full_variance(doa_spectrum, steering_vec.shape[0], sidelobe_level=0.25)
            #logging.error(f"obj_dict:{obj_dict}")
            range = detObj2D['rangeIdx'][i] * self.range_resolution
            doppler = detObj2D['dopplerIdx'][i] * self.doppler_resolution
            for obj in obj_dict:
                azimuth = obj['peakLoc']/180 * np.pi
                # points.append([range, doppler, azimuth])
                points.append([range * np.sin(azimuth), range * np.cos(azimuth), doppler]) # [y, x, doppler]
                #logging.error(points)
        logging.error(f"Total points: {len(points)}, range resolution: {self.range_resolution}")   
        return np.array(points) 

    def get_lidarpointcloud():
        return None
    
    def get_spectrogram(self, idx=None):
        return None

    def get_image(self, idx=None):
        #return self.image[idx] if idx is not None else self.image
        image_file = os.path.join(self.feature_path,'image',f"image_{idx}.npy")
        image = np.load(image_file)
        return image
    
    def get_depthimage(self, idx=None):
        # return self.depth_image[idx] if idx is not None else self.depth_image
        image_file = os.path.join(self.feature_path,'depth_image',f"depth_image_{idx}.npy")
        image = np.load(image_file)
        return image

    def __len__(self):
        return self.frame_sync
    
    def __getitem__(self, idx):
        data_dict = {}
        for feature in self.features:
            data_dict[feature] = getattr(self, feature)[idx]
        return data_dict
    
    def set_features(self, features):
        self.features = features


class RADIal(Dataset):
    name = ""
    features = []
    config = ""

    image_count = 8252
    lidarframe_count = 8252
    radarframe_count = 8252
    frame_sync = 8252

    def __init__(self):
        # RADIal data has two formats: raw and ready-to-use
        self.name = "RADIal ready-to-use dataset instance" 
        self.features = ['image', 'lidarPC', 'adc', 'radarPC']
        

    def parse(self, file_id, file_path, file_name, config, difficult=True):
        def get_sorted_filenames(directory):
            # Get a sorted list of all file names in the given directory
            return sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])
        
        self.root_dir = os.path.join(file_path, file_name)
        self.image_filenames = get_sorted_filenames(os.path.join(self.root_dir, 'camera'))
        self.lidarpointcloud_filenames = get_sorted_filenames(os.path.join(self.root_dir, 'laser_PCL'))
        self.RD_filenames = get_sorted_filenames(os.path.join(self.root_dir, 'radar_FFT'))
        self.radarpointcloud_filenames = get_sorted_filenames(os.path.join(self.root_dir, 'radar_PCL'))

        self.labels = pd.read_csv(os.path.join(self.root_dir, 'labels.csv')).to_numpy()

        # Keeps only easy samples
        if(difficult==False):
            ids_filters=[]
            ids = np.where( self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]

        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.labels[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())

        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))

    def get_RAD(self, idx=None):
        return None
    
    def get_RA(self, idx=None):
        return None

    def get_RD(self, idx=None):
        rd = np.load(self.RD_filenames[idx])
        rd = np.concatenate([rd.real,rd.imag], axis=2)
        return rd

    def get_radarpointcloud(self, idx=None):
        # range,azimuth,elevation,power,doppler,x,y,z,v
        pc = np.load(self.radarpointcloud_filenames[idx], allow_pickle=True)[[5,6,7],:]   # Keeps only x,y,z
        pc = np.rollaxis(pc,1,0)
        pc[:,1] *= -1
        return pc

    def get_lidarpointcloud(self, idx=None):
        return np.load(self.lidarpointcloud_filenames[idx], allow_pickle=True)[:,:3]

    def get_image(self, idx=None): 
        image = np.asarray(Image.open(self.image_filenames[idx]))
        return image

    def get_depthimage(self, idx=None):
        return None
      
    def get_spectrogram(self, idx=None):
        return None

    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):
        
        # Get the sample id
        sample_id = self.sample_keys[index] 

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m
        # format as following [Range,Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix,y2_pix]
        box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 

        ######################
        #  Encode the labels #
        ######################
        out_label=[]
        # if(self.encoder!=None):
        out_label = self.encode(box_labels).copy()      

        # Read the Radar FFT data
        # radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npy".format(sample_id))
        # input = np.load(radar_name,allow_pickle=True)
        # radar_FFT = np.concatenate([input.real,input.imag],axis=2)
        radar_FFT = self.get_RD(index)
        if(self.statistics is not None):
            for i in range(len(self.statistics['input_mean'])):
                radar_FFT[...,i] -= self.statistics['input_mean'][i]
                radar_FFT[...,i] /= self.statistics['input_std'][i]

        # Read the segmentation map
        segmap_name = os.path.join(self.root_dir,'radar_Freespace',"freespace_{:06d}.png".format(sample_id))
        segmap = Image.open(segmap_name) # [512,900]
        # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
        # We crop the fov to 89.6deg
        segmap = self.crop(segmap)
        # and we resize to half of its size
        segmap = np.asarray(self.resize(segmap))==255

        # Read the camera image
        # img_name = os.path.join(self.root_dir,'camera',"image_{:06d}.jpg".format(sample_id))
        # image = np.asarray(Image.open(img_name))
        image = self.get_image(index)

        return radar_FFT, segmap,out_label,box_labels,image
    
    def set_features(self, features):
        self.features = features

    def encode(self,labels):
        geometry = {
            "ranges": [512,896,1],
            "resolution": [0.201171875,0.2],
            "size": 3
        }
        self.statistics = {
            "input_mean":[-2.6244e-03, -2.1335e-01,  1.8789e-02, -1.4427e+00, -3.7618e-01,
                1.3594e+00, -2.2987e-01,  1.2244e-01,  1.7359e+00, -6.5345e-01,
                3.7976e-01,  5.5521e+00,  7.7462e-01, -1.5589e+00, -7.2473e-01,
                1.5182e+00, -3.7189e-01, -8.8332e-02, -1.6194e-01,  1.0984e+00,
                9.9929e-01, -1.0495e+00,  1.9972e+00,  9.2869e-01,  1.8991e+00,
               -2.3772e-01,  2.0000e+00,  7.7737e-01,  1.3239e+00,  1.1817e+00,
               -6.9696e-01,  4.4288e-01],
            "input_std":[20775.3809, 23085.5000, 23017.6387, 14548.6357, 32133.5547, 28838.8047,
                27195.8945, 33103.7148, 32181.5273, 35022.1797, 31259.1895, 36684.6133,
                33552.9258, 25958.7539, 29532.6230, 32646.8984, 20728.3320, 23160.8828,
                23069.0449, 14915.9053, 32149.6172, 28958.5840, 27210.8652, 33005.6602,
                31905.9336, 35124.9180, 31258.4316, 31086.0273, 33628.5352, 25950.2363,
                29445.2598, 32885.7422],
            "reg_mean":[0.4048094369863972,0.3997392847799934],
            "reg_std":[0.6968599580482511,0.6942950877813826]
        }
        regression_layer = 2

        INPUT_DIM = (geometry['ranges'][0],geometry['ranges'][1],geometry['ranges'][2])
        OUTPUT_DIM = (regression_layer + 1, INPUT_DIM[0] // 4 , INPUT_DIM[1] // 4 )

        map = np.zeros(OUTPUT_DIM )

        for lab in labels:
            # [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]

            if(lab[0]==-1):
                continue

            range_bin = int(np.clip(lab[0]/geometry['resolution'][0]/4,0,OUTPUT_DIM[1]))
            range_mod = lab[0] - range_bin*geometry['resolution'][0]*4

            # ANgle and deg
            angle_bin = int(np.clip(np.floor(lab[1]/geometry['resolution'][1]/4 + OUTPUT_DIM[2]/2),0,OUTPUT_DIM[2]))
            angle_mod = lab[1] - (angle_bin- OUTPUT_DIM[2]/2)*geometry['resolution'][1]*4 

            if(geometry['size']==1):
                map[0,range_bin,angle_bin] = 1
                map[1,range_bin,angle_bin] = (range_mod - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                map[2,range_bin,angle_bin] = (angle_mod - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
            else:

                s = int((geometry['size']-1)/2)
                r_lin = np.linspace(geometry['resolution'][0]*s, -geometry['resolution'][0]*s,
                                    geometry['size'])*4
                a_lin = np.linspace(geometry['resolution'][1]*s, -geometry['resolution'][1]*s,
                                    geometry['size'])*4
                
                px_a, px_r = np.meshgrid(a_lin, r_lin)

                if(angle_bin>=s and angle_bin<(OUTPUT_DIM[2]-s)):
                    map[0,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = 1
                    map[1,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = ((px_r+range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2,range_bin-s:range_bin+(s+1),angle_bin-s:angle_bin+(s+1)] = ((px_a + angle_mod) - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1] 
                elif(angle_bin<s):
                    map[0,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = 1
                    map[1,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = ((px_r[:,s-angle_bin:] + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2,range_bin-s:range_bin+(s+1),0:angle_bin+(s+1)] = ((px_a[:,s-angle_bin:] + angle_mod)- self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]
                    
                elif(angle_bin>=OUTPUT_DIM[2]):
                    end = s+(OUTPUT_DIM[2]-angle_bin)
                    map[0,range_bin-s:range_bin+(s+1),angle_bin-s:] = 1
                    map[1,range_bin-s:range_bin+(s+1),angle_bin-s:] = ((px_r[:,:end] + range_mod) - self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]
                    map[2,range_bin-s:range_bin+(s+1),angle_bin-s:] = ((px_a[:,:end] + angle_mod)- self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]

        return map
    


class RADIalRaw(Dataset):
    name = ""
    features = []
    feature_path = ""
    config = ""

    image_count = 0
    depthimage_count = 0
    radarframe_count = 0
    frame_sync = 0

    def __init__(self, features=None):
        self.name = "RADIal raw dataset instance"
        self.features = ['image', 'lidarPC', 'adc']

        # Radar parameters
        self.numSamplePerChirp = 512
        self.numRxPerChip = 4
        self.numChirps = 256
        self.numRxAnt = 16
        self.numTxAnt = 12
        self.numReducedDoppler = 16
        self.numChirpsPerLoop = 16
    
    def parse_recording(folder):
        recorder_folder_path = Path(folder)
        
        list_files = [file for file in os.listdir(recorder_folder_path) if os.isfile(os.path.join(recorder_folder_path, file))]
        
        # list the number of sensor recorded
        list_of_sensors = []
        list_files2 = []
        offsetTable = {'camera':0,'scala':-40000,'radar_ch0':-180000,'radar_ch1':-180000,'radar_ch2':-180000,'radar_ch3':-180000,'gps':0,'can':0}
        for file in list_files:
            sensor = file[len(recorder_folder_path.name)+1:]
            sensor = sensor[:sensor.rfind('.')]

            if(sensor in list(offsetTable.keys())):
                list_of_sensors.append(sensor)
                list_files2.append(file)

        list_files = list_files2
        dict_sensor = {s:{'filename':recorder_folder_path/list_files[i],'timestamp': [],'timeofissue': [],'sample': [], 'offset': [], 'datasize' :[]} for i,s in enumerate(list_of_sensors)}

        # 1. Open the REC file
        rec_file_name = recorder_folder_path.name+'_events_log.rec'
        rec_file_path = recorder_folder_path/rec_file_name
        
        df = pd.read_csv(rec_file_path,header=None)
        data = df.values
        
        for line in data:
            elt = line[0].split()
            timestamp = int(elt[1])
            timeofissue = int(elt[4])
            sample = int(elt[7])
            sensor = elt[10]

            if(sensor not in dict_sensor.keys()):
                continue
            
            dict_sensor[sensor]['timestamp'].append(timestamp)
            dict_sensor[sensor]['timeofissue'].append(timeofissue)
            dict_sensor[sensor]['sample'].append(sample)
            
            if(len(elt)==17):
                # timestamp - sample - sensor - offset - datasize
                dict_sensor[sensor]['offset'].append(int(elt[13]))
                dict_sensor[sensor]['datasize'].append(int(elt[16]))
                
        for sensor in dict_sensor:
            if(len(dict_sensor[sensor]['timeofissue'])>0 and len(dict_sensor[sensor]['timestamp'])>0):

                offset = dict_sensor[sensor]['timeofissue'][0] - dict_sensor[sensor]['timestamp'][0] + offsetTable[sensor]
                for i in range(len(dict_sensor[sensor]['timestamp'])):
                    dict_sensor[sensor]['timestamp'][i] += offset

            dict_sensor[sensor]['timestamp'] = np.asarray(dict_sensor[sensor]['timestamp'])
            dict_sensor[sensor]['timeofissue'] = np.asarray(dict_sensor[sensor]['timeofissue'])
            dict_sensor[sensor]['sample'] = np.asarray(dict_sensor[sensor]['sample'])
            dict_sensor[sensor]['offset'] = np.asarray(dict_sensor[sensor]['offset'])
            dict_sensor[sensor]['datasize'] = np.asarray(dict_sensor[sensor]['datasize'])

        return dict_sensor


    def parse(self, file_id, file_path, file_name, config, master=None, tolerance=200000, sync_mode='timestamp', silent=False): 
        # 1. Open the REC file
        self.rec_file_name = file_name +'_events_log.rec'
        self.rec_file_path = os.path.join(file_path, file_name, self.rec_file_name)
        
        labls = np.array([str(i) for i in range(22)]) # create some row names
        df = pd.read_csv(self.rec_file_path,header=None,names=labls,sep='[-|\s+]',engine='python')
        nbColumn = df.shape[1]
        self.df = df.iloc[:, np.arange(1,nbColumn,4)]
        self.df.columns = ['timestamp', 'timeofissue','data_sample', 'sensor', 'offset','datasize']
        self.sensorsFilters = self.df['sensor'].unique()
        self.filters = []
        self.df_filtered = self.df

        id_to_del = []
        nb_corrupted = 0
        nb_tolerance = 0

        self.dicts = self.parse_recording(os.path.join(file_path, file_name).name)
        # for Each radar sample, find the clostest sample for each sensor
        if(master is None):
            # by default, we use the Radar as Matser sensor
            if('radar_ch0' not in self.dicts or 'radar_ch1' not in self.dicts 
               or 'radar_ch2' not in self.dicts or 'radar_ch3' not in self.dicts):
                print('Error: recording does not contains the 4 radar chips')
            
            keys =list(self.dicts.keys())
            self.keys = keys
            
            if('gps' in self.dicts):
                keys.remove('gps')
            if('preview' in self.dicts):
                keys.remove('preview')
            if('None' in self.dicts):
                keys.remove('None')
            keys.remove('radar_ch0')
            keys.remove('radar_ch1')
            keys.remove('radar_ch2')
            keys.remove('radar_ch3')
            
            self.table=[]
            
            # Check the length of all radar recordings!
            NbSample = len(self.dicts['radar_ch3']['timestamp'])
            print(f"NbSample: ")
            # Sequence is radar_ch3 radar_ch0 radar_ch2 radar_ch1
            for i in range(NbSample):
                timestamp = self.dicts['radar_ch3']['timestamp'][i]
                timeofissue = self.dicts['radar_ch3']['timeofissue'][i]
                FrameNumber = self.dicts['radar_ch3']['sample'][i]

                idx0 = np.where(self.dicts['radar_ch0']['sample']==(FrameNumber+1))[0]
                idx2 = np.where(self.dicts['radar_ch2']['sample']==(FrameNumber+2))[0]
                idx1 = np.where(self.dicts['radar_ch1']['sample']==(FrameNumber+3))[0]
                match={}

                match['radar_ch3'] = i

                if(len(idx0)==0 or len(idx1)==0 or len(idx2)==0):
                    id_to_del.append(i)
                    nb_corrupted+=1
                    match['radar_ch0'] = -1
                    match['radar_ch1'] = -1
                    match['radar_ch2'] = -1
                else:
                    match['radar_ch0'] = idx0[0]
                    match['radar_ch1'] = idx1[0]
                    match['radar_ch2'] = idx2[0]

                
                if(self.sync_mode=='timestamp'):
                    for k in keys:
                        if(len(self.dicts[k]['timestamp'])>0):
                            time_diff = np.abs(np.asarray(self.dicts[k]['timestamp']) - timestamp)
                            vmin = time_diff.min()
                            index_min = time_diff.argmin()

                            if(vmin>tolerance):
                                index_min=-1
                        else:
                            index_min=-1
                        
                        if(index_min==-1):
                            nb_tolerance+=1
                            id_to_del.append(i)

                        match[k] = index_min
                else:
                    for k in keys:
                        if(len(self.dicts[k]['timeofissue'])>0):
                            time_diff = np.abs(np.asarray(self.dicts[k]['timeofissue']) - timeofissue)
                            vmin = time_diff.min()
                            index_min = time_diff.argmin()
                            if(vmin>tolerance):
                                index_min=-1
                        else:
                            index_min=-1
                        
                        if(index_min==-1):
                            nb_tolerance+=1
                            id_to_del.append(i)

                        match[k] = index_min
            
                self.table.append(match)

            self.table = np.asarray(self.table)

            # Keep only sync samples
            id_to_del = np.unique(np.asarray(id_to_del))
            id_total = np.arange(len(self.table))
            self.id_valid = np.setdiff1d(id_total, id_to_del)
            if(not self.silent):
            	print('Total tolerance errors: ',nb_tolerance/len(self.table)*100,'%')
            	print('Total corrupted frames: ',nb_corrupted/len(self.table)*100,'%')
            self.table = self.table[self.id_valid]


        elif(master=='camera'):
            # we discard the radar, and consider only camera, laser, can
            if('camera' not in self.dicts):
                print('Error: recording does not contains camera')
            
            keys =list(self.dicts.keys())

            if('gps' in self.dicts):
                keys.remove('gps')
            if('radar_ch0' in self.dicts):
                keys.remove('radar_ch0')                
            if('radar_ch1' in self.dicts):
                keys.remove('radar_ch1')  
            if('radar_ch2' in self.dicts):
                keys.remove('radar_ch2')  
            if('radar_ch3' in self.dicts):
                keys.remove('radar_ch3')  
            if('preview' in self.dicts):
                keys.remove('preview')
            if('None' in self.dicts):
                keys.remove('None')

            self.keys = keys

            self.table=[]
            for i in range(len(self.dicts['camera']['timestamp'])):

                timestamp = self.dicts['camera']['timestamp'][i]  
                timeofissue = self.dicts['camera']['timeofissue'][i]

                match={}
                match['camera'] = i
                
                if(self.sync_mode=='timestamp'):
                    for k in keys:
                        if(len(self.dicts[k]['timestamp'])>0):
                            time_diff = np.abs(np.asarray(self.dicts[k]['timestamp']) - timestamp)
                            vmin = time_diff.min()
                            index_min = time_diff.argmin()
                        
                            if(vmin>tolerance):
                                nb_tolerance+=1
                                index_min = -1
                                id_to_del.append(i)
                        else:
                            index_min = -1
                        
                        match[k] = index_min
                else:
                    for k in keys:
                        if(len(self.dicts[k]['timeofissue'])>0):
                            time_diff = np.abs(np.asarray(self.dicts[k]['timeofissue']) - timeofissue)
                            vmin = time_diff.min()
                            index_min = time_diff.argmin()
                        
                            if(vmin>tolerance):
                                nb_tolerance+=1
                                id_to_del.append(i)
                                index_min = -1
                        else:
                            index_min = -1
                        
                        match[k] = index_min
            
                self.table.append(match)

            self.table = np.asarray(self.table)
            # Keep only sync samples
            id_to_del = np.unique(np.asarray(id_to_del))
            id_total = np.arange(len(self.table))
            id_to_keep = np.setdiff1d(id_total, id_to_del)
            if(not self.silent):
            	print('Total tolerance errors: ',nb_tolerance/len(self.table)*100,'%')
            self.table = self.table[id_to_keep]

        else:
            print('Mode not supported')
            return


        self.can_frames={'timestamp':[],'ID':[],'data':[]}
        if('can' in self.dicts):
            A = []
            for i in range(len(self.readers['can'].dict['offset'])):
                A.append(self.readers['can'].GetData(i))
            
            A=np.concatenate(A)
            
            for i in range(len(A)):
                self.can_frames['timestamp'].append(A[i]['timestamp'])
                self.can_frames['ID'].append(A[i]['ID'])
                self.can_frames['data'].append(A[i]['DATA'])
            self.can_frames['ID'] = np.asarray(self.can_frames['ID'])
            self.can_frames['timestamp'] = np.asarray(self.can_frames['timestamp'])

        # parse into feature_path
        feature_path = Path(os.getenv('TMP_ROOTDIR')).joinpath(str(file_id))
        feature_path.mkdir(parents=True, exist_ok=True)
        self.feature_path = feature_path

        (feature_path / "image").mkdir(parents=True, exist_ok=True)
        (feature_path / "adc").mkdir(parents=True, exist_ok=True)
        (feature_path / "lidarPC").mkdir(parents=True, exist_ok=True)
        if(np.shape(self.dicts['camera']['offset'])[0]>0):
            f = open(str(self.dicts['camera']['filename']),'rb')
        else:
            f = cv2.VideoCapture(str(self.dicts['camera']['filename']))
        fd = open(str(self.dicts['scala']['filename']),'rb')
        fd_ch0 = open(str(self.dicts['radar_ch0']['filename']),'rb')
        fd_ch1 = open(str(self.dicts['radar_ch1']['filename']),'rb')
        fd_ch2 = open(str(self.dicts['radar_ch2']['filename']),'rb')
        fd_ch3 = open(str(self.dicts['radar_ch3']['filename']),'rb')
        struct_fmt = '=7f4B'
        struct_len = struct.calcsize(struct_fmt)
        struct_unpack = struct.Struct(struct_fmt).unpack_from
        for idx in self.table:
            # MJPG mode
            if(np.shape(self.dicts['camera']['offset'])[0]>0):
                offset = int(self.dicts['camera']['offset'][idx])
                length = int(self.dicts['camera']['datasize'][idx])
                f.seek(offset)
                data = f.read(length)
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            # AVI MOde
            else:
                f.set(cv2.CAP_PROP_POS_FRAMES,idx)
                ret, image = f.read()
            cv2.imwrite(os.path.join(feature_path, "image", f"image_{idx}.jpg"), image)

            # lidar pc
            offset = self.dicts['scala']['offset'][idx]
            datasize = self.dicts['scala']['datasize'][idx]
            fd.seek(offset)
            pts3d=[]
            for i in range(int(datasize/struct_len)):
                pts3d.append(struct_unpack(fd.read(struct_len)))
            pts3d = np.asarray(pts3d)
            np.save(os.path.join(feature_path, "lidarpc", f"lidarpc_{idx}.npy"), pts3d)

            # radar adc
            offset = int(self.dicts['radar_ch0']['offset'][idx])
            datasize = self.dicts['radar_ch0']['datasize'][idx]
            fd_ch0.seek(offset)
            adc0 = np.fromfile(fd_ch0, dtype=np.int16, count=int(datasize/2))

            offset = int(self.dicts['radar_ch1']['offset'][idx])
            datasize = self.dicts['radar_ch1']['datasize'][idx]
            fd_ch1.seek(offset)
            adc1 = np.fromfile(fd_ch1, dtype=np.int16, count=int(datasize/2))

            offset = int(self.dicts['radar_ch2']['offset'][idx])
            datasize = self.dicts['radar_ch2']['datasize'][idx]
            fd_ch2.seek(offset)
            adc2 = np.fromfile(fd_ch2, dtype=np.int16, count=int(datasize/2))

            offset = int(self.dicts['radar_ch3']['offset'][idx])
            datasize = self.dicts['radar_ch3']['datasize'][idx]
            fd_ch3.seek(offset)
            adc3 = np.fromfile(fd_ch3, dtype=np.int16, count=int(datasize/2))

            frame0 = np.reshape(adc0[0::2] + 1j*adc0[1::2], (self.numSamplePerChirp,self.numRxPerChip, self.numChirps), order ='F').transpose((0,2,1))   
            frame1 = np.reshape(adc1[0::2] + 1j*adc1[1::2], (self.numSamplePerChirp,self.numRxPerChip, self.numChirps), order ='F').transpose((0,2,1))   
            frame2 = np.reshape(adc2[0::2] + 1j*adc2[1::2], (self.numSamplePerChirp,self.numRxPerChip, self.numChirps), order ='F').transpose((0,2,1))   
            frame3 = np.reshape(adc3[0::2] + 1j*adc3[1::2], (self.numSamplePerChirp,self.numRxPerChip, self.numChirps), order ='F').transpose((0,2,1))   
            adc = np.concatenate([frame3,frame0,frame1,frame2],axis=2)
            np.save(os.path.join(feature_path, "adc", f"adc_{idx}.npy"), adc)

        self.AoA_mat = np.load(os.path.join(file_path, "CalibrationTable.npy"),allow_pickle=True).item()

        if(self.method == 'PC'):
            self.CFAR_fct = CA_CFAR(win_param=(9,9,3,3), threshold=2, rd_size=(self.numSamplePerChirp,16))
            self.CalibMat = np.rollaxis(self.AoA_mat['Signal'],2,1).reshape(self.AoA_mat['Signal'].shape[0]*self.AoA_mat['Signal'].shape[2],self.AoA_mat['Signal'].shape[1])
        else:
            # For RA map estimation, we consider only one elevation, the one parallel to the road plan (index=5)
            self.CalibMat=self.AoA_mat['Signal'][...,5]
        
        if(self.device =='cuda'):
            # if(self.lib=='CuPy'):
            #     print('CuPy on GPU will be used to execute the processing')
            #     cp.cuda.Device(0).use()
            #     self.CalibMat = cp.array(self.CalibMat,dtype='complex64')
            #     self.window = cp.array(self.AoA_mat['H'][0])
            # else:
            print('PyTorch on GPU will be used to execute the processing')
            self.CalibMat = torch.from_numpy(self.CalibMat).to('cuda')
            self.window = torch.from_numpy(self.AoA_mat['H'][0]).to('cuda')   
        else:
            print('CPU will be used to execute the processing')
            self.window = self.AoA_mat['H'][0]
            
        # Build hamming window table to reduce side lobs
        hanningWindowRange = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numSamplePerChirp ))/(self.numSamplePerChirp -1))))
        hanningWindowDoppler = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(self.numChirps ))/(self.numChirps -1))))
        self.range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange,1), repeats=self.numChirps, axis=1),2)
        self.doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=self.numSamplePerChirp, axis=0),2)
    
        ## indexes shift to find Tx spots
        self.dividend_constant_arr = np.arange(0, self.numReducedDoppler*self.numChirpsPerLoop ,self.numReducedDoppler)


    def __len__(self):
        return len(self.table)

    def set_features(self, features):
        self.features = features

    def __getitem__(self, index):   
        frame = () 
        for f in self.features:
            if f == 'image':
                file = os.path.join(self.feature_path,"image",f"image_{index}.jpg")
                data = np.asarray(Image.open(file))
            elif f == 'lidarPC':
                file = os.path.join(self.feature_path,"lidarpc",f"lidarpc_{index}.npy")
                data = np.load(file)
            elif f == 'adc':
                file = os.path.join(self.feature_path,"adc",f"adc_{index}.npy")
                data = np.load(file)
            frame = frame + (data,)
        return data
    
    def get_RD(self, idx=None):
        file = os.path.join(self.feature_path,"adc",f"adc_{idx}.npy")
        complex_adc = np.load(file)
        # 2- Remoce DC offset
        complex_adc = complex_adc - np.mean(complex_adc, axis=(0,1))

        # 3- Range FFTs
        range_fft = np.fft(np.multiply(complex_adc,self.range_fft_coef),self.numSamplePerChirp,axis=0) # mkl
    
        # 4- Doppler FFts
        RD_spectrums = np.fft(np.multiply(range_fft,self.doppler_fft_coef),self.numChirps,axis=1) # mkl
        return RD_spectrums
    
    def get_RA(self, idx=None):
        file = os.path.join(self.feature_path,"adc",f"adc_{idx}.npy")
        complex_adc = np.load(file)
        # 2- Remoce DC offset
        complex_adc = complex_adc - np.mean(complex_adc, axis=(0,1))

        # 3- Range FFTs
        range_fft = np.fft(np.multiply(complex_adc,self.range_fft_coef),self.numSamplePerChirp,axis=0) # mkl
    
        # 4- Doppler FFts
        RD_spectrums = np.fft(np.multiply(range_fft,self.doppler_fft_coef),self.numChirps,axis=1) # mkl

        doppler_indexes = []
        for doppler_bin in range(self.numChirps):
            DopplerBinSeq = np.remainder(doppler_bin+ self.dividend_constant_arr, self.numChirps)
            DopplerBinSeq = np.concatenate([[DopplerBinSeq[0]],DopplerBinSeq[5:]])
            doppler_indexes.append(DopplerBinSeq)
        
        MIMO_Spectrum = RD_spectrums[:,doppler_indexes,:].reshape(RD_spectrums.shape[0]*RD_spectrums.shape[1],-1)

        if(self.device=='cpu'):
            # Multiply with Hamming window to reduce side lobes
            MIMO_Spectrum = np.multiply(MIMO_Spectrum,self.window)

            Azimuth_spec = np.abs(self.CalibMat@MIMO_Spectrum.transpose())
            Azimuth_spec = Azimuth_spec.reshape(self.AoA_mat['Signal'].shape[0],RD_spectrums.shape[0],RD_spectrums.shape[1])

            RA_map = np.sum(np.abs(Azimuth_spec),axis=2)          
            return RA_map.transpose()
        else:      
            # if(self.lib=='CuPy'):
            #     MIMO_Spectrum = cp.array(MIMO_Spectrum)
            #     # Multiply with Hamming window to reduce side lobes
            #     MIMO_Spectrum = cp.multiply(MIMO_Spectrum,self.window).transpose()
            #     Azimuth_spec = cp.abs(cp.dot(self.CalibMat,MIMO_Spectrum))
            #     Azimuth_spec = Azimuth_spec.reshape(self.AoA_mat['Signal'].shape[0],RD_spectrums.shape[0],RD_spectrums.shape[1])
            #     RA_map = np.sum(np.abs(Azimuth_spec),axis=2)
            #     return RA_map.transpose().get()
            # else:
            MIMO_Spectrum = torch.from_numpy(MIMO_Spectrum).to('cuda')
            # Multiply with Hamming window to reduce side lobes
            MIMO_Spectrum = torch.transpose(torch.multiply(MIMO_Spectrum,self.window),1,0).cfloat()
            Azimuth_spec = torch.abs(torch.matmul(self.CalibMat,MIMO_Spectrum))
            Azimuth_spec = Azimuth_spec.reshape(self.AoA_mat['Signal'].shape[0],RD_spectrums.shape[0],RD_spectrums.shape[1])
            RA_map = torch.sum(torch.abs(Azimuth_spec),axis=2)
            return RA_map.detach().cpu().numpy().transpose()


    def __find_TX0_position(self,power_spectrum,range_bins,reduced_doppler_bins):        
        doppler_idx = np.tile(reduced_doppler_bins,(self.numReducedDoppler,1)).transpose()+np.repeat(np.expand_dims(np.arange(0,self.numChirps,self.numReducedDoppler),0),len(range_bins),axis=0)
        doppler_idx = np.concatenate([doppler_idx,doppler_idx[:,:4]],axis=1)
        range_bins = [[r] for r in range_bins]
        cumsum = np.cumsum(power_spectrum[range_bins,doppler_idx],axis=1) 
        N = 4
        mat = (cumsum[:,N:] - cumsum[:,:-N]) / N
        section_idx = np.argmin(mat,axis=1)
        doppler_bins = section_idx*self.numReducedDoppler+reduced_doppler_bins
        return doppler_bins
    

    def get_radarpointcloud(self, idx=None):
        RD_spectrums = self.get_RD(idx)

        # 1- Compute power spectrum
        power_spectrum = np.sum(np.abs(RD_spectrums),axis=2)

        # 2- Apply CFAR
        # But because Tx are phase shifted of DopplerShift=16, then reduce spectrum to MaxDoppler/16 on Doppler axis
        reduced_power_spectrum = np.sum(power_spectrum.reshape(512,16,16),axis=1)
        peaks = self.CFAR_fct(reduced_power_spectrum)
        RangeBin,DopplerBin_conv = np.where(peaks>0)

        # 3- Need to find TX0 position to rebuild the MIMO spectrum in the correct order
        DopplerBin_candidates = self.__find_TX0_position(power_spectrum, RangeBin, DopplerBin_conv)
        RangeBin_candidates = [[i] for i in RangeBin]
        doppler_indexes = []
        for doppler_bin in DopplerBin_candidates:
            DopplerBinSeq = np.remainder(doppler_bin+ self.dividend_constant_arr, self.numChirps)
            DopplerBinSeq = np.concatenate([[DopplerBinSeq[0]],DopplerBinSeq[5:]]).astype('int')
            doppler_indexes.append(DopplerBinSeq)
            

        # 4- Extract and reshape the Rx * Tx matrix into the MIMO spectrum
        MIMO_Spectrum = RD_spectrums[RangeBin_candidates,doppler_indexes,:].reshape(len(DopplerBin_candidates),-1)
        MIMO_Spectrum = np.multiply(MIMO_Spectrum,self.window)
        
        # 5- AoA: maker a cross correlation between the recieved signal vs. the calibration matrix 
        # to identify azimuth and elevation angles
        ASpec=np.abs(self.CalibMat@MIMO_Spectrum.transpose())
        
        # 6- Extract maximum per (Range,Doppler) bins
        x,y = np.where(np.isnan(ASpec))
        ASpec[x,y] = 0
        az,el = np.unravel_index(np.argmax(ASpec,axis=0),(self.AoA_mat['Signal'].shape[0],self.AoA_mat['Signal'].shape[2]))
        az = np.deg2rad(self.AoA_mat['Azimuth_table'][az])
        el = np.deg2rad(self.AoA_mat['Elevation_table'][el])
        
        RangeBin = RangeBin/self.numSamplePerChirp*103.
        
        return np.vstack([RangeBin,DopplerBin_candidates,az,el]).transpose()

    def get_lidarpointcloud(self, idx=None):
        file = os.path.join(self.feature_path,"lidarpc",f"lidarpc_{idx}.npy")
        pc = np.load(file)
        return pc

    def get_image(self, idx=None): 
        file = os.path.join(self.feature_path,"image",f"image_{idx}.jpg")
        image = np.asarray(Image.open(file))
        return image

    def get_depthimage(self, idx=None):
        return None
      
    def get_spectrogram(self, idx=None):
        return None

