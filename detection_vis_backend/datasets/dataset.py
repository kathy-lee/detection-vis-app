import rosbag
import os
import logging
import struct
import cv2
import imageio
import io
#import mkl_fft
import torch
import math
import pickle
import json
import random
import time
import numpy as np
import scipy.io as spio
import torchvision.transforms as transform
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from torchvision.transforms import Resize,CenterCrop
from torch.utils.data import Dataset
from PIL import Image
from mmwave import dsp
from mmwave.dsp.utils import Window
from loguru import logger


from detection_vis_backend.datasets.utils import read_radar_params, reshape_frame, gen_steering_vec, peak_search_full_variance, generate_confmaps, load_anno_txt, read_pointcloudfile, inv_trans, quat_to_rotation, get_transformations, VFlip, HFlip, normalize, complexTo2Channels, smoothOnehot, iou3d, flip_vertical, flip_horizontal, confmap2ra, find_nearest, generate_confmap, normalize_confmap, add_noise_channel, get_co_vec, bi_var_gauss, plain_gauss, get_center_map, get_orent_map
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
    name = "RaDICaL dataset instance"    

    def __init__(self, features=None):
        self.feature_path = ""
        self.config = ""

        self.frames_count = {} 
        self.timestamps = {} # dict of list
        self.sync_indices = [] # list of dict
        self.sync_mode = False
        self.frame_sync = 0
        self.features = ['image', 'depth_image', 'adc']


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
            self.timestamps['image'] = []
            for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=['/camera/color/image_raw'])):
                # print(t.secs)
                # print(t.nsecs)
                # print(msg.header.stamp.secs)
                # print(msg.header.stamp.nsecs)
                # print(msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9)
                self.timestamps['image'].append(t.secs + t.nsecs*1e-9)
                assert msg.encoding == "rgb8"
                dtype = np.dtype("uint8")  # 8-bit color image
                dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
                image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, 3)  # 3 for RGB
                # self.image.append(image)
                np.save(os.path.join(feature_path, "image", f"image_{idx}.npy"), image)
            self.frames_count['image'] = idx + 1

        if "/camera/aligned_depth_to_color/image_raw" in topics_dict:
            (feature_path / "depth_image").mkdir(parents=True, exist_ok=True)
            self.timestamps['depth_image'] = []
            for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw'])):
                # print(t.secs)
                # print(t.nsecs)
                # print(msg.header.stamp.secs)
                # print(msg.header.stamp.nsecs)
                # print(msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9)
                self.timestamps['depth_image'].append(t.secs + t.nsecs*1e-9)
                assert msg.encoding == "16UC1"
                dtype = np.dtype("uint16")  # 16-bit grayscale image
                dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
                image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
                # self.depth_image.append(image)
                np.save(os.path.join(feature_path, "depth_image", f"depth_image_{idx}.npy"), image)
            self.frames_count['depth_image'] = idx + 1

        if "/radar_data" in topics_dict:
            (feature_path / "adc").mkdir(parents=True, exist_ok=True)
            self.timestamps['adc'] = []
            for idx, (topic, msg, t) in enumerate(bag.read_messages(topics=['/radar_data'])):
                self.timestamps['adc'].append(t.secs + t.nsecs*1e-9)
                arr = np.array(msg.data)
                complex_arr = reshape_frame(arr,304,4,2,64)
                adc = np.swapaxes(complex_arr, 1, 2)
                # self.ADC.append(transformed)
                np.save(os.path.join(feature_path, "adc", f"adc_{idx}.npy"), adc)
            self.frames_count['adc'] = idx + 1
            # Other radar features have the same frames as ADC
            self.frames_count['RD'] = self.frames_count['adc']
            self.frames_count['RA'] = self.frames_count['adc']
            self.frames_count['RAD'] = self.frames_count['adc']
            self.frames_count['radarPC'] = self.frames_count['adc']
            self.frames_count['spectrogram'] = self.frames_count['adc']

        
        bag.close()
        
        non_zero_lengths = [l for l in self.frames_count.values() if l != 0]
        if len(non_zero_lengths) == 0:
            self.frame_sync = 0
        elif len(non_zero_lengths) == 1:
            self.frame_sync = non_zero_lengths[0]
        else:
            if all(length == non_zero_lengths[0] for length in non_zero_lengths):
                self.frame_sync = non_zero_lengths[0]
            else:
                self.frame_sync = 0
                # get current feature set and create sync indices
                #feature_timestamps = [self.timestamps[f] for f in self.features]
                feature_timestamps = {k: self.timestamps[k] for k in self.features if k in self.timestamps}

                feature_frames_count = [self.frames_count[f] for f in self.features]
                reference_feature = self.features[feature_frames_count.index(min(feature_frames_count))]
                for i,ti in enumerate(self.timestamps[reference_feature]):
                    index = {}
                    for f in self.features:
                        if f == reference_feature:
                            index[f] = i
                        else:
                            time_diff = [abs(ti-t) for t in feature_timestamps[f]]
                            idx_closest = time_diff.index(min(time_diff))
                            index[f] = idx_closest
                    self.sync_indices.append(index)
        logging.error(f"sync incdices = {len(self.sync_indices)}")
        # parse radar config from config file
        self.radar_cfg = read_radar_params(self.config) 
        self.numRangeBins = self.radar_cfg['profiles'][0]['adcSamples'] #radar_cube.shape[0]
        self.virt_ant = self.radar_cfg['numLanes'] * len(self.radar_cfg['chirps']) #radar_cube.shape[1]
        self.numDopplerBins = self.radar_cfg['numChirps'] // len(self.radar_cfg['chirps']) #radar_cube.shape[2]

        # self.numTxAntennas = 2, self.numRxAntennas = 4
        self.range_resolution, bandwidth = dsp.range_resolution(self.radar_cfg['profiles'][0]['adcSamples'],
                                             self.radar_cfg['profiles'][0]['adcSampleRate'] / 1000,
                                             self.radar_cfg['profiles'][0]['freqSlopeConst'] / 1e12)
        self.Rmax = self.range_resolution * self.numRangeBins
        self.doppler_resolution = dsp.doppler_resolution(bandwidth,
                                      start_freq_const=self.radar_cfg['profiles'][0]['start_frequency'] / 1e9,
                                      ramp_end_time=self.radar_cfg['profiles'][0]['rampEndTime'] * 1e6,
                                      idle_time_const=self.radar_cfg['profiles'][0]['idle'] * 1e6,
                                      num_loops_per_frame=self.radar_cfg['numChirps'] / len(self.radar_cfg['chirps']),
                                      num_tx_antennas=self.radar_cfg['numTx'])
        self.angle_range = 90 #  (int): The desired span of thetas for the angle spectrum. Used for gen_steering_vec
        self.angle_resolution = 1 #  (float): The desired angular resolution for gen_steering_vec
        self.angle_bins = (self.angle_range * 2) // self.angle_resolution + 1
        self.num_vec, self.steering_vec = gen_steering_vec(self.angle_range, self.angle_resolution, self.virt_ant)
        return

    def plot_to_array(self, plt):
        # Convert the Matplotlib plot to a NumPy array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plot_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plot_img = cv2.imdecode(plot_arr, 1)
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB) # Convert to RGB from BGR
        return plot_img

    def frames_to_video(self, features):
        stack_frames = []
        function_dict = {
            'RAD': self.get_RAD,
            'RD': self.get_RD,
            'RA': self.get_RA,
            'spectrogram': self.get_spectrogram,
            'radarPC': self.get_radarpointcloud,
            'lidarPC': self.get_lidarpointcloud,
            'image': self.get_image,
            'depth_image': self.get_depthimage,
        }
        logging.error(f"total frames: {len(self.sync_indices)}")
        # for idx in range(len(self.sync_indices)):
        #     lst = []
        #     for f in features:
        #         feature_data = function_dict[f](idx)
        #         #logging.error(f"raw {f}: {feature_data.shape}")
                
        #         if f == 'RD': 
        #             #logging.error('RD begin->')
        #             feature_data = (feature_data -feature_data.min())/(feature_data.max()-feature_data.min())*255
        #             feature_data = cv2.cvtColor(feature_data.astype('uint8'), cv2.COLOR_GRAY2BGR)
        #             feature_data = cv2.applyColorMap(feature_data, cv2.COLORMAP_VIRIDIS)
        #             feature_data = cv2.transpose(feature_data)
        #             feature_data = cv2.flip(feature_data, flipCode=0)
        #             feature_data = cv2.resize(feature_data, dsize=(192, 720))
        #             #logging.error('RD created')
        #             #logging.error(f"converted {f}->:{feature_data.shape}")
        #         elif f == 'RA' or f == 'spectrogram': 
        #             #logging.error('RA begin->')
        #             feature_data = (feature_data -feature_data.min())/(feature_data.max()-feature_data.min())*255
        #             feature_data = cv2.cvtColor(feature_data.astype('uint8'), cv2.COLOR_GRAY2BGR)
        #             feature_data = cv2.applyColorMap(feature_data, cv2.COLORMAP_VIRIDIS)
        #             feature_data = cv2.flip(feature_data, flipCode=-1)
        #             feature_data = cv2.resize(feature_data, dsize=(432, 720))
        #             #logging.error('RA created')
        #         elif f == 'depth_image':
        #             feature_data = cv2.cvtColor(feature_data.astype('uint8'),cv2.COLOR_GRAY2BGR)
        #         elif f == 'radarPC' or f == 'lidarPC': # point cloud
        #             #logging.error('PCL begin->')
        #             #logging.error(feature_data)
        #             plt.figure(figsize=(8, 6))
        #             if feature_data.size > 0:
        #                 plt.plot(feature_data[:,1], feature_data[:,0], '.')
        #             plt.xlim(-20, 20)
        #             plt.ylim(0, 20)
        #             plt.grid()
        #             plot_arr = self.plot_to_array(plt)
        #             #logging.error(f"raw plot: {plot_arr.shape}")
        #             plt.close()
        #             feature_data = cv2.resize(plot_arr, dsize=(1024, 720))
        #             #logging.error(f"converted {f}->:{feature_data.shape}")
        #             #logging.error('PCL created')

        #         lst.append(feature_data)
                   
        #     frame = np.hstack(lst)
        #     stack_frames.append(frame)
        #     logging.error(idx)

        # output_path = os.path.join(self.feature_path, '_'.join(features) + '.mp4')
        # with imageio.get_writer(output_path, mode='I', fps=5) as writer:
        #     for idx, img in enumerate(stack_frames):
        #         cv2.putText(img, f"Frame: {idx}", 
        #                     (10, 30),  # Position
        #                     cv2.FONT_HERSHEY_SIMPLEX,  # Font
        #                     1,  # Font scale
        #                     (0, 255, 0),  # Color (Green in this case)
        #                     2)  # Line thickness

        #         writer.append_data(img)

####new
        output_path = os.path.join(self.feature_path, '_'.join(features) + '.mp4')
        with imageio.get_writer(output_path, mode='I', fps=5) as writer:
            for idx in range(len(self.sync_indices)):
                lst = []
                for f in features:
                    feature_data = function_dict[f](idx)
                    #logging.error(f"raw {f}: {feature_data.shape}")
                    
                    if f == 'RD': 
                        #logging.error('RD begin->')
                        feature_data = (feature_data -feature_data.min())/(feature_data.max()-feature_data.min())*255
                        feature_data = cv2.cvtColor(feature_data.astype('uint8'), cv2.COLOR_GRAY2BGR)
                        feature_data = cv2.applyColorMap(feature_data, cv2.COLORMAP_VIRIDIS)
                        feature_data = cv2.transpose(feature_data)
                        feature_data = cv2.flip(feature_data, flipCode=0)
                        feature_data = cv2.resize(feature_data, dsize=(192, 720))
                        #logging.error('RD created')
                        #logging.error(f"converted {f}->:{feature_data.shape}")
                    elif f == 'RA' or f == 'spectrogram': 
                        #logging.error('RA begin->')
                        feature_data = (feature_data -feature_data.min())/(feature_data.max()-feature_data.min())*255
                        feature_data = cv2.cvtColor(feature_data.astype('uint8'), cv2.COLOR_GRAY2BGR)
                        feature_data = cv2.applyColorMap(feature_data, cv2.COLORMAP_VIRIDIS)
                        feature_data = cv2.flip(feature_data, flipCode=-1)
                        feature_data = cv2.resize(feature_data, dsize=(432, 720))
                        #logging.error('RA created')
                    elif f == 'depth_image':
                        feature_data = cv2.cvtColor(feature_data.astype('uint8'),cv2.COLOR_GRAY2BGR)
                    elif f == 'radarPC' or f == 'lidarPC': # point cloud
                        #logging.error('PCL begin->')
                        #logging.error(feature_data)
                        plt.figure(figsize=(8, 6))
                        if feature_data.size > 0:
                            plt.plot(feature_data[:,1], feature_data[:,0], '.')
                        plt.xlim(-20, 20)
                        plt.ylim(0, 20)
                        plt.grid()
                        plot_arr = self.plot_to_array(plt)
                        #logging.error(f"raw plot: {plot_arr.shape}")
                        plt.close()
                        feature_data = cv2.resize(plot_arr, dsize=(1024, 720))
                        #logging.error(f"converted {f}->:{feature_data.shape}")
                        #logging.error('PCL created')

                    lst.append(feature_data)
                    
                frame = np.hstack(lst)
                cv2.putText(frame, f"Frame: {idx}", 
                                (10, 30),  # Position
                                cv2.FONT_HERSHEY_SIMPLEX,  # Font
                                1,  # Font scale
                                (0, 255, 0),  # Color (Green in this case)
                                2)  # Line thickness

                writer.append_data(frame)
                logging.error(idx)
####
        return output_path

    def get_RAD(self, idx=None, for_visualize=False):
        return None
    
    def get_ADC(self, idx=None, for_visualize=False):
        if self.sync_mode:
            index = self.sync_indices[idx]['adc']
        else:
            index = idx
        adc_file = os.path.join(self.feature_path,'adc',f"adc_{index}.npy")
        adc = np.load(adc_file)
        return adc
    
    def get_RA(self, idx=None, for_visualize=False):
        adc = self.get_ADC(idx)
        # rf = RadarFrame(radar_config)
        # beamformed_range_azimuth = rf.compute_range_azimuth(adc) 
        range_cube = dsp.range_processing(adc, window_type_1d=Window.BLACKMAN)
        range_cube = np.swapaxes(range_cube, 0, 2)
        ra = np.zeros((self.numRangeBins, self.angle_bins), dtype=complex)
        for i in range(self.numRangeBins):
            ra[i,:], _ = dsp.aoa_capon(range_cube[i], self.steering_vec)
        np.flipud(np.fliplr(ra))
        ra = np.log(np.abs(ra))  
        return ra 

    def get_RD(self, idx=None, for_visualize=False):
        # rf = RadarFrame(radar_config)
        # rf.raw_cube = self.get_ADC(idx)
        # range_doppler = rf.range_doppler
        adc = self.get_ADC(idx)
        range_cube = dsp.range_processing(adc, window_type_1d=Window.BLACKMAN)
        range_doppler, _ = dsp.doppler_processing(range_cube, interleaved=False, num_tx_antennas=2, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)
        range_doppler = np.fft.fftshift(range_doppler, axes=1)
        range_doppler[np.isinf(range_doppler)] = 0  # replace Inf with zero
        # rd = np.concatenate([range_doppler.real,range_doppler.imag], axis=2)
        return range_doppler

    def get_radarpointcloud(self, idx=None, for_visualize=False):
        adc = self.get_ADC(idx)
        #logging.error(f"#########################################")
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

        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', '(2,)<f4', '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()
        #logging.error(f"detObj2DRaw:{detObj2DRaw.shape}")
        # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, self.numDopplerBins, reserve_neighbor=True)
        # logging.error(f"detObj2DRaw:{detObj2DRaw.shape}")
        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, self.numDopplerBins)
        SNRThresholds2 = np.array([[2, 15], [10, 10], [35, 10]])
        peakValThresholds2 = np.array([[2, 50]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, 
                                           max_range=self.numRangeBins, min_range=0.5, range_resolution=self.range_resolution)
        #logging.error(f"detObj2D:{detObj2D.shape}")
        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
        # logging.error(f"azimuthInput:{azimuthInput.shape}")

        # 4. AoA
        num_vec, steering_vec = gen_steering_vec(self.angle_range, self.angle_resolution, 8)

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
        #logging.error(f"Total points: {len(points)}, range resolution: {self.range_resolution}")   
        return np.array(points) 

    def get_lidarpointcloud(self, idx=None, for_visualize=False):
        return None
    
    def get_spectrogram(self, idx=None, for_visualize=False):
        adc = self.get_ADC(idx)
        return None

    def get_image(self, idx=None, for_visualize=False):
        if self.sync_mode:
            index = self.sync_indices[idx]['image']
        else:
            index = idx
        image_path = os.path.join(self.feature_path,'image',f"image_{index}.npy")
        #image = np.load(image_path)
        return image_path
    
    def get_depthimage(self, idx=None, for_visualize=False):
        if self.sync_mode:
            index = self.sync_indices[idx]['depth_image']
        else:
            index = idx
        image_path = os.path.join(self.feature_path,'depth_image',f"depth_image_{index}.npy")
        #image = np.load(image_path)
        return image_path

    def __len__(self):
        return self.frame_sync
    
    def __getitem__(self, idx):
        data_dict = {}
        for feature in self.features:
            data_dict[feature] = getattr(self, feature)[idx]
        return data_dict
    
    def get_feature(self, feature_name, idx=None, for_visualize=False):
        function_dict = {
            'RAD': self.get_RAD,
            'RD': self.get_RD,
            'RA': self.get_RA,
            'spectrogram': self.get_spectrogram,
            'radarPC': self.get_radarpointcloud,
            'lidarPC': self.get_lidarpointcloud,
            'image': self.get_image,
            'depth_image': self.get_depthimage,
        }
        feature_data = function_dict[feature_name](idx, for_visualize=for_visualize)
        return feature_data

    def get_label(self, feature_name, idx=None):
        # RaDICaL dataset has no groundtruth label
        return []


class RADIal(Dataset):
    name = "RADIal ready-to-use dataset instance"  # RADIal data has two formats: raw and ready-to-use

    def __init__(self):
        self.config = ""

        self.frame_sync = 8252
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

    def get_RAD(self, idx=None, for_visualize=False):
        return None
    
    def get_RA(self, idx=None, for_visualize=False):
        return None

    def get_RD(self, idx=None, for_visualize=False):
        rd_raw = np.load(self.RD_filenames[idx])
        if for_visualize:
            #rangedoppler = feature_image[...,::2] + 1j * feature_image[...,1::2]
            power_spectrum = np.sum(np.abs(rd_raw), axis=2)
            power_spectrum = np.log10(power_spectrum)
            return power_spectrum
        else:
            rd = np.concatenate([rd_raw.real,rd_raw.imag], axis=2)
            return rd

    def get_radarpointcloud(self, idx=None, for_visualize=False):
        # range,azimuth,elevation,power,doppler,x,y,z,v
        pc = np.load(self.radarpointcloud_filenames[idx], allow_pickle=True)[[5,6,7],:]   # Keeps only x,y,z
        pc = np.rollaxis(pc,1,0)
        pc[:,1] *= -1
        return pc

    def get_lidarpointcloud(self, idx=None, for_visualize=False):
        return np.load(self.lidarpointcloud_filenames[idx], allow_pickle=True)[:,:3]

    def get_image(self, idx=None, for_visualize=False): 
        #image = np.asarray(Image.open(self.image_filenames[idx]))
        return self.image_filenames[idx]

    def get_depthimage(self, idx=None, for_visualize=False):
        return None
      
    def get_spectrogram(self, idx=None, for_visualize=False):
        return None

    def get_feature(self, feature_name, idx=None, for_visualize=False):
        function_dict = {
            'RAD': self.get_RAD,
            'RD': self.get_RD,
            'RA': self.get_RA,
            'spectrogram': self.get_spectrogram,
            'radarPC': self.get_radarpointcloud,
            'lidarPC': self.get_lidarpointcloud,
            'image': self.get_image,
            'depth_image': self.get_depthimage,
        }
        feature_data = function_dict[feature_name](idx, for_visualize=for_visualize)
        return feature_data
    
    def get_label(self, feature_name, idx=None):
        sample_id = self.sample_keys[idx] 
        entries_indexes = self.label_dict[sample_id]
        labels = self.labels[entries_indexes]
        labels = labels[:,1:-3].astype(np.float32)
        gt = []
        for label in labels:
            if(label[0]==-1):
                break # -1 means no object
            if feature_name == "image":
                # Note: coordinates are divided by 2 as image were saved in quarter resolution
                vertices_on_image = [x/2 for x in label[:4].tolist()]  
                gt.append(vertices_on_image)
            elif feature_name == "lidarPC":
                gt.append(label[4:6].tolist())
            elif feature_name == "radarPC":
                gt.append(label[7:9].tolist())
            else: # RD
                label[9] *= 512/103 # 512 range bins for 103m
                gt.append(label[[9,11]].tolist())
        return gt
    
    def prepare_for_train(self, features, train_cfg, model_cfg, splittype=None):
        self.segmentation_head = True if model_cfg['segmentation_head'] else False
        return

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
        # format as following: [Range,Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix,y2_pix]
        box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 

        encoded_label = self.encode(box_labels).copy()      

        # Read the Radar FFT data
        # radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npy".format(sample_id))
        # input = np.load(radar_name,allow_pickle=True)
        # radar_FFT = np.concatenate([input.real,input.imag],axis=2)
        radar_FFT = self.get_RD(index)
        if(self.statistics is not None):
            for i in range(len(self.statistics['input_mean'])):
                radar_FFT[...,i] -= self.statistics['input_mean'][i]
                radar_FFT[...,i] /= self.statistics['input_std'][i]
        radar_FFT = np.transpose(radar_FFT, axes=(2,0,1))
        data_dict = {'RD': radar_FFT, 'encoded_label': encoded_label, 'box_label': box_labels}
        
        # Read the segmentation map
        if self.segmentation_head:
            segmap_name = os.path.join(self.root_dir,'radar_Freespace',"freespace_{:06d}.png".format(sample_id))
            segmap = Image.open(segmap_name) # [512,900]
            # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
            # We crop the fov to 89.6deg
            segmap = self.crop(segmap)
            # and we resize to half of its size
            segmap = np.asarray(self.resize(segmap))==255
            data_dict.update({'seg_label': segmap})
   
        return data_dict # radar_FFT, segmap, encoded_label, box_labels
    
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
    name = "RADIal raw dataset instance"  # RADIal data has two formats: raw and ready-to-use

    def __init__(self):
        self.feature_path = ""
        self.config = ""

        self.frame_sync = 0
        self.features = ['image', 'lidarPC', 'adc']

        # Radar parameters
        self.numSamplePerChirp = 512
        self.numRxPerChip = 4
        self.numChirps = 256
        self.numRxAnt = 16
        self.numTxAnt = 12
        self.numReducedDoppler = 16
        self.numChirpsPerLoop = 16
    
    def parse_recording(self, folder):
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
        if not master:
            # by default, we use the Radar as Matser sensor
            if 'radar_ch0' not in self.dicts or 'radar_ch1' not in self.dicts or 'radar_ch2' not in self.dicts or 'radar_ch3' not in self.dicts:
                print('Error: recording does not contains the 4 radar chips')
            
            keys =list(self.dicts.keys())
            self.keys = keys
            
            if 'gps' in self.dicts:
                keys.remove('gps')
            if 'preview' in self.dicts:
                keys.remove('preview')
            if 'None' in self.dicts:
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

                if len(idx0)==0 or len(idx1)==0 or len(idx2)==0:
                    id_to_del.append(i)
                    nb_corrupted+=1
                    match['radar_ch0'] = -1
                    match['radar_ch1'] = -1
                    match['radar_ch2'] = -1
                else:
                    match['radar_ch0'] = idx0[0]
                    match['radar_ch1'] = idx1[0]
                    match['radar_ch2'] = idx2[0]

                
                if self.sync_mode=='timestamp':
                    for k in keys:
                        if len(self.dicts[k]['timestamp'])>0:
                            time_diff = np.abs(np.asarray(self.dicts[k]['timestamp']) - timestamp)
                            vmin = time_diff.min()
                            index_min = time_diff.argmin()

                            if vmin>tolerance:
                                index_min=-1
                        else:
                            index_min=-1
                        
                        if(index_min==-1):
                            nb_tolerance+=1
                            id_to_del.append(i)

                        match[k] = index_min
                else:
                    for k in keys:
                        if len(self.dicts[k]['timeofissue'])>0:
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
            self.table = self.table[self.id_valid]
            if not silent:
                print(f'Total tolerance errors: {nb_tolerance/len(self.table)*100:.2f}%')
                print(f'Total corrupted frames: {nb_corrupted/len(self.table)*100:.2f}%')
        elif master=='camera':
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
            self.table = self.table[id_to_keep]
            if not silent:
                print(f'Total tolerance errors: {nb_tolerance/len(self.table)*100:.2f}%')
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
    
    def get_RD(self, idx=None, for_visualize=False):
        file = os.path.join(self.feature_path,"adc",f"adc_{idx}.npy")
        complex_adc = np.load(file)
        # 2- Remoce DC offset
        complex_adc = complex_adc - np.mean(complex_adc, axis=(0,1))

        # 3- Range FFTs
        range_fft = np.fft(np.multiply(complex_adc,self.range_fft_coef),self.numSamplePerChirp,axis=0) # mkl
    
        # 4- Doppler FFts
        RD_spectrums = np.fft(np.multiply(range_fft,self.doppler_fft_coef),self.numChirps,axis=1) # mkl
        return RD_spectrums
    
    def get_RA(self, idx=None, for_visualize=False):
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
    

    def get_radarpointcloud(self, idx=None, for_visualize=False):
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

    def get_lidarpointcloud(self, idx=None, for_visualize=False):
        file = os.path.join(self.feature_path,"lidarpc",f"lidarpc_{idx}.npy")
        pc = np.load(file)
        return pc

    def get_image(self, idx=None, for_visualize=False): 
        image_path = os.path.join(self.feature_path,"image",f"image_{idx}.jpg")
        #image = np.asarray(Image.open(image_path))
        return image_path

    def get_depthimage(self, idx=None, for_visualize=False):
        return None
      
    def get_spectrogram(self, idx=None, for_visualize=False):
        return None
    
    def get_feature(self, feature_name, idx=None, for_visualize=False):
        function_dict = {
            'RAD': self.get_RAD,
            'RD': self.get_RD,
            'RA': self.get_RA,
            'spectrogram': self.get_spectrogram,
            'radarPC': self.get_radarpointcloud,
            'lidarPC': self.get_lidarpointcloud,
            'image': self.get_image,
            'depth_image': self.get_depthimage,
        }
        feature_data = function_dict[feature_name](idx, for_visualize=for_visualize)
        return feature_data
    
    def get_label(self, feature_name, idx=None):
        # Raw Radial dataset is partly labeled
        return []



class CRUW(Dataset):
    name = "CRUW ROD2021 dataset instance"

    def __init__(self, features=None):
        self.feature_path = ""
        self.config = ""

        self.frame_sync = 0
        self.features = ['image', 'RA']

    def parse(self, file_id, file_path, file_name, config):
        self.config = config
        self.root_path = file_path
        self.seq_name = file_name

        def get_sorted_filenames(directory):
            # Get a sorted list of all file names in the given directory
            return sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])
        
        self.image_filenames = get_sorted_filenames(os.path.join(self.root_path, 'TRAIN_CAM_0', file_name, 'IMAGES_0'))
        self.frame_sync = len(self.image_filenames)

        self.labels = os.path.join(self.root_path, 'TRAIN_RAD_H_ANNO', self.seq_name+'.txt')
        with open(self.config, 'r') as file:
            self.sensor_cfg = json.load(file)
        camera_configs = self.sensor_cfg['camera_cfg']
        radar_configs = self.sensor_cfg['radar_cfg']
        n_class = 3

        # Create pkl file to keep data infos in the seq file
        self.feature_path = Path(os.getenv('TMP_ROOTDIR')).joinpath(str(file_id))
        self.feature_path.mkdir(parents=True, exist_ok=True)
        self.pkl_path = os.path.join(self.feature_path, self.seq_name + '.pkl')
        print("Sequence %s saving to %s" % (self.seq_name, self.pkl_path))
        overwrite = False
        # if overwrite:
        #     if os.path.exists(os.path.join(data_dir, split)):
        #         shutil.rmtree(os.path.join(data_dir, split))
        #     os.makedirs(os.path.join(data_dir, split))
        try:
            if not overwrite and os.path.exists(self.pkl_path):
                print("%s already exists, skip" % self.pkl_path)
                return

            image_dir = os.path.join(self.root_path, 'TRAIN_CAM_0', self.seq_name, camera_configs['image_folder'])
            if os.path.exists(image_dir):
                image_paths = sorted([os.path.join(image_dir, name) for name in os.listdir(image_dir) if
                                    name.endswith(camera_configs['ext'])])
                n_frame = len(image_paths)
            else:  # camera images are not available
                image_paths = None
                n_frame = None

            radar_dir = os.path.join(self.root_path, 'TRAIN_RAD_H', self.seq_name, radar_configs['chirp_folder'])
            if n_frame is not None:
                assert len(os.listdir(radar_dir)) == n_frame * len(radar_configs['chirp_ids'])
            else:  # radar frames are not available
                n_frame = int(len(os.listdir(radar_dir)) / len(radar_configs['chirp_ids']))
            radar_paths = []
            for frame_id in range(n_frame):
                chirp_paths = []
                for chirp_id in radar_configs['chirp_ids']:
                    path = os.path.join(radar_dir, '%06d_%04d.' % (frame_id, chirp_id) +
                                        radar_configs['ext'])
                    chirp_paths.append(path)
                radar_paths.append(chirp_paths)
            # else:
            #     raise ValueError

            data_dict = dict(
                data_root=self.root_path,
                # data_path=seq_path,
                seq_name=self.seq_name,
                n_frame=n_frame,
                image_paths=image_paths,
                radar_paths=radar_paths,
                anno=None,
            )

            # if split == 'demo' or not os.path.exists(seq_anno_path):
            #     # no labels need to be saved
            #     pickle.dump(data_dict, open(self.pkl_path, 'wb'))
            #     continue
            # else:
            anno_obj = {}
            seq_anno_path = os.path.join(self.root_path, 'TRAIN_RAD_H_ANNO', self.seq_name + '.txt')
            #if config_dict['dataset_cfg']['anno_ext'] == '.txt':
            anno_obj['metadata'] = load_anno_txt(seq_anno_path, n_frame, radar_configs)
            # elif config_dict['dataset_cfg']['anno_ext'] == '.json':
            #     with open(os.path.join(seq_anno_path), 'r') as f:
            #         anno = json.load(f)
            #     anno_obj['metadata'] = anno['metadata']
            # else:
            #     raise

            anno_obj['confmaps'] = generate_confmaps(anno_obj['metadata'], n_class, False, radar_configs)
            data_dict['anno'] = anno_obj
            # save pkl files
            pickle.dump(data_dict, open(self.pkl_path, 'wb'))
            # end frames loop
        except Exception as e:
            print("Error while preparing %s: %s" % (self.seq_name, e))
        return 
    
    def get_image(self, idx=None, for_visualize=False): 
        #image = np.asarray(Image.open(self.image_filenames[idx]))
        return self.image_filenames[idx]
    
    def get_RA(self, idx=None, for_visualize=False):
        chirp_path = os.path.join(self.root_path, 'TRAIN_RAD_H', self.seq_name, 'RADAR_RA_H', '%06d_0000.npy' % idx) # 000000_0192.npy
        ra = np.load(chirp_path)
        if for_visualize:
            ra_image = np.sqrt(ra[:, :, 0] ** 2 + ra[:, :, 1] ** 2)
            return ra_image
        else:
            return ra

    def get_RAD(self, idx=None, for_visualize=False):
        return None
    
    def get_RD(self, idx=None):
        return None

    def get_radarpointcloud(self, idx=None, for_visualize=False):
        return None

    def get_lidarpointcloud(self, idx=None, for_visualize=False):
        return None

    def get_depthimage(self, idx=None, for_visualize=False):
        return None
      
    def get_spectrogram(self, idx=None, for_visualize=False):
        return None

    def get_feature(self, feature_name, idx=None, for_visualize=False):
        function_dict = {
            'RAD': self.get_RAD,
            'RD': self.get_RD,
            'RA': self.get_RA,
            'spectrogram': self.get_spectrogram,
            'radarPC': self.get_radarpointcloud,
            'lidarPC': self.get_lidarpointcloud,
            'image': self.get_image,
            'depth_image': self.get_depthimage,
        }
        feature_data = function_dict[feature_name](idx, for_visualize=for_visualize)
        return feature_data
    
    def get_label(self, feature_name, idx=None):
        seq_details = pickle.load(open(self.pkl_path, 'rb'))
        lst_categories = seq_details['anno']['metadata'][idx]['rad_h']['obj_info']['categories']
        lst_center_ids = seq_details['anno']['metadata'][idx]['rad_h']['obj_info']['center_ids']

        gt = []
        for center_id, category in zip(lst_center_ids, lst_categories):
            if feature_name == "RA":
                center_id = [int(x) for x in center_id]
                gt.append(center_id + [category])
        return gt
    
    def prepare_for_train(self, features, train_cfg, model_cfg, splittype=None):
        self.n_class = 3 # dataset.object_cfg.n_class
        self.win_size = train_cfg['win_size'] 

        if splittype in ('train', 'val') or splittype is None:
            self.step = train_cfg['train_step']
            self.stride = train_cfg['train_stride']
        else:
            self.step = train_cfg['test_step']
            self.stride = train_cfg['test_stride']

        self.is_random_chirp = True
        self.noise_channel = False

        # Dataloader for MNet
        if 'mnet_cfg' in model_cfg:
            in_chirps, _ = model_cfg['mnet_cfg']
            self.n_chirps = in_chirps
        elif model_cfg['class'] in ('RECORD', 'RECORDNoLstm', 'RECORDNoLstmMulti'):
            self.n_chirps = 4
        else:
            self.n_chirps = 1

        self.chirp_ids = self.sensor_cfg['radar_cfg']['chirp_ids']

        data_details = pickle.load(open(self.pkl_path, 'rb'))
        self.image_paths = data_details['image_paths']
        self.radar_paths = data_details['radar_paths']
        if data_details['anno'] is not None:
            self.obj_infos = data_details['anno']['metadata']
            self.confmaps = data_details['anno']['confmaps']
        n_frame = data_details['n_frame']
        n_data_in_seq = (n_frame - (self.win_size * self.step - 1)) // self.stride + (
            1 if (n_frame - (self.win_size * self.step - 1)) % self.stride > 0 else 0)
        self.datasamples_length = n_data_in_seq

        self.model_type = model_cfg['class']
        self.rng_grid = confmap2ra(self.sensor_cfg['radar_cfg'], name='range')
        self.agl_grid = confmap2ra(self.sensor_cfg['radar_cfg'], name='angle')
        return
    
    def __len__(self):
        return self.datasamples_length #self.frame_sync

    def __getitem__(self, index):
        data_dict = dict(
            status=True,
            seq_name=self.seq_name,
            image_paths=[]
        )

        if self.is_random_chirp:
            chirp_id = random.randint(0, len(self.chirp_ids) - 1)
        else:
            chirp_id = 0

        # Dataloader for MNet
        if self.n_chirps > 1:
            chirp_id = self.chirp_ids
        # if 'mnet_cfg' in self.model_cfg:
        #     chirp_id = self.chirp_ids

        radar_configs = self.sensor_cfg['radar_cfg']
        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']

        # Load radar data
        try:
            data_id = index * self.stride
            data_dict['start_frame'] = data_id
            data_dict['end_frame'] = data_id + self.win_size * self.step - 1
            if self.model_type in ("RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2", "RadarFormer_hrformer2d"):
                if isinstance(chirp_id, int):
                    radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                    for idx, frameid in enumerate(range(data_id, data_id + self.win_size * self.step, self.step)):
                        radar_npy_win[idx, :, :, :] = np.load(self.radar_paths[frameid][chirp_id])
                        data_dict['image_paths'].append(self.image_paths[frameid])
                elif isinstance(chirp_id, list):
                    radar_npy_win = np.zeros((self.win_size, self.n_chirps, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                    for idx, frameid in enumerate(range(data_id, data_id + self.win_size * self.step, self.step)):
                        for cid, c in enumerate(chirp_id):
                            npy_path = self.radar_paths[frameid][cid]
                            radar_npy_win[idx, cid, :, :, :] = np.load(npy_path)
                        data_dict['image_paths'].append(self.image_paths[frameid])
                else:
                    raise TypeError
                
                # Dataloader for MNet
                if self.n_chirps > 1: # if 'mnet_cfg' in self.model_cfg:
                    radar_npy_win = np.transpose(radar_npy_win, (4, 0, 1, 2, 3))
                    assert radar_npy_win.shape == (2, self.win_size, self.n_chirps, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
                else:
                    radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
                    assert radar_npy_win.shape == (2, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            elif self.model_type in ('RECORD', 'RECORDNoLstm', 'RECORDNoLstmMulti'):
                if isinstance(chirp_id, int):
                    radar_npy_win = torch.zeros((self.win_size, 2, ramap_rsize, ramap_asize), dtype=torch.float32)
                    for idx, frameid in enumerate(range(data_id, data_id + self.win_size * self.step, self.step)):
                        radar_npy_win[idx, :, :, :] = torch.from_numpy(
                            np.transpose(np.load(self.radar_paths[frameid][chirp_id]), (2, 0, 1)))
                        #if self.split != 'test':
                        data_dict['image_paths'].append(self.image_paths[frameid])
                        # else:
                        #     data_dict['image_paths'].append(self.radar_paths[frameid])
                elif isinstance(chirp_id, list):
                    radar_npy_win = torch.zeros((self.win_size, self.n_chirps * 2, ramap_rsize, ramap_asize), dtype=torch.float32)
                    for idx, frameid in enumerate(range(data_id, data_id + self.win_size * self.step, self.step)):
                        for cid, c in enumerate(chirp_id):
                            npy_path = self.radar_paths[frameid][cid]
                            radar_npy_win[idx, cid * 2:cid * 2 + 2, :, :] = torch.from_numpy(
                                np.transpose(np.load(npy_path), (2, 0, 1)))
                        # if self.split != 'test':
                        data_dict['image_paths'].append(self.image_paths[frameid])
                        # else:
                        #     data_dict['image_paths'].append(self.radar_paths[frameid])
                else:
                    raise TypeError
                radar_npy_win = radar_npy_win.transpose(1, 0)

                if self.model_type == 'RECORDNoLstmMulti':
                    c, t, h, w = radar_npy_win.shape
                    radar_npy_win = radar_npy_win.reshape(c*t, h, w)
                elif self.model_type == 'RECORDNoLstm':
                    c, t, h, w = radar_npy_win.shape
                    assert t == 1
                    radar_npy_win = radar_npy_win.reshape(c, h, w)
                    
            else:
                raise ValueError
            
            data_dict['RA'] = radar_npy_win
        except:
            # in case load npy fail
            data_dict['status'] = False
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
            log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
            with open(os.path.join('./tmp', log_name), 'w') as f_log:
                f_log.write('npy path: ' + self.radar_paths[frameid][chirp_id] + \
                            '\nframe indices: %d:%d:%d' % (data_id, data_id + self.win_size * self.step, self.step))
            return data_dict


        # Load annotations
        if len(self.confmaps) != 0:
            confmap_gt = self.confmaps[data_id:data_id + self.win_size * self.step:self.step]
            confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            obj_info = self.obj_infos[data_id:data_id + self.win_size * self.step:self.step]
            if self.noise_channel:
                assert confmap_gt.shape == \
                    (self.n_class + 1, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            else:
                confmap_gt = confmap_gt[:self.n_class]
                assert confmap_gt.shape == \
                    (self.n_class, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

            # data_dict['anno'] = dict(
            #     obj_infos=obj_info,
            #     confmaps=confmap_gt,
            # )
            data_dict.update({'obj_infos': obj_info, 'gt_mask': confmap_gt})
        # else:
        #     data_dict['anno'] = None

        if self.model_type in ('RECORD', 'RECORDNoLstm', 'RECORDNoLstmMulti') and data_dict['gt_mask'] is not None: 
            data_dict['gt_mask'] = data_dict['gt_mask'][:, -1]
        return data_dict


class CARRADA(Dataset):
    name = "CARRADA dataset instance"

    def __init__(self, features=None):
        self.feature_path = ""
        self.config = ""

        self.frame_sync = 0
        self.features = ['image', 'RA', 'RD', 'AD', 'RAD']

    def parse(self, file_id, file_path, file_name, config):
        self.config = config
        self.root_path = file_path
        self.seq_name = file_name
        self.path_to_seq = os.path.join(self.root_path, 'Carrada', file_name)

        def get_sorted_filenames(directory):
            # Get a sorted list of all file names in the given directory
            return sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])
        
        self.image_filenames = get_sorted_filenames(os.path.join(self.path_to_seq, 'camera_images'))
        self.RD_filenames = get_sorted_filenames(os.path.join(self.path_to_seq, 'range_doppler_numpy'))
        self.RA_filenames = get_sorted_filenames(os.path.join(self.path_to_seq, 'range_angle_numpy'))
        self.AD_filenames = get_sorted_filenames(os.path.join(self.path_to_seq, 'angle_doppler_raw'))
        self.frame_sync = len(self.RD_filenames)
        self.RAD_filenames = get_sorted_filenames(os.path.join(self.root_path, 'Carrada_RAD', file_name, 'RAD_numpy'))

        anno_path = os.path.join(self.root_path, 'Carrada', 'annotations_frame_oriented.json')
        with open(anno_path, 'r') as fp:
            self.annos = json.load(fp)

    def get_image(self, idx=None, for_visualize=False): 
        #image = np.asarray(Image.open(self.image_filenames[idx]))
        return self.image_filenames[idx]
    
    def get_RA(self, idx=None, for_visualize=False):
        ra = np.load(self.RA_filenames[idx])
        return ra

    def get_RAD(self, idx=None, for_visualize=False):
        rad = np.load(self.RAD_filenames[idx])
        return rad
    
    def get_RD(self, idx=None, for_visualize=False):
        # idx = 117 # a testing frame with object labels in '2020-02-28-13-09-58'
        rd = np.load(self.RD_filenames[idx])
        if for_visualize:
            rd = np.rot90(rd, k=2) # To align with feature display in UI
        return rd

    def get_radarpointcloud(self, idx=None, for_visualize=False):
        return None

    def get_lidarpointcloud(self, idx=None, for_visualize=False):
        return None

    def get_depthimage(self, idx=None, for_visualize=False):
        return None
      
    def get_spectrogram(self, idx=None, for_visualize=False):
        return None
    
    def get_feature(self, feature_name, idx=None, for_visualize=False):
        function_dict = {
            'RAD': self.get_RAD,
            'RD': self.get_RD,
            'RA': self.get_RA,
            'spectrogram': self.get_spectrogram,
            'radarPC': self.get_radarpointcloud,
            'lidarPC': self.get_lidarpointcloud,
            'image': self.get_image,
            'depth_image': self.get_depthimage,
        }
        feature_data = function_dict[feature_name](idx, for_visualize=for_visualize)
        return feature_data
    
    def get_label(self, feature_name, idx=None):
        # idx = 117 # a testing frame with object labels in '2020-02-28-13-09-58'
        frame = "{:06d}".format(idx)
        objs = self.annos[self.seq_name][frame]
        gt = []
        categories = {0: 'background', 1: 'pedestrian', 2: 'cyclist', 3: 'car'}
        if objs:
            for obj in objs.values():
                category = categories[obj['range_angle']['label']]
                if feature_name == "RA":
                    points = obj['range_angle']['box']
                    gt.append([points[0][1], points[0][0], points[1][1], points[1][0], category])
                elif feature_name == "RD":
                    points = obj['range_doppler']['box']
                    height, width = 256, 64
                    lower_left_corner = [width - max(points[0][1], points[1][1]) - 1, height - max(points[0][0], points[1][0]) - 1]
                    upper_right_corner = [width - min(points[0][1], points[1][1]) - 1, height - min(points[0][0], points[1][0]) - 1]
                    obj = [
                        lower_left_corner[0], 
                        lower_left_corner[1], 
                        upper_right_corner[0], 
                        upper_right_corner[1]
                    ]
                    gt.append([obj[0], obj[1], obj[2], obj[3], category])
        return gt

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
    
    def prepare_for_train(self, features, train_cfg, model_cfg, splittype=None):
        self.win_frames = train_cfg['win_size']
        self.features = features

        radar_range_max, radar_range_resolution = 50, 0.2
        radar_vel_max, radar_vel_resolution = 13.43, 0.42
        radar_angle_max, radar_angle_resolution = 180, 0.7
        num_chirps_in_frame, num_samples_in_chirp, num_angles = 64, 256, 256
        self.rng_grid = [i * radar_range_resolution for i in range(num_samples_in_chirp)]
        self.agl_grid = [i * radar_angle_resolution / radar_angle_max * np.pi for i in range(int(- num_angles / 2), int(num_angles / 2))]
        self.dpl_grid = [ i * radar_vel_resolution for i in range(int(- num_chirps_in_frame / 2), int(num_chirps_in_frame / 2))]
        
        self.model_type = model_cfg['class']
        if self.model_type in ('RECORD', 'RECORDNoLstm', 'RECORDNoLstmMulti', 'MVRECORD', 'DAROD') :
            self.process_signal = True
            self.transformations = get_transformations(transform_names=train_cfg['transformations'])
            self.add_temp = True
            self.annotation_type = 'dense'
            self.path_to_annots = os.path.join(self.path_to_seq, 'annotations', self.annotation_type)
            self.norm_type = train_cfg['norm_type']
            self.n_class = 4
            self.datasamples_length = self.frame_sync - self.win_frames + 1
        elif self.model_type == 'RadarCrossAttention':
            anno_path = os.path.join(self.root_path, 'new_gt_anno.json')
            with open(anno_path, 'r') as f:
                self.annos = json.load(f)
            tracks = self.annos[self.seq_name]['tracks']
            # Get data frame intervals in the seq
            intervals = []
            for i, track in enumerate(tracks):
                if i == 0:
                    last_frame = track[0]
                    new_frame = track[0]
                    intervals.append(new_frame)
                elif i == len(tracks) - 1:
                    intervals.append(track[-1])
                else:
                    last_frame = tracks[i-1][0]
                    new_frame = track[0]
                    if new_frame != last_frame + 1:
                        intervals.append(last_frame+4)
                        intervals.append(new_frame)
            print(intervals)
            seq_intervals = []
            self.indexmapping = []
            self.datasamples_length = 0
            for i in range(int(len(intervals)/2)):
                seq_intervals.append([intervals[2*i], intervals[2*i + 1]])
                self.datasamples_length += intervals[2*i + 1] - intervals[2*i] + 2 - self.win_frames 
                self.indexmapping.extend(list(range(intervals[2*i], intervals[2*i + 1] + 2 - self.win_frames)))
            self.gauss_type = train_cfg['gauss_type']
            self.center_offset = model_cfg['center_offset']
            self.orientation = model_cfg['orientation']
        return

    def __len__(self):
        return self.datasamples_length

    def __getitem__(self, index):
        logger.debug(f"Data item index: {index}")
        if self.model_type in ('RECORD', 'RECORDNoLstm', 'RECORDNoLstmMulti', 'MVRECORD', 'DAROD') :
            frame_id = index + self.win_frames - 1
            init_frame_name = "{:06d}".format(frame_id)
            frame_names = [str(f_id).zfill(6) for f_id in range(frame_id-self.win_frames+1, frame_id+1)]
            if self.features == ['RD'] or self.features == ['RA']: 
                # Get radar feature data 
                featurestr = 'range_doppler' if self.features == ['RD'] else 'range_angle'
                feature_matrices = list()
                mask = np.load(os.path.join(self.path_to_annots, init_frame_name, f'{featurestr}.npy'))

                for frame_name in frame_names:
                    if self.process_signal:
                        feature_matrix = np.load(os.path.join(self.path_to_seq, f'{featurestr}_processed', frame_name + '.npy'))
                    else:
                        feature_matrix = np.load(os.path.join(self.path_to_seq, f'{featurestr}_raw', frame_name + '.npy'))
                    feature_matrices.append(feature_matrix)
                feature_matrix = np.dstack(feature_matrices)
                feature_matrix = np.rollaxis(feature_matrix, axis=-1)   
                feature_frame = {'matrix': feature_matrix, 'gt_mask': mask}
                # Apply the same transform to all representations
                if np.random.uniform(0, 1) > 0.5:
                    is_vflip = True
                else:
                    is_vflip = False
                if np.random.uniform(0, 1) > 0.5:
                    is_hflip = True
                else:
                    is_hflip = False
                feature_frame = self.transform(feature_frame, is_vflip=is_vflip, is_hflip=is_hflip)
                
                if  self.model_type in ('RECORD', 'RECORDNoLstm', 'RECORDNoLstmMulti'):
                    # Expand one more dim
                    if self.add_temp:
                        if isinstance(self.add_temp, bool):
                            feature_frame['matrix'] = np.expand_dims(feature_frame['matrix'], axis=0)
                        else:
                            assert isinstance(self.add_temp, int)
                            feature_frame['matrix'] = np.expand_dims(feature_frame['matrix'], axis=self.add_temp)
                    # Apply normalization
                    feature_frame['matrix'] = normalize(feature_frame['matrix'], featurestr, norm_type=self.norm_type)
                    if self.model_type == 'RECORDNoLstmMulti':
                        c, t, h, w = feature_frame['matrix'].shape
                        feature_frame['matrix'] = feature_frame['matrix'].reshape(c*t, h, w)
                    elif self.model_type == 'RECORDNoLstm':
                        c, t, h, w = feature_frame['matrix'].shape
                        assert t == 1
                        feature_frame['matrix'] = feature_frame['matrix'].reshape(c, h, w)
                    frame = {self.features[0]: feature_frame['matrix'], 'gt_mask': feature_frame['gt_mask']}
                else:
                    # Get ground truth boxes and labels for DAROD network
                    gt_boxes = []
                    gt_labels = []
                    if self.annos[self.seq_name][init_frame_name]:
                        for obj_anno in self.annos[self.seq_name][init_frame_name].values():
                            #obj_anno['range_doppler']['dense']
                            gt_boxes.append(obj_anno[featurestr]['box'])
                            gt_labels.append(obj_anno[featurestr]['label'])
                    frame = {self.features[0]: feature_frame['matrix'], 'label': gt_labels, 'boxes': gt_boxes}
            elif len(self.features) > 1:
                rd_matrices = list()
                ra_matrices = list()
                ad_matrices = list()
                rd_mask = np.load(os.path.join(self.path_to_annots, init_frame_name,
                                            'range_doppler.npy'))
                ra_mask = np.load(os.path.join(self.path_to_annots, init_frame_name,
                                            'range_angle.npy'))
                for frame_name in frame_names:
                    if self.process_signal:
                        rd_matrix = np.load(os.path.join(self.path_to_seq,
                                                        'range_doppler_processed',
                                                        frame_name + '.npy'))
                        ra_matrix = np.load(os.path.join(self.path_to_seq,
                                                        'range_angle_processed',
                                                        frame_name + '.npy'))
                        ad_matrix = np.load(os.path.join(self.path_to_seq,
                                                        'angle_doppler_processed',
                                                        frame_name + '.npy'))
                    else:
                        rd_matrix = np.load(os.path.join(self.path_to_seq,
                                                        'range_doppler_raw',
                                                        frame_name + '.npy'))
                        ra_matrix = np.load(os.path.join(self.path_to_seq,
                                                        'range_angle_raw',
                                                        frame_name + '.npy'))
                        ad_matrix = np.load(os.path.join(self.path_to_seq,
                                                        'angle_doppler_raw',
                                                        frame_name + '.npy'))

                    rd_matrices.append(rd_matrix)
                    ra_matrices.append(ra_matrix)
                    ad_matrices.append(ad_matrix)

                # Apply the same transfo to all representations
                if np.random.uniform(0, 1) > 0.5:
                    is_vflip = True
                else:
                    is_vflip = False
                if np.random.uniform(0, 1) > 0.5:
                    is_hflip = True
                else:
                    is_hflip = False

                rd_matrix = np.dstack(rd_matrices)
                rd_matrix = np.rollaxis(rd_matrix, axis=-1)
                rd_frame = {'matrix': rd_matrix, 'gt_mask': rd_mask}
                rd_frame = self.transform(rd_frame, is_vflip=is_vflip, is_hflip=is_hflip)
                if self.add_temp:
                    if isinstance(self.add_temp, bool):
                        rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'], axis=0)
                    else:
                        assert isinstance(self.add_temp, int)
                        rd_frame['matrix'] = np.expand_dims(rd_frame['matrix'],
                                                            axis=self.add_temp)

                ra_matrix = np.dstack(ra_matrices)
                ra_matrix = np.rollaxis(ra_matrix, axis=-1)
                ra_frame = {'matrix': ra_matrix, 'gt_mask': ra_mask}
                ra_frame = self.transform(ra_frame, is_vflip=is_vflip, is_hflip=is_hflip)
                if self.add_temp:
                    if isinstance(self.add_temp, bool):
                        ra_frame['matrix'] = np.expand_dims(ra_frame['matrix'], axis=0)
                    else:
                        assert isinstance(self.add_temp, int)
                        ra_frame['matrix'] = np.expand_dims(ra_frame['matrix'],
                                                            axis=self.add_temp)

                ad_matrix = np.dstack(ad_matrices)
                ad_matrix = np.rollaxis(ad_matrix, axis=-1)
                # Fill fake mask just to apply transform
                ad_frame = {'matrix': ad_matrix, 'gt_mask': rd_mask.copy()}
                ad_frame = self.transform(ad_frame, is_vflip=is_vflip, is_hflip=is_hflip)
                if self.add_temp:
                    if isinstance(self.add_temp, bool):
                        ad_frame['matrix'] = np.expand_dims(ad_frame['matrix'], axis=0)
                    else:
                        assert isinstance(self.add_temp, int)
                        ad_frame['matrix'] = np.expand_dims(ad_frame['matrix'],
                                                            axis=self.add_temp)

                rd_frame['matrix'] = normalize(rd_frame['matrix'], 'range_doppler', norm_type=self.norm_type)
                ra_frame['matrix'] = normalize(ra_frame['matrix'], 'range_angle', norm_type=self.norm_type)
                ad_frame['matrix'] = normalize(ad_frame['matrix'], 'angle_doppler', norm_type=self.norm_type)
                frame = {'RD': rd_frame['matrix'], 'rd_mask': rd_frame['gt_mask'],
                        'RA': ra_frame['matrix'], 'ra_mask': ra_frame['gt_mask'],
                        'AD': ad_frame['matrix']}

            # camera_path = os.path.join(self.path_to_seq, 'camera_images', frame_name + '.jpg')
            # frame.update({'image_path': camera_path})
        elif self.model_type == 'RadarCrossAttention':
            raw_index = self.indexmapping[index]
            frame = {'image_path': self.image_filenames[raw_index]}
            ra_map = np.zeros((self.win_frames, 256, 256))
            rd_map = np.zeros((self.win_frames, 256, 64))
            ad_map = np.zeros((self.win_frames, 64, 256))
            for i in range(self.win_frames):
                frame_id = raw_index + i
                ra_map[i, ::] = self.get_RA(frame_id)

                rd_map_temp = self.get_RD(frame_id)
                # flip the axis in the order of increasing velocity (L->R)
                rd_map_temp = np.flip(rd_map_temp, 0)
                rd_map_temp = np.flip(rd_map_temp, 1)
                rd_map_temp[:, 31:34]=0 # removing dc component
                rd_map[i, ::] = np.float32(rd_map_temp)

                rad_map = self.get_RAD(frame_id)
                ad_map_temp = np.fft.ifftshift(rad_map, axes=0)
                ad_map_temp = np.fft.ifft(ad_map_temp, axis=0)
                ad_map_temp = pow(np.abs(ad_map_temp), 2)
                ad_map_temp = np.sum(ad_map_temp, axis=0)
                ad_map_temp = 10*np.log10(ad_map_temp + 1)
                ad_map_temp = np.transpose(ad_map_temp)
                ad_map_temp[31:34,:]=0
                ad_map_temp = np.float32(ad_map_temp)
                ad_map[i, ::] = ad_map_temp
            # TODO: normalize rd_map, ad_map, rd_map
            raw_index = str(raw_index).zfill(6)
            if self.gauss_type == "Bivar":
                gauss_map = bi_var_gauss(self.annos[self.seq_name][raw_index])    
            else:                                        
                gauss_map = plain_gauss(self.annos[self.seq_name][raw_index], s_r=15, s_a=15)
            frame.update({'RD': rd_map, 'RA': ra_map, 'AD': ad_map, 'gt_mask': gauss_map})
            if self.center_offset:
                center_offset = get_center_map(self.annos[self.seq_name][raw_index], vect=get_co_vec()) 
                frame.update({'gt_center_map': center_offset}) 
            if self.orientation:
                orient_map = get_orent_map(self.annos[self.seq_name][raw_index])  
                frame.update({'gt_orent_map': orient_map}) 
            
            for key,value in frame.items():
                if key != 'image_path':
                    logger.debug(f"{key}: {value.shape}")
        else:
            raise ValueError(f"CARRADA dataset doesn't support the model type({self.model_type}) training/inference.")
        return frame



class RADDetDataset(Dataset):
    name = "RADDet dataset instance"

    def __init__(self, features=None):
        self.feature_path = ""
        self.config = ""

        self.frame_sync = 0
        self.features = ['image', 'RAD']
        self.radar_cfg = {
            "designed_frequency" : 76.8,
            "config_frequency" : 77,
            "range_size" : 256,
            "doppler_size" : 64,
            "azimuth_size" : 256,
            "range_resolution" : 0.1953125,
            "angular_resolution" : 0.006135923,
            "velocity_resolution" : 0.41968030701528203
		}

    def parse(self, file_id, file_path, file_name, config):
        self.config = config
        self.root_path = file_path
        self.seq_name = file_name

        def get_sorted_filenames(directory):
            # Get a sorted list of all file names in the given directory
            return sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])
        
        self.image_filenames = get_sorted_filenames(os.path.join(self.root_path, 'stereo_image'))
        self.frame_sync = len(self.image_filenames)
        
        self.RAD_filenames = get_sorted_filenames(os.path.join(self.root_path, 'RAD'))

        self.anno_filenames = get_sorted_filenames(os.path.join(self.root_path, 'gt'))

    def get_image(self, idx=None, for_visualize=False): 
        #image = np.asarray(Image.open(self.image_filenames[idx]))
        return self.image_filenames[idx]
    
    def get_RAD(self, idx=None, for_visualize=False):
        rad = np.load(self.RAD_filenames[idx])
        return rad
    
    def get_RD(self, idx=None, for_visualize=False):
        rad = self.get_RAD(idx)
        rad = np.abs(rad)
        rad = pow(rad, 2)
        rd = np.sum(rad, axis=1)
        rd = 10 * np.log10(rd + 1.)
        return rd
    
    def get_RA(self, idx=None, for_visualize=False):
        rad = self.get_RAD(idx)
        rad = np.abs(rad)
        rad = pow(rad, 2)
        ra = np.sum(rad, axis=-1)
        ra = 10 * np.log10(ra + 1.)
        return ra

    def get_radarpointcloud(self, idx=None, for_visualize=False):
        return None

    def get_lidarpointcloud(self, idx=None, for_visualize=False):
        return None

    def get_depthimage(self, idx=None, for_visualize=False):
        return None
      
    def get_spectrogram(self, idx=None, for_visualize=False):
        return None
    
    def get_feature(self, feature_name, idx=None, for_visualize=False):
        function_dict = {
            'RAD': self.get_RAD,
            'RD': self.get_RD,
            'RA': self.get_RA,
            'spectrogram': self.get_spectrogram,
            'radarPC': self.get_radarpointcloud,
            'lidarPC': self.get_lidarpointcloud,
            'image': self.get_image,
            'depth_image': self.get_depthimage,
        }
        feature_data = function_dict[feature_name](idx, for_visualize=for_visualize)
        return feature_data
    
    def get_label(self, feature_name, idx=None):
        gt = []
        with open(self.anno_filenames[idx], "rb") as f:
            objs = pickle.load(f)
        for i in range(len(objs["classes"])):
            bbox3d = objs["boxes"][i]
            cls = objs["classes"][i]
            cart_box = objs["cart_boxes"][i]
            if feature_name == "RD":
                y_c, x_c, h, w = (bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5])
                y1, y2, x1, x2 = int(y_c-h/2), int(y_c+h/2), int(x_c-w/2), int(x_c+w/2)
                gt.append([x1, y1, x2, y2, cls])
                print("####################### RD label ######################")
                print([x1, y1, x2, y2, cls])
            elif feature_name == "RA":
                y_c, x_c, h, w = (bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4])
                y1, y2, x1, x2 = int(y_c-h/2), int(y_c+h/2), int(x_c-w/2), int(x_c+w/2)
                gt.append([x1, y1, x2, y2, cls])
                print("####################### RA label ######################")
                print([x1, y1, x2, y2, cls])
            #cart_box = np.array([cart_box])   
        return gt
    
    def prepare_for_train(self, features, train_cfg, model_cfg, splittype=None):
        self.data_stats = {
            "all_classes" : ["person", "bicycle", "car", "motorcycle", "bus", "truck" ],
            "global_mean_log" : 3.2438383,
            "global_max_log" : 10.0805629,
            "global_min_log" : 0.0,
            "global_variance_log" : 6.8367246,
            "max_boxes_per_frame" : 30,
            "trainset_portion" : 0.8
        }
        #assert model_cfg["n_class"] == len(self.data_stats["all_classes"])
        if 'input_size' in model_cfg:
            self.input_shape = model_cfg["input_size"]
        elif 'input_size' in train_cfg:
            self.input_shape = train_cfg["input_size"]
        self.features = features

        if 'transformations' in train_cfg:
            self.transformations = train_cfg['transformations']

        if model_cfg["class"] == "RADDet":
            self.model_type = "RADDet"
            self.anchor_boxes = np.array(model_cfg["anchor_boxes"])
            self.headoutput_shape = [3,16,16,4,78]
            self.grid_strides = np.array(self.input_shape[:3]) / np.array(self.headoutput_shape[1:4])
        else:
            self.model_type = model_cfg["class"]
        return 
    
    def transform(self, frame, gt_boxes, is_vflip=False, is_hflip=False):
        func_dict = {
            'vflip': flip_vertical,
            'hflip': flip_horizontal
        }
        if self.transformations is not None:
            for transform in self.transformations:
                frame, gt_boxes = func_dict[transform](frame, gt_boxes)
        return frame, gt_boxes
    
    def __len__(self):
        return self.frame_sync

    def __getitem__(self, index):
        RAD_complex = self.get_RAD(index)
        # Gloabl Normalization
        RAD_data = complexTo2Channels(RAD_complex)
        RAD_data = (RAD_data - self.data_stats["global_mean_log"]) / self.data_stats["global_variance_log"]
        
        with open(self.anno_filenames[index], "rb") as f:
            gt_instances = pickle.load(f)

        if self.model_type == "RADDet":
            # decode ground truth boxes to YOLO format
            gt_labels, has_label, gt_boxes = self.encodeToLabels(gt_instances)
            feature_data = RAD_data
            feature_data = np.transpose(feature_data, (2, 0, 1))
            feature_name = "RAD"
        elif self.model_type == "DAROD":
            if self.features == ["RD"]:
                feature_data = self.get_RD(index)
                feature_name = "RD"
            elif self.features == ["RA"]:
                feature_data = self.get_RA(index)
                feature_name = "RA"

            x_shape, y_shape = feature_data.shape[1], feature_data.shape[0]
            boxes = gt_instances["boxes"]
            classes = gt_instances["classes"]
            gt_boxes = []
            gt_labels = []
            for (box, class_) in zip(boxes, classes):
                yc, xc, h, w = box[0], box[2], box[3], box[5]
                y1, y2, x1, x2 = int(yc - h / 2), int(yc + h / 2), int(xc - w / 2), int(xc + w / 2)
                if x1 < 0:
                    # Create 2 boxes
                    x1 += x_shape
                    box1 = [y1 / y_shape, x1 / x_shape, y2 / y_shape, x_shape / x_shape]
                    box2 = [y1 / y_shape, 0 / x_shape, y2 / y_shape, x2 / x_shape]
                    #
                    gt_boxes.append(box1)
                    gt_labels.append(class_)
                    #
                    gt_boxes.append(box2)
                    gt_labels.append(class_)
                elif x2 >= x_shape:
                    x2 -= x_shape
                    box1 = [y1 / y_shape, x1 / x_shape, y2 / y_shape, x_shape / x_shape]
                    box2 = [y1 / y_shape, 0 / x_shape, y2 / y_shape, x2 / x_shape]
                    #
                    gt_boxes.append(box1)
                    gt_labels.append(class_)
                    #
                    gt_boxes.append(box2)
                    gt_labels.append(class_)
                else:
                    gt_boxes.append([y1 / y_shape, x1 / x_shape, y2 / y_shape, x2 / x_shape])
                    gt_labels.append(class_)
            
            gt_labels = [self.data_stats["all_classes"].index(class_name)+1 for class_name in gt_labels] # Plus 1: map to 1~6
            gt_labels = np.array(gt_labels)
            gt_boxes = np.array(gt_boxes)
            feature_data, gt_boxes = self.transform(feature_data, gt_boxes) 
            feature_data = np.expand_dims(feature_data, axis=0)  
        else:
            raise ValueError("Model type not supported")    
        return {feature_name: feature_data, 'label': gt_labels, 'boxes': gt_boxes, 'image_path': self.image_filenames[index]}
    
    def encodeToLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xyzwhd = np.zeros((self.data_stats["max_boxes_per_frame"], 7))
        gt_labels = np.zeros(list(self.headoutput_shape[1:4]) + \
                        [len(self.anchor_boxes)] + \
                        [len(self.data_stats["all_classes"]) + 7]) # (16, 16, 4, 6, 13)

        ### start transferring box to ground truth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.data_stats["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xyzwhd = gt_instances["boxes"][i]
            class_id = self.data_stats["all_classes"].index(class_name)
            if i < self.data_stats["max_boxes_per_frame"]:
                raw_boxes_xyzwhd[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd[i, 6] = class_id
            class_onehot = smoothOnehot(class_id, len(self.data_stats["all_classes"]))
            
            exist_positive = False

            grid_strid = self.grid_strides
            anchor_stage = self.anchor_boxes
            box_xyzwhd_scaled = box_xyzwhd[np.newaxis, :].astype(np.float32)
            box_xyzwhd_scaled[:, :3] /= grid_strid
            anchorstage_xyzwhd = np.zeros([len(anchor_stage), 6])
            anchorstage_xyzwhd[:, :3] = np.floor(box_xyzwhd_scaled[:, :3]) + 0.5
            anchorstage_xyzwhd[:, 3:] = anchor_stage.astype(np.float32)

            iou_scaled = iou3d(box_xyzwhd_scaled, anchorstage_xyzwhd, \
                                        self.input_shape)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]).\
                                    astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, zind, iou_mask, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, iou_mask, 6:7] = 1.
                gt_labels[xind, yind, zind, iou_mask, 7:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                anchor_ind = np.argmax(iou_scaled)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]).\
                                    astype(np.int32)
                gt_labels[xind, yind, zind, anchor_ind, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, anchor_ind, 6:7] = 1.
                gt_labels[xind, yind, zind, anchor_ind, 7:] = class_onehot

        has_label = False
        for label_stage in gt_labels:
            if label_stage.max() != 0:
                has_label = True
        gt_labels = np.array([np.where(gt_i == 0, 1e-16, gt_i) for gt_i in gt_labels])
        return gt_labels, has_label, raw_boxes_xyzwhd
    

class Astyx(Dataset):
    name = "Astyx 2019 dataset instance"

    def __init__(self, features=None):
        self.feature_path = ""
        self.config = ""

        self.frame_sync = 0
        self.features = ['image', 'radarPC', 'lidarPC']
        
    def parse(self, file_id, file_path, file_name, config):
        self.config = config
        self.root_path = file_path
        
        def get_sorted_filenames(directory):
            # Get a sorted list of all file names in the given directory
            return sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])
        
        self.image_filenames = get_sorted_filenames(os.path.join(self.root_path, 'camera_front'))
        self.frame_sync = len(self.image_filenames)
        self.radarPC_filenames = get_sorted_filenames(os.path.join(self.root_path, 'radar_6455'))
        self.lidarPC_filenames = get_sorted_filenames(os.path.join(self.root_path, 'lidar_vlp16'))


    def get_image(self, idx=None, for_visualize=False): 
        #image = np.asarray(Image.open(self.image_filenames[idx]))
        return self.image_filenames[idx]
    
    def get_RAD(self, idx=None, for_visualize=False):
        return None
    
    def get_RD(self, idx=None, for_visualize=False):
        return None
    
    def get_RA(self, idx=None, for_visualize=False):
        return None

    def get_radarpointcloud(self, idx=None, for_visualize=False):
        pc = read_pointcloudfile(self.radarPC_filenames[idx])
        return pc

    def get_lidarpointcloud(self, idx=None, for_visualize=False):
        pc = read_pointcloudfile(self.lidarPC_filenames[idx])
        return pc

    def get_depthimage(self, idx=None, for_visualize=False):
        return None
      
    def get_spectrogram(self, idx=None, for_visualize=False):
        return None
    
    def get_feature(self, feature_name, idx=None, for_visualize=False):
        function_dict = {
            'RAD': self.get_RAD,
            'RD': self.get_RD,
            'RA': self.get_RA,
            'spectrogram': self.get_spectrogram,
            'radarPC': self.get_radarpointcloud,
            'lidarPC': self.get_lidarpointcloud,
            'image': self.get_image,
            'depth_image': self.get_depthimage,
        }
        feature_data = function_dict[feature_name](idx, for_visualize=for_visualize)
        return feature_data
    
    def get_label(self, feature_name, idx=None):
        file_name = "{:06d}".format(idx) + ".json"

        #T_toLidar, T_toCamera, K = get_calibration(filename)
        with open(os.path.join(self.root_path, 'calibration', file_name), mode='r') as f:
            calib_data = json.load(f)
        
        T_fromLidar = np.array(calib_data['sensors'][1]['calib_data']['T_to_ref_COS'])
        T_fromCamera = np.array(calib_data['sensors'][2]['calib_data']['T_to_ref_COS'])
        K = np.array(calib_data['sensors'][2]['calib_data']['K'])

        T_toLidar = inv_trans(T_fromLidar)
        T_toCamera = inv_trans(T_fromCamera)

        #objects, classids = get_objects(filename)
        gt = []
        with open(os.path.join(self.root_path, 'groundtruth_obj3d', file_name), mode='r') as f:
            gt_data = json.load(f)
        objects_info = gt_data['objects']
        objects = []
        classids = []
        for p in objects_info:
            center = np.array(p['center3d'])
            dimension = np.array(p['dimension3d'])
            w = dimension[0]
            l = dimension[1]
            h = dimension[2]
            orientation = np.array(p['orientation_quat'])
            classids.append(p['classname'])

            x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
            # rotate and translate 3d bounding box
            R = quat_to_rotation(orientation)
            # ##########################
            # yaw = qaut_to_angle(orientation)
            # rotMat = np.array([
            #     [np.cos(yaw), -np.sin(yaw), 0.0],
            #     [np.sin(yaw), np.cos(yaw), 0.0],
            #     [0.0, 0.0, 1.0]])
            # ##########################
            # case 1: rotate + translate
            bbox = np.vstack([x_corners, y_corners, z_corners])
            bbox = np.dot(R, bbox)
            bbox = bbox + center[:, np.newaxis]

            # case 2: translate + rotate
            # bbox = np.vstack([x_corners, y_corners, z_corners]) + center[:,np.newaxis]
            # bbox = np.dot(R, bbox)

            bbox = np.transpose(bbox)
            objects.append(bbox)

        if feature_name == "radarPC":
            for obj, cls in zip(objects, classids):
                gt.append(obj.flatten().tolist() + [cls])

        if feature_name == "lidarPC":
            objects_lidar = []
            for obj, cls in zip(objects, classids):
                obj_lidar = np.dot(T_toLidar[0:3, 0:3], np.transpose(obj))
                T = T_toLidar[0:3, 3]
                obj_lidar = obj_lidar + T[:, np.newaxis]
                obj_lidar = np.transpose(obj_lidar)
                objects_lidar.append(obj_lidar)
                gt.append(obj_lidar.flatten().tolist() + [cls])

        if feature_name == "image":
            objects_2Dimage = []
            for obj, cls in zip(objects, classids):
                obj_camera = np.dot(T_toCamera[0:3, 0:3], np.transpose(obj))
                T = T_toCamera[0:3, 3]
                obj_camera = obj_camera + T[:, np.newaxis]
                obj_image = np.dot(K, obj_camera)
                obj_image = obj_image / obj_image[2, :]
                obj_image = np.delete(obj_image, 2, 0)
                objects_2Dimage.append(obj_image)
                gt.append(obj_image.flatten().tolist() + [cls])
        return gt
    
    def prepare_for_train(self, features, train_cfg, model_cfg, splittype=None):
        return 

    def __len__(self):
        return self.frame_sync

    def __getitem__(self, index):
        
       return None


class UWCR(Dataset):
    name = "UWCR dataset instance"

    def __init__(self, features=None):
        self.feature_path = ""
        self.config = ""

        self.frame_sync = 0
        self.features = ['image', 'adc']
        
    def parse(self, file_id, file_path, file_name, config):
        self.config = config
        self.root_path = file_path
        self.seq_name = file_name
        
        def get_sorted_filenames(directory):
            # Get a sorted list of all file names in the given directory
            return sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])
        
        self.image_filenames = get_sorted_filenames(os.path.join(self.root_path, file_name, 'images_0'))
        self.label_filenames = get_sorted_filenames(os.path.join(self.root_path, file_name, 'text_labels'))
        self.radar_filenames = get_sorted_filenames(os.path.join(self.root_path, file_name, 'radar_raw_frame'))
        
        # find the overlapped indices in the sequence
        image_filename = os.path.basename(self.image_filenames[0]) 
        image_filename_id = int(image_filename.split('.')[0]) 
        radar_filename = os.path.basename(self.radar_filenames[0])
        radar_filename_id = int(radar_filename.split('.')[0])
        label_filename = os.path.basename(self.label_filenames[0])
        label_filename_id = int(label_filename.split('.')[0])
        if image_filename_id != radar_filename_id or radar_filename_id != label_filename_id or image_filename_id != label_filename_id:
            id_max = max(image_filename_id, radar_filename_id, label_filename_id)
            if image_filename_id != id_max:
                self.image_filenames = self.image_filenames[id_max - image_filename_id:]
            if radar_filename_id != id_max:
                self.radar_filenames = self.radar_filenames[id_max - radar_filename_id:]
            if label_filename_id != id_max:
                self.label_filenames = self.label_filenames[id_max - label_filename_id:]
        image_filename = os.path.basename(self.image_filenames[-1]) 
        image_filename_id = int(image_filename.split('.')[0]) 
        radar_filename = os.path.basename(self.radar_filenames[-1])
        radar_filename_id = int(radar_filename.split('.')[0])
        label_filename = os.path.basename(self.label_filenames[-1])
        label_filename_id = int(label_filename.split('.')[0])
        if image_filename_id != radar_filename_id or radar_filename_id != label_filename_id or image_filename_id != label_filename_id:
            id_min = min(image_filename_id, radar_filename_id, label_filename_id)
            if image_filename_id != id_min:
                self.image_filenames = self.image_filenames[:id_min - id_max + 1]
            if radar_filename_id != id_min:
                self.radar_filenames = self.radar_filenames[:id_min - id_max + 1]
            if label_filename_id != id_min:
                self.label_filenames = self.label_filenames[:id_min - id_max + 1]

        self.frame_sync = len(self.image_filenames) 
        with open(self.config, 'r') as f:
            self.radar_cfg = json.load(f)
        self.n_angle = 128
        self.n_vel = 128
        self.n_range = 128
        self.n_chirp = 255
        self.n_rx = 8
        self.n_sample = 128
        self.label_map = {0: 'pedestrian',
                        2: 'car',
                        3: 'motorbike',
                        5: 'bus',
                        7: 'truck',
                        80: 'cyclist'}
        return 

    def get_image(self, idx=None, for_visualize=False): 
        #image = np.asarray(Image.open(self.image_filenames[idx]))
        return self.image_filenames[idx]
    
    def get_RAD(self, idx=None, for_visualize=False):
        return None
    
    def get_ADC(self, idx=None, for_visualize=False):
        mat = spio.loadmat(self.radar_filenames[idx], squeeze_me=True)
        adc = np.asarray(mat["adcData"]) #  (128, 255, 4, 2)
        adc = np.concatenate((adc[:, :, :, 0], adc[:, :, :, 1]), axis=2) 
        adc = np.swapaxes(adc, 1, 2) # (128, 8, 255): samples, antennas, chirps
        return adc
    
    def get_RD(self, idx=None, for_visualize=False):
        data = self.get_ADC(idx)
        # range fft
        hanning_win = np.hamming(self.n_sample)
        win_data = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=np.complex128)
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                win_data[:, i, j] = np.multiply(data[:, i, j], hanning_win)
        rd = np.fft.fft(win_data, self.n_range, axis=0) # (128, 8, 255)
        
        if for_visualize:
            rd = np.abs(rd[:, 0, :])
        return rd

    def get_RV_VA_slice(self, idx=None, for_visualize=False):
        data = self.get_RD(idx)
        # RV slice
        hanning_win = np.hamming(self.n_vel)
        win_data1 = np.zeros([data.shape[0], data.shape[1], self.n_vel], dtype=np.complex128)
        win_data2 = np.zeros([data.shape[0], data.shape[1], self.n_vel], dtype=np.complex128)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                win_data1[i, j, :] = np.multiply(data[i, j, 0:self.n_vel], hanning_win)
                win_data2[i, j, :] = np.multiply(data[i, j, self.n_vel - 1:], hanning_win)

        fft_data_raw1 = np.fft.fft(win_data1, self.n_vel, axis=2)
        fft_data_raw1 = np.fft.fftshift(fft_data_raw1, axes=2)
        fft3d_data1 = np.sum(np.abs(fft_data_raw1), axis=1) / self.n_rx
        fft3d_data1 = np.expand_dims(fft3d_data1, axis=2)

        fft_data_raw2 = np.fft.fft(win_data2, self.n_vel, axis=2)
        fft_data_raw2 = np.fft.fftshift(fft_data_raw2, axes=2)
        fft3d_data2 = np.sum(np.abs(fft_data_raw2), axis=1) / self.n_rx
        fft3d_data2 = np.expand_dims(fft3d_data2, axis=2)

        # output format [range, velocity, 2chirps] : (128, 128, 2)
        rv = np.float32(np.concatenate((fft3d_data1, fft3d_data2), axis=2))

        # VA slice
        rv_raw1 = fft_data_raw1
        rv_raw2 = fft_data_raw2
        hanning_win = np.hamming(self.n_rx)
        win_data1 = np.zeros([rv_raw1.shape[0], rv_raw1.shape[1], rv_raw1.shape[2]], dtype=np.complex128)
        win_data2 = np.zeros([rv_raw2.shape[0], rv_raw2.shape[1], rv_raw2.shape[2]], dtype=np.complex128)
        for i in range(rv_raw1.shape[0]):
            for j in range(rv_raw1.shape[2]):
                win_data1[i, :, j] = np.multiply(rv_raw1[i, :, j], hanning_win)
                win_data2[i, :, j] = np.multiply(rv_raw2[i, :, j], hanning_win)

        fft_data_raw1 = np.fft.fft(win_data1, self.n_angle, axis=1)
        fft3d_data1 = np.sum(np.abs(np.fft.fftshift(fft_data_raw1, axes=1)), axis=0) / rv_raw1.shape[0]
        fft3d_data1 = np.expand_dims(fft3d_data1, axis=2)

        fft_data_raw2 = np.fft.fft(win_data2, self.n_angle, axis=1)
        fft3d_data2 = np.sum(np.abs(np.fft.fftshift(fft_data_raw2, axes=1)), axis=0) / rv_raw2.shape[0]
        fft3d_data2 = np.expand_dims(fft3d_data2, axis=2)

        # output format [angle, velocity, 2chirps] : (128, 128, 2)
        va = np.float32(np.concatenate((fft3d_data1, fft3d_data2), axis=2))
        return rv, va
    
    def get_RA(self, idx=None, for_visualize=False):
        data = self.get_RD(idx)
        hanning_win = np.hamming(self.n_rx)
        win_data = np.zeros([data.shape[0], data.shape[1], data.shape[2]], dtype=np.complex128)
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                win_data[i, :, j] = np.multiply(data[i, :, j], hanning_win)

        fft_data_raw = np.fft.fft(win_data, self.n_angle, axis=1)
        fft3d_data_cmplx = np.fft.fftshift(fft_data_raw, axes=1)

        if for_visualize:
            ra = np.abs(fft3d_data_cmplx[:, :, 0])
        else:
            filter_static =  False
            keep_complex = False
            if keep_complex:
                ra = fft3d_data_cmplx
            else:
                fft_data_real = np.expand_dims(fft3d_data_cmplx.real, axis=3)
                fft_data_imag = np.expand_dims(fft3d_data_cmplx.imag, axis=3)
                # output format [range, angle, chirps, real/imag] : (128, 128, 255, 2)
                ra = np.float32(np.concatenate((fft_data_real, fft_data_imag), axis=3))
            
            if filter_static:
                ra = ra - np.mean(ra, axis=2, keepdims=True)
        return ra

    def get_radarpointcloud(self, idx=None, for_visualize=False):
        return None

    def get_lidarpointcloud(self, idx=None, for_visualize=False):
        return None

    def get_depthimage(self, idx=None, for_visualize=False):
        return None
      
    def get_spectrogram(self, idx=None, for_visualize=False):
        return None
    
    def get_feature(self, feature_name, idx=None, for_visualize=False):
        function_dict = {
            'RAD': self.get_RAD,
            'RD': self.get_RD,
            'RA': self.get_RA,
            'spectrogram': self.get_spectrogram,
            'radarPC': self.get_radarpointcloud,
            'lidarPC': self.get_lidarpointcloud,
            'image': self.get_image,
            'depth_image': self.get_depthimage,
        }
        feature_data = function_dict[feature_name](idx, for_visualize=for_visualize)
        return feature_data
    
    def get_label(self, feature_name, idx=None):
        filename = self.label_filenames[idx]
        try:
            labels = pd.read_csv(filename)
        except pd.errors.EmptyDataError:
            return []

        gt = []
        for _, obj in labels.iterrows():
            category = self.label_map[obj[1]]
            center_x, center_y, width, length = obj[2:]

            if feature_name == "RA":
                bbox = [center_x - width / 2, center_y - length / 2, 
                        center_x + width / 2, center_y + length / 2, 
                        category]
                gt.append(bbox)
        return gt
    
    def prepare_for_train(self, features, train_cfg, model_cfg, splittype=None):
        self.n_class = 3
        self.class_ids = {'pedestrian': 0, 'cyclist': 1, 'car': 2, 'truck': 2}
        self.win_size = train_cfg['win_size'] 
        if splittype in ('train', 'val') or splittype is None:
            self.step = train_cfg['train_step']
            self.stride = train_cfg['train_stride']
        else:
            self.step = train_cfg['test_step']
            self.stride = train_cfg['test_stride']

        n_frame = self.frame_sync
        n_data_in_seq = (n_frame - (self.win_size * self.step - 1)) // self.stride + (
            1 if (n_frame - (self.win_size * self.step - 1)) % self.stride > 0 else 0)
        self.datasamples_length = n_data_in_seq
        self.rng_grid = confmap2ra(self.radar_cfg, name='range')
        self.agl_grid = confmap2ra(self.radar_cfg, name='angle')
        self.noise_channel = False
        return 

    def __len__(self):
        return self.datasamples_length

    def __getitem__(self, index):
        logger.info(f"Data item index: {index}")
        data_id = index * self.stride
        radar_npy_win_ra = np.zeros((self.win_size * 2, self.radar_cfg['ramap_rsize'], self.radar_cfg['ramap_asize'], 2), dtype=np.float32)
        radar_npy_win_rv = np.zeros((self.win_size * 2, self.radar_cfg['ramap_rsize'], self.radar_cfg['ramap_vsize'], 1), dtype=np.float32)
        radar_npy_win_va = np.zeros((self.win_size * 2, self.radar_cfg['ramap_asize'], self.radar_cfg['ramap_vsize'], 1), dtype=np.float32)
        confmap_gt = np.zeros((self.win_size, self.n_class + 1, self.radar_cfg['ramap_asize'], self.radar_cfg['ramap_vsize']), dtype=np.float32)
        obj_info = []
        for idx, frameid in enumerate(range(data_id, data_id + self.win_size * self.step, self.step)):
            # load ra slice
            # format of radar_npy_win_ra [chirp, range, angle, real/imag]
            ra = self.get_RA(idx)
            radar_npy_win_ra[idx * 2, :, :, :] = ra[:, :, 0, :]
            radar_npy_win_ra[idx * 2 + 1, :, :, :] = ra[:, :, 128, :]
            # load rv slice
            # format of radar_npy_win_rv [chirp, range, velocity, real]
            rv, va = self.get_RV_VA_slice(idx)
            radar_npy_win_rv[idx * 2, :, :, 0] = rv[:, :, 0]
            radar_npy_win_rv[idx * 2 + 1, :, :, 0] = rv[:, :, 1]
            # load va slice
            # format of radar_npy_win_rv [chirp, angle, velocity, real]
            radar_npy_win_va[idx * 2, :, :, 0] = va[:, :, 0]
            radar_npy_win_va[idx * 2 + 1, :, :, 0] = va[:, :, 1]
            # label file: [uid, class, px, py, wid, len]
            obj_in_frame = []
            try:
                labels = pd.read_csv(self.label_filenames[frameid], header=None).values.tolist()
                n_obj = len(labels)
                for obj in labels:
                    category = self.label_map[int(obj[1])]
                    # class_id = self.class_ids[category]
                    x = obj[2]
                    y = obj[3]
                    distance = math.sqrt(x ** 2 + y ** 2)
                    angle = math.degrees(math.atan(x / y))  # in degree
                    if distance > self.radar_cfg['rr_max'] or distance < self.radar_cfg['rr_min']:
                        continue
                    if angle > self.radar_cfg['ra_max'] or angle < self.radar_cfg['ra_min']:
                        continue
                    rng_idx, _ = find_nearest(self.rng_grid, distance)
                    agl_idx, _ = find_nearest(self.agl_grid, angle)
                    obj_in_frame.append([rng_idx, agl_idx, category])
                confmap_gt_in_frame = generate_confmap(n_obj, obj_in_frame, self.radar_cfg)
                confmap_gt_in_frame = normalize_confmap(confmap_gt_in_frame)
                confmap_gt_in_frame = add_noise_channel(confmap_gt_in_frame, self.radar_cfg)
                assert confmap_gt_in_frame.shape == (self.n_class + 1, self.radar_cfg['ramap_rsize'], self.radar_cfg['ramap_asize'])
            except pd.errors.EmptyDataError:
                confmap_gt_in_frame = np.zeros((self.n_class + 1, self.radar_cfg['ramap_rsize'], self.radar_cfg['ramap_asize']), dtype=float)
                confmap_gt_in_frame[-1, :, :] = 1.0  # initialize noise channal
                
            confmap_gt[idx, :, :, :] = confmap_gt_in_frame
            obj_info.append(obj_in_frame)
            
        radar_npy_win_ra = np.transpose(radar_npy_win_ra, (3, 0, 1, 2))
        radar_npy_win_rv = np.transpose(radar_npy_win_rv, (3, 0, 1, 2))
        radar_npy_win_va = np.transpose(radar_npy_win_va, (3, 0, 1, 2))
        assert radar_npy_win_ra.shape == (2, self.win_size * 2, self.radar_cfg['ramap_rsize'], self.radar_cfg['ramap_asize'])
        assert radar_npy_win_rv.shape == (1, self.win_size * 2, self.radar_cfg['ramap_rsize'], self.radar_cfg['ramap_vsize'])
        assert radar_npy_win_va.shape == (1, self.win_size * 2, self.radar_cfg['ramap_asize'], self.radar_cfg['ramap_vsize'])

        confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
        if self.noise_channel:
            assert confmap_gt.shape == \
                    (self.n_class + 1, self.win_size, self.radar_cfg['ramap_rsize'], self.radar_cfg['ramap_asize'])
        else:
            confmap_gt = confmap_gt[:self.n_class]
            assert confmap_gt.shape == \
                    (self.n_class, self.win_size, self.radar_cfg['ramap_rsize'], self.radar_cfg['ramap_asize'])
        
        assert len(obj_info) == self.win_size
        data_dict = {'RA': radar_npy_win_ra, 'RD': radar_npy_win_rv, 'AD': radar_npy_win_va, \
                     'confmap_gt': confmap_gt, 'gt_label': obj_info, \
                     'seq_name': self.seq_name,  'start_frame': data_id, 'end_frame': data_id + self.win_size * self.step - 1}
        return data_dict