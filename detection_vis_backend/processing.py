import rosbag
import os
import logging
import numpy as np

from detection_vis_backend.radarframe import RadarFrame
from detection_vis_backend.utils import read_radar_params, reshape_frame
from scipy import signal

class DatasetFactory:
    _instances = {}
    _singleton_instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._singleton_instance:
            cls._singleton_instance = super().__new__(cls)
        return cls._singleton_instance

    def get_instance(self, class_name):
        if class_name not in self._instances:
            # Fetch the class from globals, create a singleton instance
            cls = globals()[class_name]
            self._instances[class_name] = cls()
        return self._instances[class_name]


class RaDICaL:
    name = ""
    config = ""
    image = []
    depth_image = []
    ADC = []
    RAD = []
    RA = []
    RD = []
    radarpointcloud = []
    spectrogram = []

    image_count = 0
    depthimage_count = 0
    radarframe_count = 0
    frame_sync = 0


    def __init__(self):
        self.name = "RaDICaL dataset instance"

    def parse(self, file_path, file_name, config):
        self.config = config
        file = os.path.join(file_path, file_name)
        try:
            bag = rosbag.Bag(file)
        except rosbag.ROSBagException:
            print(f"No file found at {file}")
        topics_dict = bag.get_type_and_topic_info()[1]

        if "/camera/color/image_raw" in topics_dict:
            for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw']):
                # print(t.secs)
                # print(t.nsecs)
                # print(msg.header.stamp.secs)
                # print(msg.header.stamp.nsecs)
                # print(msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9)
                assert msg.encoding == "rgb8"
                dtype = np.dtype("uint8")  # 8-bit color image
                dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
                image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, 3)  # 3 for RGB
                self.image.append(image)
            self.image_count = len(self.image)

        if "/camera/aligned_depth_to_color/image_raw" in topics_dict:
            for topic, msg, t in bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw']):
                # print(t.secs)
                # print(t.nsecs)
                # print(msg.header.stamp.secs)
                # print(msg.header.stamp.nsecs)
                # print(msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9)
                assert msg.encoding == "16UC1"
                dtype = np.dtype("uint16")  # 16-bit grayscale image
                dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
                image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
                self.depth_image.append(image)
            self.depthimage_count = len(self.depth_image)

        if "/radar_data" in topics_dict:
            for topic, msg, t in bag.read_messages(topics=['/radar_data']):
                #   print(t.secs)
                #   print(t.nsecs  print(t.secs + t.nsecs*1e-9)
                # if(count > 0):
                #   print(t.secs + t.nsecs*1e-9 - last)
                # last = t.secs + t.nsecs*1e-9
                # print("\nProcessing no.", count, "th radar msg:")
                
                #print(len(msg.data))
                arr = np.array(msg.data)
                #print(arr[0])
                complex_arr = reshape_frame(arr,304,4,2,64)
                #print(complex_arr.shape) # (32, 304, 8)
                transformed = np.swapaxes(complex_arr, 1, 2)
                #print(transformed.shape)
                self.ADC.append(transformed)
            self.radarframe_count = len(self.ADC)
        
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
        


    def get_RAD(self):
        return self.RAD
    
    def get_RA(self, idx):
        if not self.RA:
            radar_config = read_radar_params(self.config) 
            rf = RadarFrame(radar_config)
            for x in self.ADC:
                beamformed_range_azimuth = rf.compute_range_azimuth(x) 
                beamformed_range_azimuth = np.log(np.abs(beamformed_range_azimuth))
                self.RA.append(beamformed_range_azimuth)
        return self.RA[idx]


    def get_RD(self, idx):
        if not self.RD:
            radar_config = read_radar_params(self.config)
            rf = RadarFrame(radar_config)
            for i,x in enumerate(self.ADC): 
                rf.raw_cube = x
                range_doppler = rf.range_doppler
                range_doppler = np.transpose(range_doppler)
                range_doppler[np.isinf(range_doppler)] = 0  # replace Inf with zero
                self.RD.append(range_doppler)
        return self.RD[idx]


    def get_radarpointcloud(self, idx):
        for x in self.RA:
            # radar_config = read_radar_params(config)
            # radar_config = read_radar_params("indoor_human_rcs.cfg") # for local test
            # rf = RadarFrame(radar_config)
            y = np.random.rand(30,20)
            self.radarpointcloud.append(y)
        return self.radarpointcloud[idx]

    def get_lidarpointcloud():
        return None
    
    def get_spectrogram(self, idx):
        for x in self.ADC:
            # radar_config = read_radar_params(config)
            # radar_config = read_radar_params("indoor_human_rcs.cfg") # for local test
            # rf = RadarFrame(radar_config)
            #, tfa = signal.stft()
            tfa = np.random.rand(30,20)
            self.spectrogram.append(tfa)
        return self.spectrogram[idx]

    def get_image(self, idx):
        return self.image[idx]
    
    def get_depthimage(self, idx):
        return self.depth_image[idx]

    

class RADIal:
    images = []
    RAD = []
    radar_pointcloud = []
    lidar_pointcloud = []

    def __init__(self):
        self.name = "RADIal dataset instance"

    def parse(config):
        print("")

    def get_RAD():
        print("")
    
    def get_RA():
        print("")

    def get_RD():
        print("")

    def get_radarpointcloud():
        print("")

    def get_lidarpointcloud():
        print("")

    def get_spectrogram():
        print("")




