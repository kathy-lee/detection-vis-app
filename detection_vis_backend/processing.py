import rosbag
import os
import numpy as np

from detection_vis_backend.radarframe import RadarFrame
from detection_vis_backend.utils import read_radar_params, reshape_frame

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
    AD = []
    radarpointcloud = []
    spectrogram = []

    def __init__(self):
        self.name = "RaDICaL dataset instance"

    def parse(self, file_path, file_name, config):
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

        print("#################################")
        print(len(self.image), len(self.depth_image), len(self.ADC))
        bag.close()


    def get_RAD(self):
        return self.RAD
    
    def get_RA(self):
        if not self.RA:
            for x in self.RAD:
                # radar_config = read_radar_params(config)
                radar_config = read_radar_params(self.config) # for local test
                rf = RadarFrame(radar_config)
                beamformed_range_azimuth = rf.compute_range_azimuth(x) 
                beamformed_range_azimuth = np.log(np.abs(beamformed_range_azimuth))
                self.RA.append(beamformed_range_azimuth)
        return self.RA


    def get_RD(self):
        if not self.RD:
            # radar_config = read_radar_params(config)
            radar_config = read_radar_params("indoor_human_rcs.cfg") # for local test
            rf = RadarFrame(radar_config)
            range_doppler = rf.range_doppler
            print(range_doppler.shape, range_doppler[0,0])
            self.RD = range_doppler
        return self.RD

    def get_AD(self):
        for x in self.RAD:
            # radar_config = read_radar_params(config)
            # radar_config = read_radar_params("indoor_human_rcs.cfg") # for local test
            # rf = RadarFrame(radar_config)
            y = np.random.rand(30,20)
            self.AD.append(y)
        return self.AD

    def get_radarpointcloud(self):
        for x in self.RAD:
            # radar_config = read_radar_params(config)
            # radar_config = read_radar_params("indoor_human_rcs.cfg") # for local test
            # rf = RadarFrame(radar_config)
            y = np.random.rand(30,20)
            self.radarpointcloud.append(y)
        return self.radarpointcloud

    def get_lidarpointcloud():
        return None
    
    def get_spectrogram(self):
        for x in self.RAD:
            # radar_config = read_radar_params(config)
            # radar_config = read_radar_params("indoor_human_rcs.cfg") # for local test
            # rf = RadarFrame(radar_config)
            y = np.random.rand(30,20)
            self.spectrogram.append(y)
        return self.spectrogram

    def get_image(self):
        return self.image
    
    def get_depthimage(self):
        return self.depth_image

    

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

    def get_AD():
        print("")

    def get_radarpointcloud():
        print("")

    def get_lidarpointcloud():
        print("")

    def get_spectrogram():
        print("")




