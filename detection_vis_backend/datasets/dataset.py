import rosbag
import os
import logging
import numpy as np
import torchvision.transforms as transform
import pandas as pd


from detection_vis_backend.datasets.radarframe import RadarFrame
from detection_vis_backend.datasets.utils import read_radar_params, reshape_frame
from scipy import signal
from torchvision.transforms import Resize,CenterCrop
from torch.utils.data import Dataset
from PIL import Image



class DatasetFactory:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(DatasetFactory, cls).__new__(cls, *args, **kwargs)
            cls._instance.instance_dict = {}  # this dictionary stores the instances
        return cls._instance

    def get_instance(self, class_name, id):
        if id in self.instance_dict:
            return self.instance_dict[id]
        
        class_obj = globals()[class_name]()
        self.instance_dict[id] = class_obj
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


    def __init__(self, features=None):
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
        

    def get_RAD(self, idx=None):
        return self.RAD if idx else self.RAD
    
    def get_RA(self, idx=None):
        if not self.RA:
            radar_config = read_radar_params(self.config) 
            rf = RadarFrame(radar_config)
            for x in self.ADC:
                beamformed_range_azimuth = rf.compute_range_azimuth(x) 
                beamformed_range_azimuth = np.log(np.abs(beamformed_range_azimuth))
                self.RA.append(beamformed_range_azimuth)
        return self.RA[idx] if idx else self.RA

    def get_RD(self, idx=None):
        if not self.RD:
            radar_config = read_radar_params(self.config)
            rf = RadarFrame(radar_config)
            for i,x in enumerate(self.ADC): 
                rf.raw_cube = x
                range_doppler = rf.range_doppler
                range_doppler = np.transpose(range_doppler)
                range_doppler[np.isinf(range_doppler)] = 0  # replace Inf with zero
                self.RD.append(range_doppler)
        return self.RD[idx] if idx else self.RD

    def get_radarpointcloud(self, idx=None):
        for x in self.RA:
            # radar_config = read_radar_params(config)
            # radar_config = read_radar_params("indoor_human_rcs.cfg") # for local test
            # rf = RadarFrame(radar_config)
            y = np.random.rand(30,20)
            self.radarpointcloud.append(y)
        return self.radarpointcloud[idx] if idx else self.radarpointcloud

    def get_lidarpointcloud():
        return None
    
    def get_spectrogram(self, idx=None):
        for x in self.ADC:
            # radar_config = read_radar_params(config)
            # radar_config = read_radar_params("indoor_human_rcs.cfg") # for local test
            # rf = RadarFrame(radar_config)
            #, tfa = signal.stft()
            tfa = np.random.rand(30,20)
            self.spectrogram.append(tfa)
        return self.spectrogram[idx] if idx else self.spectrogram

    def get_image(self, idx=None):
        return self.image[idx] if idx else self.image
    
    def get_depthimage(self, idx=None):
        return self.depth_image[idx] if idx else self.depth_image

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
    image = []
    RAD = []
    RA = []
    RD = []
    radar_pointcloud = []
    lidar_pointcloud = []

    image_count = 8252
    lidarframe_count = 8252
    radarframe_count = 8252
    frame_sync = 8252

    def parse(self, file_path, file_name, config, difficult=True):
        def get_sorted_filenames(directory):
            # Get a sorted list of all file names in the given directory
            return sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])
        
        self.image_filenames = get_sorted_filenames(os.path.join(file_path, file_name, 'camera'))
        self.lidarpointcloud_filenames = get_sorted_filenames(os.path.join(file_path, file_name, 'laser_PCL'))
        self.RD_filenames = get_sorted_filenames(os.path.join(file_path, file_name, 'radar_FFT'))
        self.radarpointcloud_filenames = get_sorted_filenames(os.path.join(file_path, file_name, 'radar_PCL'))

        self.labels = pd.read_csv(os.path.join(file_path, file_name, 'labels.csv')).to_numpy()

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
        if idx is not None:
            return np.load(self.RD_filenames[idx])
        else:
            for i in self.RD_filenames:
                rangedoppler = np.load(self.RD_filenames[i])
                self.RD.append(rangedoppler)
            return self.RD

    def get_radarpointcloud(self, idx=None):
        if idx is not None:
            return np.load(self.radarpointcloud_filenames[idx])
        else:
            for i in self.radarpointcloud_filenames:
                pc = np.load(self.radarpointcloud_filenames[i])
                self.radar_pointcloud.append(pc)
            return self.radar_pointcloud

    def get_lidarpointcloud(self, idx=None):
        if idx is not None:
            return np.load(self.lidarpointcloud_filenames[idx], allow_pickle=True)[:,:3]
        else:
            for i in self.lidarpointcloud_filenames:
                pc = np.load(self.lidarpointcloud_filenames[i], allow_pickle=True)[:,:3]
                self.lidar_pointcloud.append(pc)
            return self.lidar_pointcloud

    def get_image(self, idx=None): 
        if idx is not None:
            image = np.asarray(Image.open(self.image_filenames[idx]))
            return image
        else:
            for i in self.image_filenames:
                image = np.asarray(Image.open(self.image_filenames[i]))
                self.image.append(image)
            return self.image

    def get_depthimage(self, idx=None):
        return None
      
    def get_spectrogram(self, idx=None):
        return None

    def __init__(self):
        # RADIal data has two formats: raw and ready-to-use
        self.name = "RADIal dataset instance" 

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

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix]
        box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 


        ######################
        #  Encode the labels #
        ######################
        out_label=[]
        if(self.encoder!=None):
            out_label = self.encoder(box_labels).copy()      

        # Read the Radar FFT data
        radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npy".format(sample_id))
        input = np.load(radar_name,allow_pickle=True)
        radar_FFT = np.concatenate([input.real,input.imag],axis=2)
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
        img_name = os.path.join(self.root_dir,'camera',"image_{:06d}.jpg".format(sample_id))
        image = np.asarray(Image.open(img_name))

        return radar_FFT, segmap,out_label,box_labels,image
