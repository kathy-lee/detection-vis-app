import rosbag
import os
import logging
import numpy as np
import torchvision.transforms as transform
import pandas as pd
from pathlib import Path


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
    
    # image = []
    # depth_image = []
    # ADC = []
    # RAD = []
    # RA = []
    # RD = []
    # radarpointcloud = []
    # spectrogram = []

    image_count = 0
    depthimage_count = 0
    radarframe_count = 0
    frame_sync = 0


    def __init__(self, features=None):
        self.name = "RaDICaL dataset instance"

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
        return
        

    def get_RAD(self, idx=None):
        return None
    
    def get_ADC(self, idx=None):
        adc_file = os.path.join(self.feature_path,'adc',f"adc_{idx}.npy")
        adc = np.load(adc_file)
        return adc
    
    def get_RA(self, idx=None):
        radar_config = read_radar_params(self.config) 
        rf = RadarFrame(radar_config)
        adc = self.get_ADC(idx)
        beamformed_range_azimuth = rf.compute_range_azimuth(adc) 
        beamformed_range_azimuth = np.log(np.abs(beamformed_range_azimuth))        
        return beamformed_range_azimuth

    def get_RD(self, idx=None):
        radar_config = read_radar_params(self.config)
        rf = RadarFrame(radar_config)
        rf.raw_cube = self.get_ADC(idx)
        range_doppler = rf.range_doppler
        range_doppler = np.transpose(range_doppler)
        range_doppler[np.isinf(range_doppler)] = 0  # replace Inf with zero
        return range_doppler

    def get_radarpointcloud(self, idx=None):
        return None

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
        if idx is not None:
            rd = np.load(self.RD_filenames[idx])
            rd = np.concatenate([rd.real,rd.imag], axis=2)
            return rd
        else:
            for i in self.RD_filenames:
                rd = np.load(self.RD_filenames[i])['arr_0']
                rd = np.concatenate([rd.real,rd.imag],axis=2)
                self.RD.append(rd)
            return self.RD

    def get_radarpointcloud(self, idx=None):
        if idx is not None:
            pc = np.load(self.radarpointcloud_filenames[idx], allow_pickle=True)[[5,6,7],:]   # Keeps only x,y,z
            pc = np.rollaxis(pc,1,0)
            pc[:,1] *= -1
            return pc
        else:
            for i in self.radarpointcloud_filenames:
                pc = np.load(self.radarpointcloud_filenames[i], allow_pickle=True)[[5,6,7],:]   # Keeps only x,y,z
                pc = np.rollaxis(pc,1,0)
                pc[:,1] *= -1
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
        # format as following [Range,Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix,y2_pix]
        box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4]].astype(np.float32) 


        ######################
        #  Encode the labels #
        ######################
        out_label=[]
        # if(self.encoder!=None):
        out_label = self.encode(box_labels).copy()      

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
