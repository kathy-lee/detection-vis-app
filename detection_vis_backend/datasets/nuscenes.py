import os
import json
import numpy as np


from torch.utils.data import Dataset
from loguru import logger


class NuScenes(Dataset):
    name = "nuScenes dataset instance"

    def __init__(self, features=None):
        self.feature_path = ""
        self.config = ""

        self.frame_sync = 0
        self.features = ['image', 'radarPC', 'lidarPC']
        
    def parse(self, file_id, file_path, file_name, config):
        self.config = config
        self.root_path = file_path
        
        # collect scenes info from meta info
        scenes_info =  json.load(os.path.join(file_path, 'meta/v1.0-trainval', 'scene.json'))
        samples_info = json.load(os.path.join(file_path, 'meta/v1.0-trainval', 'sample.json'))
        sample_data_info = json.load(os.path.join(file_path, 'meta/v1.0-trainval', 'sample_data.json'))
        self.image_filenames = []
        self.lidarPC_filenames = []
        self.radarPC_filenames = []
        for scene in scenes_info:
            current_sample_token = scene["first_sample_token"]
            while current_sample_token != scene["last_sample_token"]:
                sensor_data_list = [sample_data for sample_data in sample_data_info if sample_data['is_key_frame'] and sample_data["sample_token"] == current_sample_token]
                self.image_filenames.append([os.path.join(self.root_path, i["filename"]) for i in sensor_data_list if i["filename"].split('/')[1] == "CAM_FRONT"][0])
                self.radarPC_filenames.append([os.path.join(self.root_path, i["filename"]) for i in sensor_data_list if i["filename"].split('/')[1] == "RADAR_FRONT"][0])
                self.lidarPC_filenames.append([os.path.join(self.root_path, i["filename"]) for i in sensor_data_list if i["filename"].split('/')[1] == "LIDAR_TOP"][0])
                current_sample_token = [sample["token"] for sample in samples_info if sample["prev"] == current_sample_token][0]
            # The last sample
            sensor_data_list = [sample_data for sample_data in sample_data_info if sample_data['is_key_frame'] and sample_data["sample_token"] == current_sample_token]
            self.image_filenames.append([os.path.join(self.root_path, i["filename"]) for i in sensor_data_list if i["filename"].split('/')[1] == "CAM_FRONT"][0])
            self.radarPC_filenames.append([os.path.join(self.root_path, i["filename"]) for i in sensor_data_list if i["filename"].split('/')[1] == "RADAR_FRONT"][0])
            self.lidarPC_filenames.append([os.path.join(self.root_path, i["filename"]) for i in sensor_data_list if i["filename"].split('/')[1] == "LIDAR_TOP"][0])
        
        self.frame_sync = len(self.image_filenames) 
        return
                    
                        

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
        
        gt = []
        
        return gt
    
    def prepare_for_train(self, features, train_cfg, model_cfg, splittype=None):
        return 

    def __len__(self):
        return self.frame_sync

    def __getitem__(self, index):
        
       return None