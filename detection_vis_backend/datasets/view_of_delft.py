import os
import numpy as np


from torch.utils.data import Dataset
from loguru import logger


class ViewOfDelft(Dataset):
    name = "View of Delft dataset instance"

    def __init__(self, features=None):
        self.feature_path = ""
        self.config = ""

        self.frame_sync = 0
        self.features = ['image', 'radarPC', 'lidarPC']
        
    def parse(self, file_id, file_path, file_name, config):
        self.config = config
        self.root_path = file_path

        directory = os.path.join(self.root_path, 'radar/training/label_2')
        self.label_filenames = sorted([os.path.join(directory, filename) for filename in os.listdir(directory)])
        self.image_filenames = [os.path.join(self.root_path, 'radar/training/image_2', filename.split('/')[-1].replace('txt', 'jpg')) for filename in self.label_filenames]
        self.radarPC_filenames = [os.path.join(self.root_path, 'radar/training/velodyne', filename.split('/')[-1].replace('txt', 'bin')) for filename in self.label_filenames]
        self.lidarPC_filenames = [os.path.join(self.root_path, 'lidar/training/velodyne', filename.split('/')[-1].replace('txt', 'bin')) for filename in self.label_filenames]
        self.calib_radar_filenames = [os.path.join(self.root_path, 'radar/training/calib', filename.split('/')[-1]) for filename in self.label_filenames]
        self.calib_lidar_filenames = [os.path.join(self.root_path, 'lidar/training/calib', filename.split('/')[-1]) for filename in self.label_filenames]
        self.frame_sync = len(self.label_filenames) 
        logger.info(f'Parsed frames: {self.frame_sync}')
        return 

    def get_image(self, idx=None, for_visualize=False): 
        #image = np.asarray(Image.open(self.image_filenames[idx]))
        return self.image_filenames[idx]
    
    def get_RAD(self, idx=None, for_visualize=False):
        return None
    
    def get_ADC(self, idx=None, for_visualize=False):
        return None
    
    def get_RD(self, idx=None, for_visualize=False):
        return None
    
    def get_RA(self, idx=None, for_visualize=False):
        return None

    def get_radarpointcloud(self, idx=None, for_visualize=False):
        # Nx7 array: [x, y, z, RCS, v_r, v_r_compensated, time]
        # where `v_r` is the relative radial velocity, `v_r_compensated` is the absolute (i.e. ego motion compensated) radial velocity of the point.
        pc = np.fromfile(self.radarPC_filenames[idx], dtype=np.float32).reshape(-1, 7)
        return pc

    def get_lidarpointcloud(self, idx=None, for_visualize=False):
        # Nx4 array: [x,y,z,reflectance]
        pc = np.fromfile(self.lidarPC_filenames[idx], dtype=np.float32).reshape(-1, 4)
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
        # Annotation format: Nx16
        # 1    Class        Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
        # 1    truncated    Not used, only there to be compatible with KITTI format.
        # 1    occluded     Integer (0,1,2) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded.
        # 1    alpha        Observation angle of object, ranging [-pi..pi]
        # 4    bbox         2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates. This was automatically calculated from the 3D boxes.
        # 3    dimensions   3D object dimensions: height, width, length (in meters)
        # 3    location     3D object location x,y,z in camera coordinates (in meters)
        # 1    rotation     Rotation around -Z axis of the LiDAR sensor [-pi..pi]  
        # 1    score
        
        gt = []
        with open(self.label_filenames[idx], 'r') as text:
            annos = text.readlines()

        # The "image" branch below projects the boxes with T_lidar_to_camera and intrinsic, but
        # this block only loaded them for "lidarPC" -- so get_label("image", ...) raised
        # UnboundLocalError. The image projection is lidar-frame based, so it needs the lidar
        # calibration, not the radar one.
        # if feature_name == "lidarPC":
        if feature_name in ("lidarPC", "image"):
            with open(self.calib_lidar_filenames[idx], "r") as f:
                lines = f.readlines()
                intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
                T_lidar_to_camera = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
                T_lidar_to_camera = np.concatenate([T_lidar_to_camera, [[0, 0, 0, 1]]], axis=0)

        if feature_name == "radarPC":
            with open(self.calib_radar_filenames[idx], "r") as f:
                lines = f.readlines()
                intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
                T_radar_to_camera = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
                T_radar_to_camera = np.concatenate([T_radar_to_camera, [[0, 0, 0, 1]]], axis=0)
        
        for anno in annos:   
            anno = anno.split()
            cls, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = anno
            h, w, l, x, y, z, rotation = map(float, [h, w, l, x, y, z, rot])
            x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            z_corners = [0, 0, 0, 0, h, h, h, h]
            corners_3d = np.vstack([x_corners, y_corners, z_corners])

            rotation = -(rotation + np.pi / 2)  # undo changes made to rotation
            rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                                [np.sin(rotation), np.cos(rotation), 0],
                                [0, 0, 1]])
            
            if feature_name == "lidarPC":
                center = (np.linalg.inv(T_lidar_to_camera) @ np.array([x, y, z, 1]))[:3]
                obj = np.dot(rot_matrix, corners_3d).T + center
                gt.append(obj.flatten().tolist() + [cls])

            if feature_name == "radarPC":
                center = (np.linalg.inv(T_radar_to_camera) @ np.array([x, y, z, 1]))[:3]
                obj = np.dot(rot_matrix, corners_3d).T + center
                gt.append(obj.flatten().tolist() + [cls])
            
            if feature_name == "image":
                center = (np.linalg.inv(T_lidar_to_camera) @ np.array([x, y, z, 1]))[:3]
                new_corner_3d = np.dot(rot_matrix, corners_3d).T + center
                new_corners_3d_hom = np.concatenate((new_corner_3d, np.ones((8, 1))), axis=1)
                new_corners_3d_hom = T_lidar_to_camera.dot(new_corners_3d_hom.T).T
                corners_img = np.dot(new_corners_3d_hom, intrinsic.T)
                corners_img = corners_img[:, :2].T / corners_img[:, 2]
                gt.append(corners_img.flatten().tolist() + [cls])
        return gt
    
    def prepare_for_train(self, features, train_cfg, model_cfg, splittype=None):
        return 

    def __len__(self):
        return self.frame_sync

    def __getitem__(self, index):
        logger.debug(f"Data item index: {index}")
        

        data_dict = {}
        return data_dict