import os
import json
import numpy as np
import struct


from torch.utils.data import Dataset
from functools import reduce
from loguru import logger
from pyquaternion import Quaternion


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm


class NuScenes(Dataset):
    name = "nuScenes dataset instance"

    def __init__(self, features=None):
        self.feature_path = ""
        self.config = ""

        self.frame_sync = 0
        self.features = ['image', 'radarPC', 'lidarPC']
        
    def parse(self, file_id, file_path, file_name, config):
        logger.info("Parsing NuScenes data file:")
        self.config = config
        self.root_path = file_path
        
        # collect info from meta info
        with open(os.path.join(self.root_path, self.config, 'scene.json'), 'r') as f:
            self.scenes_info =  json.load(f)
        with open(os.path.join(self.root_path, self.config, 'sample.json'), 'r') as f:
            self.samples_info = json.load(f)
        with open(os.path.join(self.root_path, self.config, 'sample_data.json'), 'r') as f:
            self.sample_data_info = json.load(f)
        with open(os.path.join(self.root_path, self.config, 'sample_annotation.json'), 'r') as f:
            self.sample_annotation = json.load(f)
        with open(os.path.join(self.root_path, self.config, 'calibrated_sensor.json'), 'r') as f:
            self.calibrated_sensor = json.load(f)
        with open(os.path.join(self.root_path, self.config, 'instance.json'), 'r') as f:
            self.instances = json.load(f)
        with open(os.path.join(self.root_path, self.config, 'category.json'), 'r') as f:
            self.category = json.load(f)
        with open(os.path.join(self.root_path, self.config, 'ego_pose.json'), 'r') as f:
            self.ego_pose = json.load(f)

        # collect features (each feature as a sample_data dict) from first sample to last sample in all scenes
        self.image_frames = []
        self.lidarPC_frames = []
        self.radarPC_frames = []
        for scene in self.scenes_info:
            current_sample_token = scene["first_sample_token"]
            while current_sample_token != scene["last_sample_token"]:
                sensor_data_list = [sample_data for sample_data in self.sample_data_info if sample_data['is_key_frame'] and sample_data["sample_token"] == current_sample_token]
                self.image_frames.append([i for i in sensor_data_list if i["filename"].split('/')[1] == "CAM_FRONT"][0])
                self.radarPC_frames.append([i for i in sensor_data_list if i["filename"].split('/')[1] == "RADAR_FRONT"][0])
                self.lidarPC_frames.append([i for i in sensor_data_list if i["filename"].split('/')[1] == "LIDAR_TOP"][0])
                current_sample_token = [sample["token"] for sample in self.samples_info if sample["prev"] == current_sample_token][0]
            # The last sample
            sensor_data_list = [sample_data for sample_data in self.sample_data_info if sample_data['is_key_frame'] and sample_data["sample_token"] == current_sample_token]
            self.image_frames.append([i for i in sensor_data_list if i["filename"].split('/')[1] == "CAM_FRONT"][0])
            self.radarPC_frames.append([i for i in sensor_data_list if i["filename"].split('/')[1] == "RADAR_FRONT"][0])
            self.lidarPC_frames.append([i for i in sensor_data_list if i["filename"].split('/')[1] == "LIDAR_TOP"][0])
            logger.info(f"Parsed {scene['name']} with {scene['nbr_samples']} samples.")
        self.frame_sync = len(self.image_frames) 
        logger.info(f"In total: {self.frame_sync} samples.")
        return               

    def get_image(self, idx=None, for_visualize=False): 
        #image = np.asarray(Image.open(self.image_filenames[idx]))
        return os.path.join(self.root_path, self.image_frames[idx]["filename"])
    
    def get_RAD(self, idx=None, for_visualize=False):
        return None
    
    def get_RD(self, idx=None, for_visualize=False):
        return None
    
    def get_RA(self, idx=None, for_visualize=False):
        return None

    def read_from_file(self, file_name: str):
        """
        Loads RADAR data from a Point Cloud Data file. See details below.
        :param file_name: The path of the pointcloud file.
        :param invalid_states: Radar states to be kept. See details below.
        :param dynprop_states: Radar states to be kept. Use [0, 2, 6] for moving objects only. See details below.
        :param ambig_states: Radar states to be kept. See details below.
        To keep all radar returns, set each state filter to range(18).
        :return: <np.float: d, n>. Point cloud matrix with d dimensions and n points.

        Example of the header fields:
        # .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
        TYPE F F F I I F F F F F I I I I I I I I
        COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        WIDTH 125
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS 125
        DATA binary

        Below some of the fields are explained in more detail:

        x is front, y is left

        vx, vy are the velocities in m/s.
        vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
        We recommend using the compensated velocities.

        invalid_state: state of Cluster validity state.
        (Invalid states)
        0x01	invalid due to low RCS
        0x02	invalid due to near-field artefact
        0x03	invalid far range cluster because not confirmed in near range
        0x05	reserved
        0x06	invalid cluster due to high mirror probability
        0x07	Invalid cluster because outside sensor field of view
        0x0d	reserved
        0x0e	invalid cluster because it is a harmonics
        (Valid states)
        0x00	valid
        0x04	valid cluster with low RCS
        0x08	valid cluster with azimuth correction due to elevation
        0x09	valid cluster with high child probability
        0x0a	valid cluster with high probability of being a 50 deg artefact
        0x0b	valid cluster but no local maximum
        0x0c	valid cluster with high artefact probability
        0x0f	valid cluster with above 95m in near range
        0x10	valid cluster with high multi-target probability
        0x11	valid cluster with suspicious angle

        dynProp: Dynamic property of cluster to indicate if is moving or not.
        0: moving
        1: stationary
        2: oncoming
        3: stationary candidate
        4: unknown
        5: crossing stationary
        6: crossing moving
        7: stopped

        ambig_state: State of Doppler (radial velocity) ambiguity solution.
        0: invalid
        1: ambiguous
        2: staggered ramp
        3: unambiguous
        4: stationary candidates

        pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
        0: invalid
        1: <25%
        2: 50%
        3: 75%
        4: 90%
        5: 99%
        6: 99.9%
        7: <=100%
        """

        assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)

        meta = []
        with open(file_name, 'rb') as f:
            for line in f:
                line = line.strip().decode('utf-8')
                meta.append(line)
                if line.startswith('DATA'):
                    break

            data_binary = f.read()

        # Get the header rows and check if they appear as expected.
        assert meta[0].startswith('#'), 'First line must be comment'
        assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
        sizes = meta[3].split(' ')[1:]
        types = meta[4].split(' ')[1:]
        counts = meta[5].split(' ')[1:]
        width = int(meta[6].split(' ')[1])
        height = int(meta[7].split(' ')[1])
        data = meta[10].split(' ')[1]
        feature_count = len(types)
        assert width > 0
        assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
        assert height == 1, 'Error: height != 0 not supported!'
        assert data == 'binary'

        # Lookup table for how to decode the binaries.
        unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                         'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                         'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
        types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

        # Decode each point.
        offset = 0
        point_count = width
        points = []
        for i in range(point_count):
            point = []
            for p in range(feature_count):
                start_p = offset
                end_p = start_p + int(sizes[p])
                assert end_p < len(data_binary)
                point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
                point.append(point_p)
                offset = end_p
            points.append(point)

        # A NaN in the first point indicates an empty pointcloud.
        point = np.array(points[0])
        if np.any(np.isnan(point)):
            return np.zeros((feature_count, 0))

        # Convert to numpy matrix.
        points = np.array(points).transpose()

        # If no parameters are provided, use default settings.
        invalid_states = [0]
        dynprop_states = range(7)
        ambig_states = [3]

        # Filter points with an invalid state.
        valid = [p in invalid_states for p in points[-4, :]]
        points = points[:, valid]

        # Filter by dynProp.
        valid = [p in dynprop_states for p in points[3, :]]
        points = points[:, valid]

        # Filter by ambig_state.
        valid = [p in ambig_states for p in points[11, :]]
        points = points[:, valid]
        return points
    
    def aggregate_multisweep(self, sample_rec: dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 5,
                             min_distance: float = 1.0):
        # Get aggregated radar point cloud in reference frame.
        # The point cloud is transformed to the reference frame for visualization purposes.
        if chan.startswith("LIDAR"):
            all_pc = np.empty(shape=(4, 0))
        else:
            all_pc = np.empty(shape=(18, 0))

        # Get reference pose and timestamp.
        # ref_sd_token = sample_rec['data'][ref_chan]
        # ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        if ref_chan == "LIDAR_TOP":
            ref_sd_rec = [i for i in self.lidarPC_frames if i["sample_token"] == sample_rec["token"]][0] 
        elif ref_chan == "RADAR_FRONT":
            ref_sd_rec = [i for i in self.radarPC_frames if i["sample_token"] == sample_rec["token"]][0]
        else:
            logger.warning("ref chan not supported.") 
        ref_pose_rec = [i for i in self.ego_pose if i["token"] == ref_sd_rec["ego_pose_token"]][0] # nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = [i for i in self.calibrated_sensor if i["token"] == ref_sd_rec["calibrated_sensor_token"]][0] # nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        # sample_data_token = sample_rec['data'][chan]
        # current_sd_rec = nusc.get('sample_data', sample_data_token)
        current_sd_rec = [i for i in self.sample_data_info if i["sample_token"] == sample_rec["token"] and i['is_key_frame'] and i['filename'].split('/')[1] == chan][0] 
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            # current_pc = cls.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
            file_name = os.path.join(self.root_path, current_sd_rec["filename"])
            if chan.startswith("LIDAR"):
                assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)
                scan = np.fromfile(file_name, dtype=np.float32)
                current_pc = scan.reshape((-1, 5))[:, :4].T
            else:
                assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)
                current_pc = self.read_from_file(file_name)
            # current_pc.remove_close(min_distance)
            min_distance = 1.0
            x_filt = np.abs(current_pc[0, :]) < min_distance
            y_filt = np.abs(current_pc[1, :]) < min_distance
            not_close = np.logical_not(np.logical_and(x_filt, y_filt))
            current_pc = current_pc[:, not_close]

            # Get past pose.
            # current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            current_pose_rec = [i for i in self.ego_pose if i["token"] == current_sd_rec["ego_pose_token"]][0]
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            # current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            current_cs_rec = [i for i in self.calibrated_sensor if i["token"] == current_sd_rec['calibrated_sensor_token']][0]
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            # current_pc.transform(trans_matrix)
            current_pc[:3, :] = trans_matrix.dot(np.vstack((current_pc[:3, :], np.ones(current_pc.shape[1]))))[:3, :]

            # # Add time vector which can be used as a temporal feature.
            # time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
            # times = time_lag * np.ones((1, current_pc.nbr_points()))
            # all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc = np.hstack((all_pc, current_pc))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                # current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])
                current_sd_rec = [i for i in self.sample_data_info if i["token"] == current_sd_rec["prev"]][0]
        return all_pc
    
    def get_radarpointcloud(self, idx=None, n_radars=5, nsweeeps=6, for_visualize=False):
        # Output: Point cloud on RADAR_FRONT sensor coord system
        # FIELDS: [x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms]
        pc = np.empty(shape=(18, 0))
        sample = [sample for sample in self.samples_info if sample["token"] == self.radarPC_frames[idx]["sample_token"]][0]
        for sensor in {'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'}:
            
            pts = self.aggregate_multisweep(sample, sensor, "RADAR_FRONT")

            # ################################
            # # aggregate multisweeps for each sensor and then convert each sensor pc to ego vehicle frame
            # pts = self.aggregate_multisweep(sample, sensor, sensor)
            # # Get calibration information
            # # sample_data = nusc.get('sample_data', sample['data'][sensor])
            # sample_data = [i for i in self.sample_data_info if i["filename"].split('/')[1] == sensor and i["sample_token"] == sample["token"]][0]
            # # calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            # calibrated_sensor = [i for i in self.calibrated_sensor if i["token"] == sample_data['calibrated_sensor_token']][0]
            # # Get rotation matirx from sensor to vehicle frame
            # rotation_matrix = Quaternion(calibrated_sensor['rotation']).rotation_matrix
            # # Rotate point cloud
            # pts[:3, :] = np.dot(rotation_matrix, pts[:3, :])
            # # Translate point cloud
            # pts[:3, :] = np.add(pts[:3, :], np.expand_dims(calibrated_sensor['translation'], axis=-1))
            # # Rotate velocity vectors
            # pts[8:10, :] = np.dot(rotation_matrix[:2, :2], pts[8:10, :])
            # #################################

            pc = np.append(pc, pts, axis=1)

        pc = pc.T
        return pc

    def get_lidarpointcloud(self, idx=None, nsweeeps=1, for_visualize=False):
        # Output: Point cloud on LIDAR_TOP sensor coord system
        # FIELDS: [x, y, z, intensity, ring index]
        sample = [sample for sample in self.samples_info if sample["token"] == self.lidarPC_frames[idx]["sample_token"]][0]
        pts = self.aggregate_multisweep(sample, "LIDAR_TOP", "LIDAR_TOP")
        
        # # Get calibration information
        # # sample_data = nusc.get('sample_data', sample['data'][sensor])
        # sample_data = self.lidarPC_frames[idx]
        # # calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        # calibrated_sensor = [i for i in self.calibrated_sensor if i["token"] == sample_data['calibrated_sensor_token']][0]
        # # Get rotation matirx from sensor to vehicle frame
        # rotation_matrix = Quaternion(calibrated_sensor['rotation']).rotation_matrix
        # # Rotate point cloud
        # pts[:3, :] = np.dot(rotation_matrix, pts[:3, :])
        # # Translate point cloud
        # pts[:3, :] = np.add(pts[:3, :], np.expand_dims(calibrated_sensor['translation'], axis=-1))
        # # # Rotate velocity vectors
        # # pts[8:10, :] = np.dot(rotation_matrix[:2, :2], pts[8:10, :])

        pts = pts.T
        return pts

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
        # sample = [sample for sample in self.samples_info if sample["token"] == self.radarPC_frames[idx]["sample_token"]][0]
        objs = [obj for obj in self.sample_annotation if obj["sample_token"] == self.radarPC_frames[idx]["sample_token"]]
        # for i,obj in enumerate(objs):
        #     print(f"{i} raw obj in my app:")
        #     print(f"token: {obj['token']}")
        #     print(f"sample_token: {obj['sample_token']}")
        #     print(f"translation: {obj['translation']}")
        #     print(f"size: {obj['size']}")
        #     print(f"rotation: {obj['rotation']}")
        #     print("\n")
        # Box(record['translation'], record['size'], Quaternion(record['rotation']), name=record['category_name'], token=record['token'])

        # # Move box to ego vehicle coord system.
        # box.translate(-np.array(pose_record['translation']))
        # box.rotate(Quaternion(pose_record['rotation']).inverse)

        # #  Move box to sensor coord system.
        # box.translate(-np.array(cs_record['translation']))
        # box.rotate(Quaternion(cs_record['rotation']).inverse)

        gt = []
        for i, obj in enumerate(objs):
            # find category
            category_token = [i["category_token"] for i in self.instances if i["token"] == obj["instance_token"]][0]
            category = [i["name"] for i in self.category if i["token"] == category_token][0]

            # center
            center = np.array(obj["translation"])
            # orientation
            orientation = Quaternion(obj['rotation'])
            # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
            w, l, h = obj["size"]
            x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
            y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
            z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
            corners = np.vstack((x_corners, y_corners, z_corners))
            
            if feature_name == "lidarPC":
                ego_pose_record = [i for i in self.ego_pose if i["token"] == self.lidarPC_frames[idx]["ego_pose_token"]][0]
                calibrated_sensor_record = [i for i in self.calibrated_sensor if i["token"] == self.lidarPC_frames[idx]["calibrated_sensor_token"]][0]
            elif feature_name == "radarPC":
                ego_pose_record = [i for i in self.ego_pose if i["token"] == self.radarPC_frames[idx]["ego_pose_token"]][0]
                calibrated_sensor_record = [i for i in self.calibrated_sensor if i["token"] == self.radarPC_frames[idx]["calibrated_sensor_token"]][0]
            else:
                ego_pose_record = [i for i in self.ego_pose if i["token"] == self.image_frames[idx]["ego_pose_token"]][0]
                calibrated_sensor_record = [i for i in self.calibrated_sensor if i["token"] == self.image_frames[idx]["calibrated_sensor_token"]][0]

            # Move box to ego vehicle coord system.
            center += -np.array(ego_pose_record['translation'])
            quaternion = Quaternion(ego_pose_record['rotation']).inverse
            center = np.dot(quaternion.rotation_matrix, center)
            orientation = quaternion * orientation

            # Move box to sensor coord system.
            center += -np.array(calibrated_sensor_record['translation'])
            quaternion = Quaternion(calibrated_sensor_record['rotation']).inverse
            center = np.dot(quaternion.rotation_matrix, center)
            orientation = quaternion * orientation

            # Rotate
            corners = np.dot(orientation.rotation_matrix, corners)
            # Translate
            x, y, z = center
            corners[0, :] = corners[0, :] + x
            corners[1, :] = corners[1, :] + y
            corners[2, :] = corners[2, :] + z

            if feature_name in ("lidarPC", "radarPC"):
                # corners = corners[:2, :]
                gt.append(corners.flatten().tolist() + [category])    
            else:
                # view from camera 
                camera_intrinsic = calibrated_sensor_record["camera_intrinsic"]
                viewpad = np.eye(4)
                viewpad[:3, :3] = camera_intrinsic
                corners_img = np.concatenate((corners, np.ones((1, 8))))
                corners_img = np.dot(viewpad, corners_img)
                corners_img = corners_img[:3, :]
                corners_img = corners_img / corners_img[2:3, :].repeat(3, 0).reshape(3, 8)

                corners_img = corners_img[:2, :]
                # print(i)
                # print("converted gt corners in my app:")
                # print(corners)
                # print('\n')

                # check box if it's in 2D image of CAM_FRONT
                imsize = [1600, 900]
                visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
                visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
                visible = np.logical_and(visible, corners_img[1, :] > 0)
                visible = np.logical_and(visible, corners[2, :] > 1)
                in_front = corners[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.
                if any(visible) and all(in_front):
                    gt.append(corners_img.flatten().tolist() + [category])
        
        return gt
    
    def prepare_for_train(self, features, train_cfg, model_cfg, splittype=None):
        return 

    def __len__(self):
        return self.frame_sync

    def __getitem__(self, index):
        
       return None