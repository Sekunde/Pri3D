# Basic libs
import os, time, glob, random, pickle, copy, torch
import open3d as o3d
import numpy as np
import open3d
from scipy.spatial.transform import Rotation
from torchvision.transforms import transforms

# Dataset parent class
from torch.utils.data import Dataset
from collections import namedtuple
from common.camera_intrinsics import adjust_intrinsic

import numpy as np
import trimesh
from PIL import Image

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def odometry_to_positions(odometry):
    T_w_cam0 = odometry.reshape(3, 4)
    cam02world = np.vstack((T_w_cam0, [0, 0, 0, 1]))
    return cam02world


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    #velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    data['Tr'] = velo2cam

    return data

def load_calib(sequence_path):
      """Load and compute intrinsic and extrinsic calibration parameters."""
      # We'll build the calibration parameters as a dictionary, then
      # convert it to a namedtuple to prevent it from being modified later
      data = {}

      # Load the calibration file
      calib_filepath = os.path.join(sequence_path, 'calib.txt')
      filedata = read_calib_file(calib_filepath)

      # Create 3x4 projection matrices
      P_rect_00 = np.reshape(filedata['P0'], (3, 4))
      P_rect_10 = np.reshape(filedata['P1'], (3, 4))
      P_rect_20 = np.reshape(filedata['P2'], (3, 4))
      P_rect_30 = np.reshape(filedata['P3'], (3, 4))

      data['P_rect_00'] = P_rect_00
      data['P_rect_10'] = P_rect_10
      data['P_rect_20'] = P_rect_20
      data['P_rect_30'] = P_rect_30

      # Compute the rectified extrinsics from cam0 to camN
      T1 = np.eye(4)
      T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
      T2 = np.eye(4)
      T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
      T3 = np.eye(4)
      T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

      # Compute the velodyne to rectified camera coordinate transforms
      data['velo2cam0'] = np.reshape(filedata['Tr'], (3, 4))
      data['velo2cam0'] = np.vstack([data['velo2cam0'], [0, 0, 0, 1]])
      data['velo2cam1'] = T1.dot(data['velo2cam0'])
      data['velo2cam2'] = T2.dot(data['velo2cam0'])
      data['velo2cam3'] = T3.dot(data['velo2cam0'])

      # Compute the camera intrinsics
      data['K_cam0'] = P_rect_00[0:3, 0:3]
      data['K_cam1'] = P_rect_10[0:3, 0:3]
      data['K_cam2'] = P_rect_20[0:3, 0:3]
      data['K_cam3'] = P_rect_30[0:3, 0:3]

      # Compute the stereo baselines in meters by projecting the origin of
      # each camera frame into the velodyne frame and computing the distances
      # between them
      p_cam = np.array([0, 0, 0, 1])
      p_velo0 = np.linalg.inv(data['velo2cam0']).dot(p_cam)
      p_velo1 = np.linalg.inv(data['velo2cam1']).dot(p_cam)
      p_velo2 = np.linalg.inv(data['velo2cam2']).dot(p_cam)
      p_velo3 = np.linalg.inv(data['velo2cam3']).dot(p_cam)

      data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
      data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

      calib = namedtuple('CalibData', data.keys())(*data.values())

      return calib

class KITTI(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """

    mapping = {'2011_10_03_drive_0027_sync': '00',
               '2011_10_03_drive_0042_sync': '01',
               '2011_10_03_drive_0034_sync': '02',
               '2011_09_30_drive_0016_sync': '04',
               '2011_09_30_drive_0018_sync': '05',
               '2011_09_30_drive_0020_sync': '06',
               '2011_09_30_drive_0027_sync': '07',
               '2011_09_30_drive_0028_sync': '08',
               '2011_09_30_drive_0033_sync': '09',
               '2011_09_30_drive_0034_sync': '10' }

    def __init__(self, config, phase):
        super(KITTI, self).__init__()
        BASE_PATH = self.path


        PATH_ODOMETRY = os.path.join(BASE_PATH, 'odometry/data_odometry_velodyne/dataset/poses/')
        PATH_COLOR = os.path.join(BASE_PATH, 'odometry/data_odometry_color/dataset/sequences/')
        PATH_DEPTH = os.path.join(BASE_PATH, 'depth/data_depth_annotated/')

        PATH = os.path.join(BASE_PATH, 'preprocessed/overlap10.txt')
        PATH_CENTER = os.path.join(BASE_PATH, 'preprocessed/overlap50/')
        
        self.collate_fn = None
        self.config = config
        self.files = open(self.PATH).readlines()
        self.pose = {}
        self.intrinsic = {}
        self.transforms = {}
        #self.crop_size = config.size
        self.IMAGE_SIZE = config.size

        self.transforms['color'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std']),
        ])

        self.resize =  transforms.Resize(self.IMAGE_SIZE[0])


        # extrinsics and intrinsics
        for line in self.files:
            split = line.split()
            drive_id = split[0]
            sequence_id = self.mapping[drive_id]

            if sequence_id not in self.pose:
                data_path = os.path.join(self.PATH_ODOMETRY, '{}.txt'.format(sequence_id))
                pose = np.genfromtxt(data_path)
                self.pose[sequence_id] = pose

            if sequence_id not in self.intrinsic:
                calib = load_calib(os.path.join(self.PATH_COLOR, sequence_id))
                intrinsic = calib.K_cam2
                self.intrinsic[sequence_id] = intrinsic

    def __len__(self):
        return len(self.files)

    def build_dataset(self):
        pass

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0] - self.IMAGE_SIZE[0] // 2), 0)
        if bbox1_i + self.IMAGE_SIZE[0] >= image1.shape[1]:
            bbox1_i = image1.shape[1] - self.IMAGE_SIZE[0]
        bbox1_j = max(int(central_match[1] - self.IMAGE_SIZE[1] // 2), 0)
        if bbox1_j + self.IMAGE_SIZE[1] >= image1.shape[2]:
            bbox1_j = image1.shape[2] - self.IMAGE_SIZE[1]

        bbox2_i = max(int(central_match[2] - self.IMAGE_SIZE[0] // 2), 0)
        if bbox2_i + self.IMAGE_SIZE[0] >= image2.shape[1]:
            bbox2_i = image2.shape[1] - self.IMAGE_SIZE[0]
        bbox2_j = max(int(central_match[3] - self.IMAGE_SIZE[1] // 2), 0)
        if bbox2_j + self.IMAGE_SIZE[1] >= image2.shape[2]:
            bbox2_j = image2.shape[2] - self.IMAGE_SIZE[1]

        return (
                image1[:,
                bbox1_i : bbox1_i + self.IMAGE_SIZE[0],
                bbox1_j : bbox1_j + self.IMAGE_SIZE[1]
            ],
            np.array([bbox1_i, bbox1_j]),
            image2[:,
                bbox2_i : bbox2_i + self.IMAGE_SIZE[0],
                bbox2_j : bbox2_j + self.IMAGE_SIZE[1]
            ],
            np.array([bbox2_i, bbox2_j])
        )
    


    def __getitem__(self, idx):
        split = self.files[idx].split()
        drive_id = split[0]
        folder = split[1]
        sequence_id = self.mapping[drive_id]
        t1, t2 = int(split[2].split('.')[0]), int(split[3].split('.')[0])
        odometry1 = self.pose[sequence_id][t1]
        odometry2 = self.pose[sequence_id][t2]
        camera2world1 = odometry_to_positions(odometry1)
        camera2world2 = odometry_to_positions(odometry2)

        depth1 = os.path.join(self.PATH_DEPTH, folder, drive_id, 'proj_depth', 'groundtruth', 'image_02', '{:010d}.png'.format(t1))
        #depth1 = os.path.join(self.PATH_DEPTH, folder, drive_id, 'proj_depth', 'velodyne_raw', 'image_02', '{:010d}.png'.format(t1))
        depth1 = Image.open(depth1)
        original_size = depth1.size[::-1]
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth1) > 255)
        depth1 = np.array(depth1, dtype=int)
        depth1 = depth1.astype(np.float) / 256.
        depth1[depth1 == 0] = -1.
        depth1 = torch.from_numpy(depth1).float()

        depth2 = os.path.join(self.PATH_DEPTH, folder, drive_id, 'proj_depth', 'groundtruth', 'image_02', '{:010d}.png'.format(t2))
        #depth2 = os.path.join(self.PATH_DEPTH, folder, drive_id, 'proj_depth', 'velodyne_raw', 'image_02', '{:010d}.png'.format(t2))
        depth2 = Image.open(depth2)
        #depth2 = self.transforms['depth'](depth2)
        assert(np.max(depth2) > 255)
        depth2 = np.array(depth2, dtype=int)
        depth2 = depth2.astype(np.float) / 256.
        depth2[depth2 == 0] = -1.
        depth2 = torch.from_numpy(depth2).float()

        color1 = os.path.join(self.PATH_COLOR, sequence_id, 'image_2',"{:06d}.png".format(t1))
        color1 = Image.open(color1)
        color1 = self.transforms['color'](color1)

        color2 = os.path.join(self.PATH_COLOR, sequence_id, 'image_2',"{:06d}.png".format(t2))
        color2 = Image.open(color2)
        color2 = self.transforms['color'](color2)
        intrinsic = self.intrinsic[sequence_id]

        if self.config.resize:
            original_size = color1.shape[1:]

            color1 = self.resize(color1)[:,:, :self.IMAGE_SIZE[1]]
            color2 = self.resize(color2)[:,:, :self.IMAGE_SIZE[1]]
            depth1 = self.resize(depth1[None,:,:])[0,:,:self.IMAGE_SIZE[1]]
            depth2 = self.resize(depth2[None,:,:])[0,:,:self.IMAGE_SIZE[1]]
            bbox1= torch.FloatTensor([-0.5,-0.5])
            bbox2= torch.FloatTensor([-0.5,-0.5])
            resized_size = color1.shape[1:]
            intrinsic = adjust_intrinsic(intrinsic, original_size, resized_size)

        else:
            central_match = []
            name = '_'.join([split[1], split[2].split('.')[0], split[3].split('.')[0]]) + '.npy'
            central_match = np.load(os.path.join(self.PATH_CENTER, drive_id, name))

            # bound_check
            mask_h1 = (central_match[:,0] - self.IMAGE_SIZE[0] / 2 >= 0) & (central_match[:,0] + self.IMAGE_SIZE[0] / 2 < color1.shape[1])
            mask_w1 = (central_match[:,1] - self.IMAGE_SIZE[1] / 2 >= 0) & (central_match[:,1] + self.IMAGE_SIZE[1] / 2 < color1.shape[2])
            mask_h2 = (central_match[:,2] - self.IMAGE_SIZE[0] / 2 >= 0) & (central_match[:,2] + self.IMAGE_SIZE[0] / 2 < color2.shape[1])
            mask_w2 = (central_match[:,3] - self.IMAGE_SIZE[1] / 2 >= 0) & (central_match[:,3] + self.IMAGE_SIZE[1] / 2 < color2.shape[2])
            central_match = central_match[mask_h1&mask_w1&mask_h2&mask_w2]

            central_idx = np.random.choice(central_match.shape[0])
            color1, bbox1, color2, bbox2 = self.crop(color1, color2, central_match[central_idx])

            depth1 = depth1[
                bbox1[0] : bbox1[0] + self.IMAGE_SIZE[0],
                bbox1[1] : bbox1[1] + self.IMAGE_SIZE[1]]
            depth2 = depth2[
                bbox2[0] : bbox2[0] + self.IMAGE_SIZE[0],
                bbox2[1] : bbox2[1] + self.IMAGE_SIZE[1]]

        return {'color1': color1, 'depth1': depth1,
                'color2': color2, 'depth2': depth2,
                #'id1': fname1, 'id2': fname2, 
                'intrinsics1': torch.from_numpy(intrinsic).float(),
                'intrinsics2': torch.from_numpy(intrinsic).float(),
                'bbox1': bbox1, 'bbox2': bbox2,
                'world2camera1': torch.inverse(torch.from_numpy(camera2world1).float()),
                'world2camera2': torch.inverse(torch.from_numpy(camera2world2).float())}

if __name__ == "__main__":
    dataset = KITTIDataset(None, 'train')
    points = []
    for idx, data in enumerate(dataset):
        print(idx, len(dataset))
        points.append(data)
    print(sum(points) / len(dataset))
