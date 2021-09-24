import h5py
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import time
from tqdm import tqdm


def preprocess_image(image):
    image = image.astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    image /= 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    return image


class MegaDepth(Dataset):

    MIN_OVERLAP_RATIO = .3
    PAIRS_PER_SCENE = 4000

    def __init__(self, config=None, phase=None, chunk=False):

        self.config = config
        self.BASE_PATH = config.path
        self.IMAGE_SIZE = config.size
        self.sfm_path = os.path.join(self.BASE_PATH, 'Undistorted_SfM')
        self.meta_path = os.path.join(self.BASE_PATH, 'scene_info')
        self.collate_fn = None
        # get scenes
        self.scenes = [scene for scene in os.listdir(os.path.join(self.BASE_PATH, 'Undistorted_SfM'))]
        self.dataset = [None] * len(self.scenes) * self.PAIRS_PER_SCENE

    def build_dataset(self):
        self.dataset = []
        print('Building a new training dataset...')
        for idx, scene in enumerate(self.scenes):
            print('{}/{}'.format(idx, len(self.scenes)), end='\r')
            scene_info_path = os.path.join(self.meta_path, '%s.npz' % scene)
            if not os.path.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            overlap_matrix = scene_info['overlap_matrix']
            valid = overlap_matrix >= self.MIN_OVERLAP_RATIO
            pairs = np.vstack(np.where(valid))
            
            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            points3D_id_to_2D = scene_info['points3D_id_to_2D']
            points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']

            if self.PAIRS_PER_SCENE !=0:
                try:
                    selected_ids = np.random.choice(pairs.shape[1], self.PAIRS_PER_SCENE)
                except:
                    continue
            else:
                selected_ids = range(pairs.shape[1])
            
            for pair_idx in selected_ids:
                idx1 = pairs[0, pair_idx]
                idx2 = pairs[1, pair_idx]
                matches = np.array(list(
                    points3D_id_to_2D[idx1].keys() &
                    points3D_id_to_2D[idx2].keys()
                ))

                point3D_id = np.random.choice(matches)
                point2D1 = points3D_id_to_2D[idx1][point3D_id]
                point2D2 = points3D_id_to_2D[idx2][point3D_id]
                central_match = np.array([
                    point2D1[1], point2D1[0],
                    point2D2[1], point2D2[0]
                ])
                self.dataset.append({
                    'image_path1': image_paths[idx1],
                    'depth_path1': depth_paths[idx1],
                    'intrinsics1': intrinsics[idx1],
                    'pose1': poses[idx1],
                    'image_path2': image_paths[idx2],
                    'depth_path2': depth_paths[idx2],
                    'intrinsics2': intrinsics[idx2],
                    'pose2': poses[idx2],
                    'central_match': central_match,
                })
        np.random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        depth_path1 = os.path.join(self.BASE_PATH, pair_metadata['depth_path1'])
        image_path1 = os.path.join(self.BASE_PATH, pair_metadata['image_path1'])
        intrinsics1 = pair_metadata['intrinsics1']
        pose1 = pair_metadata['pose1']

        depth_path2 = os.path.join(self.BASE_PATH, pair_metadata['depth_path2'])
        image_path2 = os.path.join(self.BASE_PATH, pair_metadata['image_path2'])
        intrinsics2 = pair_metadata['intrinsics2']
        pose2 = pair_metadata['pose2']

        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])
        image1 = Image.open(image_path1)
        image1 = np.array(image1)
        # check image1.mode == 'RGB'

        with h5py.File(depth_path2, 'r') as hdf5_file:
            depth2 = np.array(hdf5_file['/depth'])
        image2 = Image.open(image_path2)
        image2 = np.array(image2)

        central_match = pair_metadata['central_match']

        image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

        depth1 = depth1[
            bbox1[0] : bbox1[0] + self.IMAGE_SIZE[0],
            bbox1[1] : bbox1[1] + self.IMAGE_SIZE[1]
        ]
        depth2 = depth2[
            bbox2[0] : bbox2[0] + self.IMAGE_SIZE[0],
            bbox2[1] : bbox2[1] + self.IMAGE_SIZE[1]
        ]

        return (
            image1, depth1, intrinsics1, pose1, bbox1,
            image2, depth2, intrinsics2, pose2, bbox2
        )

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0] - self.IMAGE_SIZE[0] // 2), 0)
        if bbox1_i + self.IMAGE_SIZE[0] >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.IMAGE_SIZE[0]
        bbox1_j = max(int(central_match[1] - self.IMAGE_SIZE[1] // 2), 0)
        if bbox1_j + self.IMAGE_SIZE[1] >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.IMAGE_SIZE[1]

        bbox2_i = max(int(central_match[2] - self.IMAGE_SIZE[0] // 2), 0)
        if bbox2_i + self.IMAGE_SIZE[0] >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.IMAGE_SIZE[0]
        bbox2_j = max(int(central_match[3] - self.IMAGE_SIZE[1] // 2), 0)
        if bbox2_j + self.IMAGE_SIZE[1] >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.IMAGE_SIZE[1]

        return (
            image1[
                bbox1_i : bbox1_i + self.IMAGE_SIZE[0],
                bbox1_j : bbox1_j + self.IMAGE_SIZE[1]
            ],
            np.array([bbox1_i, bbox1_j]),
            image2[
                bbox2_i : bbox2_i + self.IMAGE_SIZE[0],
                bbox2_j : bbox2_j + self.IMAGE_SIZE[1]
            ],
            np.array([bbox2_i, bbox2_j])
        )

    def __getitem__(self, idx):
        (
            image1, depth1, intrinsics1, pose1, bbox1,
            image2, depth2, intrinsics2, pose2, bbox2
        ) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1)
        image2 = preprocess_image(image2)

        return {
            'color1': torch.from_numpy(image1.astype(np.float32)),
            'depth1': torch.from_numpy(depth1.astype(np.float32)),
            'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32)),
            'world2camera1': torch.from_numpy(pose1.astype(np.float32)),
            'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
            'color2': torch.from_numpy(image2.astype(np.float32)),
            'depth2': torch.from_numpy(depth2.astype(np.float32)),
            'intrinsics2': torch.from_numpy(intrinsics2.astype(np.float32)),
            'world2camera2': torch.from_numpy(pose2.astype(np.float32)),
            'bbox2': torch.from_numpy(bbox2.astype(np.float32))
        }


if __name__ == "__main__":
    test_data = MegaDepth()
    test_data.build_dataset()
    for data in test_data:
        print(data)

