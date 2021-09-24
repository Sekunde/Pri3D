import os
import csv
import torch
import logging
import random
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset 
from torchvision.transforms import transforms

from common.camera_intrinsics import get_scannet_intrinsic
from common.io3d import write_triangle_mesh
from dataset.collate_fn import collate_fn_factory
from dataset.voxelizer import Voxelizer

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class ScanNet(Dataset):
    SPLITS = {
        'train': 'overlap30.txt',
        'val': 'overlap30_val.txt',
        'trainval': 'overlap30_trainval.txt'
        }

    @staticmethod
    def load_data(filepath):
        pointcloud = torch.load(filepath)
        coords = pointcloud[0].astype(np.float32)
        feats = pointcloud[1].astype(np.float32)
        labels = pointcloud[2].astype(np.int32)
        instances = pointcloud[3].astype(np.int32) 
        return coords, feats, labels, instances


    def __init__(self, config, phase, chunk=False):

        self.collate_fn = None
        self.chunk = chunk
        if chunk:
            self.collate_fn = collate_fn_factory(['pointcloud1', 'pointcloud2', 'feature3d1', 'feature3d2'])
        self.voxelizer = Voxelizer(voxel_size=0.02, ignore_label=0)

        self.config = config
        self.nclasses = 21
        self.image_size = config.size
        self.label_size = (self.image_size[0] // 2, self.image_size[1] // 2)
        self.depth_size = (self.image_size[0] // 1, self.image_size[1] // 1)
        self.intrinsics = get_scannet_intrinsic([self.image_size[1], self.image_size[0]])[:3,:3]
        self.path = config.path
        self.pointcloud_path = config.pointcloud_path

        self.dataset = []
        logging.info(f"Loading the subset {phase} from {self.path}")
        fnames = read_txt(os.path.join(self.path, 'splits', self.SPLITS[phase]))
        for fname in fnames:
            fname = fname.split()
            self.dataset.append([fname[0], fname[1]])

        self.transforms = {}
        self.transforms["rgb"] = transforms.Compose([
            transforms.Resize(self.image_size, Image.NEAREST),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
        ])

        self.transforms["depth"] = transforms.Compose([
            transforms.Resize(self.depth_size, Image.NEAREST),
            transforms.ToTensor(),
        ])

        print("images number in {}: {}".format(self.path, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def build_dataset(self):
        pass
    
    def __getitem__(self, index):
        fname1, fname2 = self.dataset[index]
        scene_id1 = fname1.split('/')[0]
        image_id1 = fname1.split('/')[2].split('.')[0]
        scene_id2 = fname2.split('/')[0]
        image_id2 = fname2.split('/')[2].split('.')[0]

        # -------------------------------- 2D ---------------------------------
        color_path1 = os.path.join(self.path, scene_id1, 'color', image_id1 + '.png')
        color_image1 = Image.open(color_path1)
        color_image1 = self.transforms["rgb"](color_image1)  # center crop, resize, normalize

        color_path2 = os.path.join(self.path, scene_id2, 'color', image_id2 + '.png')
        color_image2 = Image.open(color_path2)
        color_image2 = self.transforms["rgb"](color_image2)  # center crop, resize, normalize

        depth_path1 = os.path.join(self.path, scene_id1, 'depth', image_id1 + '.png')
        depth_image1 = Image.open(depth_path1)
        depth_image1 = self.transforms["depth"](depth_image1)
        depth_image1 = depth_image1 / 1000.0

        depth_path2 = os.path.join(self.path, scene_id2, 'depth', image_id2 + '.png')
        depth_image2 = Image.open(depth_path2)
        depth_image2 = self.transforms["depth"](depth_image2)
        depth_image2 = depth_image2 / 1000.0

        pose_path1 = os.path.join(self.path, scene_id1, 'pose', str(int(image_id1))+ '.txt')
        camera2world1 = torch.from_numpy(np.loadtxt(pose_path1)).float() # camera2world

        pose_path2 = os.path.join(self.path, scene_id1, 'pose', str(int(image_id2))+ '.txt')
        camera2world2 = torch.from_numpy(np.loadtxt(pose_path2)).float() # camera2world

        data_point = {'color1': color_image1, 'depth1': depth_image1[0],
                      'color2': color_image2, 'depth2': depth_image2[0],
                      'intrinsics1': torch.from_numpy(self.intrinsics).float(),
                      'intrinsics2': torch.from_numpy(self.intrinsics).float(),
                      'bbox1': torch.FloatTensor([-0.5,-0.5]), 'bbox2': torch.FloatTensor([-0.5,-0.5]),
                      'world2camera1': torch.inverse(camera2world1),
                      'world2camera2': torch.inverse(camera2world2)}


        if self.chunk:
            world_path1 = os.path.join(self.path, scene_id1, 'world', str(int(image_id1)))
            minx1, miny1, minz1, maxx1, maxy1, maxz1 = torch.load(world_path1)
            filepath1 = os.path.join(self.pointcloud_path, scene_id1 + '.pth')
            coords3d1, feature3d1, _, _ = ScanNet.load_data(filepath1)
            coords3d1 = torch.from_numpy(coords3d1).float()
            # crop
            mask_x1 = (coords3d1[:,0] > minx1) & (coords3d1[:,0] < maxx1)
            mask_y1 = (coords3d1[:,1] > miny1) & (coords3d1[:,1] < maxy1)
            mask_z1 = (coords3d1[:,2] > minz1) & (coords3d1[:,2] < maxz1)
            mask1 = mask_x1 & mask_y1 & mask_z1
            coords3d1 = coords3d1[mask1]
            feature3d1 = feature3d1[mask1]
            coords3d1 = coords3d1[None,:,:]
            feature3d1 = torch.from_numpy(feature3d1).float()[None,:,:]
            feature3d1 = feature3d1/255.0
            coords3d1, inds1, world2grid1 = self.voxelizer.voxelize(coords3d1[0])
            feature3d1 = feature3d1[0, inds1]

            data_point['pointcloud1'] = coords3d1
            data_point['feature3d1'] = feature3d1
            data_point['camera2world1'] = camera2world1
            data_point['world2grid1'] = world2grid1

            world_path2 = os.path.join(self.path, scene_id2, 'world', str(int(image_id2)))
            minx2, miny2, minz2, maxx2, maxy2, maxz2 = torch.load(world_path2)
            filepath2 = os.path.join(self.pointcloud_path, scene_id2 + '.pth')
            coords3d2, feature3d2, _, _ = ScanNet.load_data(filepath2)
            coords3d2 = torch.from_numpy(coords3d2).float()
            # crop
            mask_x2 = (coords3d2[:,0] > minx2) & (coords3d2[:,0] < maxx2)
            mask_y2 = (coords3d2[:,1] > miny2) & (coords3d2[:,1] < maxy2)
            mask_z2 = (coords3d2[:,2] > minz2) & (coords3d2[:,2] < maxz2)
            mask2 = mask_x2 & mask_y2 & mask_z2
            coords3d2 = coords3d2[mask2]
            feature3d2 = feature3d2[mask2]
            coords3d2 = coords3d2[None,:,:]
            feature3d2 = torch.from_numpy(feature3d2).float()[None,:,:]
            feature3d2 = feature3d2/255.0
            coords3d2, inds2, world2grid2 = self.voxelizer.voxelize(coords3d2[0])
            feature3d2 = feature3d2[0, inds2]

            data_point['pointcloud2'] = coords3d2
            data_point['feature3d2'] = feature3d2
            data_point['camera2world2'] = camera2world2
            data_point['world2grid2'] = world2grid2

        return data_point


if __name__ == "__main__":
    dataset = ScanNet(None, 'train')
    for data in dataset:
        print(data)

