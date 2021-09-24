import os
import csv
import torch
import numpy as np
from PIL import Image
from random import random
from torch.utils.data import Dataset 
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from common.camera_intrinsics import adjust_intrinsic
from dataset.collate_fn import collate_fn_factory

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


class ScanNet(Dataset):
    LEARNING_MAP ={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,  9: 9,
                   10: 10, 11: 11, 12: 12, 13: 0, 14: 13, 15: 0, 16: 14, 17: 0,
                   18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 15, 25: 0, 26: 0,
                   27: 0, 28: 16, 29: 0, 30: 0, 31: 0, 32: 0, 33: 17, 34: 18, 
                   35: 0, 36: 19, 37: 0, 38: 0, 39: 20, 40: 0}

    CONTENT = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
               8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 
               15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0,
               22: 1.0, 23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0,
               29: 1.0, 30: 1.0, 31: 1.0, 32: 1.0, 33: 1.0, 34: 1.0, 35: 1.0,
               36: 1.0, 37: 1.0, 38: 1.0, 39: 1.0, 40: 1.0}
    LEARNING_IGNORE = {0: True, 1: False, 2: False, 3: False, 4: False, 5: False,
                       6: False, 7: False, 8: False, 9: False, 10: False,
                       11: False, 12: False, 13: False, 14: False, 15: False, 
                       16: False, 17: False, 18: False, 19: False, 20: False}   


    SPLITS = {
        'train':    'scannet_splits/scannetv2_train.txt',
        'trainval': 'scannet_splits/scannetv2_trainval.txt',
        'val':      'scannet_splits/scannetv2_val.txt',
        'test':     'scannet_splits/scannetv2_test.txt',
        'train1':   'scannet_splits/data_efficient_by_images/scannetv2_train_1.txt',
        'train5':   'scannet_splits/data_efficient_by_images/scannetv2_train_5.txt',
        'train10':  'scannet_splits/data_efficient_by_images/scannetv2_train_10.txt',
        'train20':  'scannet_splits/data_efficient_by_images/scannetv2_train_20.txt',
        'train40':  'scannet_splits/data_efficient_by_images/scannetv2_train_40.txt',
        'train60':  'scannet_splits/data_efficient_by_images/scannetv2_train_60.txt',
        'train80':  'scannet_splits/data_efficient_by_images/scannetv2_train_80.txt',
        }

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if key > maxkey:
                maxkey = key

        # +100 hack making lut bigger just in case there are unknown labels
        lut = torch.zeros((maxkey + 100)).long()
        for key, data in mapdict.items():
            lut[key] = data
        # do the mapping
        return lut[label]

    def __init__(self, config, phase):

        self.collate_fn = None
        self.config = config
        self.PATH = config.path
        self.nclasses = 21
        self.image_size = [240, 320]
        self.depth_size = (self.image_size[0] // 1, self.image_size[1] // 1)
        self.label_size = (self.image_size[0] // 1, self.image_size[1] // 1)
        current_file_path = os.path.dirname(os.path.abspath(__file__))

        import ipdb
        ipdb.set_trace()

        scene_ids = read_txt(os.path.join(current_file_path, self.SPLITS[phase]))
        self.image_ids = []
        for scene_id in scene_ids:
            for image_id in os.listdir(os.path.join(self.PATH, scene_id, 'color')):
                self.image_ids.append(os.path.join(scene_id, image_id.split('.')[0]))

        self.transforms = {}
        self.transforms["rgb"] = Compose([
            Resize(self.image_size, Image.NEAREST),
            ToTensor(),
            Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
        ])

        self.transforms["depth"] = Compose([
            Resize(self.depth_size, Image.NEAREST),
            ToTensor(),
        ])

        self.transforms["label"] = Compose([
            Resize(self.label_size, Image.NEAREST),
            ToTensor(),
        ])

        print("images number in {}: {}".format(self.PATH, len(self.image_ids)))

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        scene_id = self.image_ids[index].split('/')[0]
        image_id = self.image_ids[index].split('/')[1]

        # -------------------------------- 2D ---------------------------------
        color_path = os.path.join(self.PATH, scene_id, 'color', image_id + '.jpg')
        color_image = Image.open(color_path)
        color_image = self.transforms["rgb"](color_image)  # center crop, resize, normalize

        depth_path = os.path.join(self.PATH, scene_id, 'depth', image_id + '.png')
        depth_image = Image.open(depth_path)
        depth_image = self.transforms["depth"](depth_image)
        depth_image = depth_image / 1000.0

        label_path = os.path.join(self.PATH, scene_id, 'label', image_id + '.png')
        label = Image.open(label_path)
        label = self.transforms["label"](label)
        label = (label * 255.0).long()
        label = ScanNet.map(label, self.LEARNING_MAP)

        return {'color': color_image, 'depth': depth_image[0], 'label2d': label[0], 'id': self.image_ids[index]}
