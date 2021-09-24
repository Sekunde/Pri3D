import os
import csv
import torch
import logging
import numpy as np
from PIL import Image
from random import random
from torch.utils.data import Dataset 
from torchvision.transforms import transforms
from torch import nn

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class ScanNetJitter(Dataset):
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

    PATH = '/checkpoint/jihou/data/scannet/partial_frames/'

    SPLITS = {
        'debug': 'debug.txt',
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
        self.nclasses = 21
        self.image_size = [240, 320]
        self.label_size = (self.image_size[0] // 2, self.image_size[1] // 2)
        self.depth_size = (self.image_size[0] // 1, self.image_size[1] // 1)

        self.image_ids = []
        logging.info(f"Loading the subset {phase} from {self.PATH}")
        fnames = read_txt(os.path.join(self.PATH, 'splits', self.SPLITS[phase]))
        for fname in fnames:
            fname = fname.split()
            self.image_ids.append([fname[0], fname[1]])

        if self.config.simclr:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, Image.NEAREST),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * 240)),
                transforms.ToTensor(),
                transforms.Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, Image.NEAREST),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
            ])

        print("images number in {}: {}".format(self.PATH, len(self.image_ids)))

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        fname1, fname2 = self.image_ids[index]
        scene_id1 = fname1.split('/')[0]
        image_id1 = fname1.split('/')[2].split('.')[0]
        scene_id2 = fname2.split('/')[0]
        image_id2 = fname2.split('/')[2].split('.')[0]

        # -------------------------------- 2D ---------------------------------
        color_path = os.path.join(self.PATH, scene_id1, 'color', image_id1 + '.png')
        color_image = Image.open(color_path)
        color_image1 = self.transform(color_image)  # center crop, resize, normalize
        color_image2 = self.transform(color_image)  # center crop, resize, normalize

        return {'color1': color_image1, 'color2': color_image2}
