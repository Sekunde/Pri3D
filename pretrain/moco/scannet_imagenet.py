
import os
import csv
import torch
import logging
import numpy as np
from PIL import Image
from random import random
from torch.utils.data import Dataset 
from torchvision.transforms import transforms

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines

class ScanNetImageNet(Dataset):
    SCANNET_PATH = '/checkpoint/jihou/data/scannet/partial_frames/'
    IMAGENET_PATH='/datasets01/imagenet_full_size/061417/train'
    IMAGENET_FILELIST = '/private/home/jihou/imagenet.txt'
    SPLITS = {
        'train': 'overlap30.txt',
        'val': 'overlap30_val.txt',
        }

    def __init__(self, transform):
        # imagenet images
        self.image_ids = open(self.IMAGENET_FILELIST).readlines()
        for idx in range(len(self.image_ids)):
            self.image_ids[idx] = [self.image_ids[idx].strip(), 1]

        # scannet images
        fnames = read_txt(os.path.join(self.SCANNET_PATH, 'splits', self.SPLITS['train']))
        for fname in fnames:
            fname = fname.split()
            self.image_ids.append([fname[0], 0])

        # transforms
        self.image_size = [240, 320]
        self.transform = transform
        self.resize = transforms.Resize(self.image_size, Image.NEAREST)
        print("images number in {}: {}".format(self.IMAGENET_PATH, len(self.image_ids)))

    def __len__(self):
        return len(self.image_ids)
    
    def get_label(self):
        labels = []
        for ind in range(len(self.image_ids)):
            _, labelid = self.image_ids[ind]
            labels.append(labelid)
        return labels
    
    def __getitem__(self, index):
        fname, labelid = self.image_ids[index]
        if labelid == 0:
            scene_id = fname.split('/')[0]
            image_id = fname.split('/')[2].split('.')[0]
            color_path = os.path.join(self.SCANNET_PATH, scene_id, 'color', image_id + '.png')
            color_image = Image.open(color_path)
            color_image = self.resize(color_image)
        elif labelid == 1:
            color_path = fname
            color_image = Image.open(os.path.join(self.IMAGENET_PATH, color_path))
        else:
            raise NotImplementedError

        color_image = self.transform(color_image)

        return color_image, labelid
