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

class ScanNet(Dataset):
    PATH = '/checkpoint/jihou/data/scannet/partial_frames/'
    SPLITS = {
        'train': 'overlap30.txt',
        'val': 'overlap30_val.txt',
        }

    def __init__(self, transform):

        self.image_size = [240, 320]

        self.image_ids = []
        fnames = read_txt(os.path.join(self.PATH, 'splits', self.SPLITS['train']))
        for fname in fnames:
            fname = fname.split()
            self.image_ids.append([fname[0], fname[1]])

        self.transform = transform
        self.resize = transforms.Resize(self.image_size, Image.NEAREST)

        print("images number in {}: {}".format(self.PATH, len(self.image_ids)))

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        fname, _ = self.image_ids[index]
        scene_id = fname.split('/')[0]
        image_id = fname.split('/')[2].split('.')[0]
        color_path = os.path.join(self.PATH, scene_id, 'color', image_id + '.png')
        color_image = Image.open(color_path)
        color_image = self.transform(self.resize(color_image))

        return color_image, 0
