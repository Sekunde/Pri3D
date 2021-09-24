import os
import csv
import torch
import logging
import random
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset 
from torchvision.transforms import transforms
from common.io3d import write_triangle_mesh

from dataset.scannet import ScanNet
from dataset.megadepth import MegaDepth

class Mixture(Dataset):

    def __init__(self, config, phase):
        self.collate_fn = None
        self.config = config
        self.dataset = []
        self.num_sample = []
        curr_sample = 0
        self.scannet = ScanNet(config, phase)
        self.megadepth = MegaDepth(config, phase)
        print("images number in Mixture: {}".format(len(self)))

    def __len__(self):
        return len(self.scannet) + len(self.megadepth)
    
    def build_dataset(self):
        self.scannet.build_dataset()
        self.megadepth.build_dataset()
    
    def __getitem__(self, index):
            if index < len(self.scannet):
                return self.scannet[index]
            elif index >= len(self.scannet):
                return self.megadepth[index-len(self.scannet)]


