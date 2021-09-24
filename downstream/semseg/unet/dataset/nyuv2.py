"""
author: Mihai Suteu
date: 15/05/19
"""
import os
import sys
import h5py
import torch
import random
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from dataset.collate_fn import collate_fn_factory

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class NYUv2(Dataset):
    """
    PyTorch wrapper for the NYUv2 dataset focused on multi-task learning.
    Data sources available: RGB, Semantic Segmentation, Surface Normals, Depth Images.
    If no transformation is provided, the image type will not be returned.

    ### Output
    All images are of size: 640 x 480
    Semantic Segmentation Classes: (0) background, (1) bed, (2) books, (3) ceiling, (4) chair, 
                                   (5) floor, (6) furniture, (7) objects, (8) painting, (9) sofa, 
                                   (10) table, (11) tv, (12) wall, (13) window

    1. RGB: 3 channel input image

    2. Semantic Segmentation: 1 channel representing one of the 14 (0 -
    background) classes. Conversion to int will happen automatically if
    transformation ends in a tensor.

    3. Surface Normals: 3 channels, with values in [0, 1].

    4. Depth Images: 1 channel with floats representing the distance in meters.
    Conversion will happen automatically if transformation ends in a tensor.
    """
    LEARNING_MAP ={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,  9: 9,
                   10: 10, 11: 11, 12: 12, 13: 13 }

    CONTENT = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
               8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0}


    LEARNING_IGNORE = {0: True, 1: False, 2: False, 3: False, 4: False, 5: False,
                       6: False, 7: False, 8: False, 9: False, 10: False,
                       11: False, 12: False, 13: False}

    def __init__(self, config, phase):

        """
        Will return tuples based on what data source has been enabled (rgb, seg etc).

        :param root: path to root folder (eg /data/NYUv2)
        :param train: whether to load the train or test set
        :param rgb_transform: the transformation pipeline for rbg images
        :param seg_transform: the transformation pipeline for segmentation images. If
        the transformation ends in a tensor, the result will be automatically
        converted to int in [0, 14)
        :param sn_transform: the transformation pipeline for surface normal images
        :param depth_transform: the transformation pipeline for depth images. If the
        transformation ends in a tensor, the result will be automatically converted
        to meters
        """
        super().__init__()
        self.config = config
        self.phase = phase
        self.collate_fn = None
        self.image_size = [240, 320]
        self.label_size = (self.image_size[0] // 1, self.image_size[1] // 1)
        self.nclasses = len(self.LEARNING_IGNORE)
        self.PATH = config.path

        self.transforms = {}
        self.transforms["rgb"] = Compose([
            Resize(self.image_size, Image.NEAREST),
            ToTensor(),
            Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
        ])

        self.transforms["label"] = Compose([
            Resize(self.label_size, Image.NEAREST),
            ToTensor(),
        ])

        self.image_ids = []
        for image_id in os.listdir(os.path.join(self.PATH, phase, 'color')):
            image_id = image_id.split('_')[-1].split('.')[0]
            self.image_ids.append(image_id)

        print("images number in {}: {}".format(self.PATH, len(self.image_ids)))

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        # -------------------------------- 2D ---------------------------------
        color_path = os.path.join(self.PATH, self.phase, 'color', 'nyu_rgb_{}.png'.format(image_id))
        color_image = Image.open(color_path)
        color_image = self.transforms["rgb"](color_image)  # center crop, resize, normalize

        label_path = os.path.join(self.PATH, self.phase, 'label', 'new_nyu_class13_{}.png'.format(image_id))
        label = Image.open(label_path)
        label = self.transforms["label"](label)
        label = (label * 255.0).long()

        return {'color': color_image, 'label2d': label[0]}

    def __len__(self):
        return len(self.image_ids)

