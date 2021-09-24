import os
import torch
import numpy as np
from PIL import Image
import scipy.misc as m
from torch.utils import data
from dataset.transform2d import RandomRotation, RandomHorizontalFlip
from dataset.transform2d import Compose as Compose_
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class KITTI(data.Dataset):

    LEARNING_MAP = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 
                    6: 0, 9: 0, 10: 0, 14: 0, 15: 0, 
                    16: 0, 18: 0, 29: 0, 30: 0, -1: 0, 
                    7:1, 8:2, 11:3, 12: 4, 13: 5, 17: 6, 19: 7, 20: 8, 
                    21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 
                    27: 15, 28: 16, 31: 17, 32: 18, 33: 19}

    CONTENT = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
               8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 
               15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0,
               22: 1.0, 23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0,
               29: 1.0, 30: 1.0, 31: 1.0, 32: 1.0, 33: 1.0, -1: 1.0}

    LEARNING_IGNORE = {0: True, 1: False, 2: False, 3: False, 4: False, 5: False,
                       6: False, 7: False, 8: False, 9: False, 10: False,
                       11: False, 12: False, 13: False, 14: False, 15: False, 
                       16: False, 17: False, 18: False, 19: False}   

    MAPPED_NAMES = ["unlabelled", "road", "sidewalk", "building", "wall", "fence", "pole",
                    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky", "person", "rider",
                    "car", "truck", "bus", "train", "motorcycle", "bicycle"]

    COLORS = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def __init__(
        self,
        config,
        phase="train",
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """


        self.PATH = config.path
        self.annotations_base = os.path.join(self.PATH, "semantic")
        images_base = os.path.join(self.PATH, "image_2")

        files = sorted(recursive_glob(rootdir=images_base, suffix=".png"))
        if phase == 'train':
            self.files = files[40:]
        elif phase == 'val':
            self.files = files[:40]
        self.config = config

        self.nclasses = 20
        self.phase = phase
        self.collate_fn = None
        self.augmentations = Compose_([RandomRotation(10), RandomHorizontalFlip(0.5)])
        #self.IMAGE_SIZE = [374, 1240]
        self.IMAGE_SIZE = config.size

        self.transforms = {}
        if config.resize and not config.random_crop:
            self.transforms["rgb"] = Compose([
                Resize(self.IMAGE_SIZE, Image.NEAREST),
                ToTensor(),
                Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
            ])

            self.transforms["label"] = Compose([
                Resize(self.IMAGE_SIZE, Image.NEAREST),
                ToTensor(),
            ])
        elif not config.resize and not config.random_crop:
            self.transforms["rgb"] = Compose([
                CenterCrop(self.IMAGE_SIZE),
                ToTensor(),
                Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
            ])

            self.transforms["label"] = Compose([
                CenterCrop(self.IMAGE_SIZE),
                ToTensor(),
            ])
        else:
            self.transforms["rgb"] = Compose([
                ToTensor(),
                Normalize(_imagenet_stats['mean'], std=_imagenet_stats['std'])
            ])

            self.transforms["label"] = Compose([
                ToTensor(),
            ])


        print("Found %d %s images" % (len(self.files), phase))

    def __len__(self):
        """__len__"""
        return len(self.files)

    def crop(self, image1, image2, central_match):
        bbox_i = max(int(central_match[0] - self.IMAGE_SIZE[0] // 2), 0)
        if bbox_i + self.IMAGE_SIZE[0] >= image1.shape[1]:
            bbox_i = image1.shape[1] - self.IMAGE_SIZE[0]
        bbox_j = max(int(central_match[1] - self.IMAGE_SIZE[1] // 2), 0)
        if bbox_j + self.IMAGE_SIZE[1] >= image1.shape[2]:
            bbox_j = image1.shape[2] - self.IMAGE_SIZE[1]

        return (
                image1[:,
                bbox_i : bbox_i + self.IMAGE_SIZE[0],
                bbox_j : bbox_j + self.IMAGE_SIZE[1]],
            image2[:,
                bbox_i : bbox_i + self.IMAGE_SIZE[0],
                bbox_j : bbox_j + self.IMAGE_SIZE[1]])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path))


        img = Image.open(img_path)
        img = self.transforms['rgb'](img)

        lbl = Image.open(lbl_path)
        lbl = self.transforms['label'](lbl)
        lbl = (lbl * 255.0).long()
        lbl = KITTI.map(lbl, self.LEARNING_MAP)

        if self.phase == 'train':
            img, lbl = self.augmentations(img, lbl)

        if self.config.random_crop and self.phase=='train':
            central_match = [np.random.choice(self.IMAGE_SIZE[0]), np.random.choice(self.IMAGE_SIZE[1])]
            img, lbl = self.crop(img, lbl, central_match)

        return {'color': img, 'label2d': lbl[0]}

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


