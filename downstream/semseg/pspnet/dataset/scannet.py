import os
import os.path
import cv2,torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class ScanNet(Dataset):
    def __init__(self, split='train', data_root=None, transform=None):
        self.split = split
        self.transform = transform
        filelist = {
                'train': 'scannetv2_train.txt',
                'val': 'scannetv2_val.txt'}
        filelist = open(os.path.join(data_root, 'splits', filelist[split])).readlines()
        filelist = [sceneid.strip() for sceneid in filelist]

        self.data_list = []
        for sceneid in os.listdir(data_root):
            if (sceneid == 'splits') or (sceneid not in filelist):
                continue

            for imageid in os.listdir(os.path.join(data_root, sceneid, 'color')):
                imageid = imageid.split('.')[0]
                self.data_list.append([os.path.join(data_root, sceneid, 'color', imageid+'.jpg'), os.path.join(data_root, sceneid, 'label', imageid+'.png')])

        self.map ={0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,  9: 8,
                   10: 9, 11: 10, 12: 11, 13: 255, 14: 12, 15: 255, 16: 13, 17: 255,
                   18: 255, 19: 255, 20: 255, 21: 255, 22: 255, 23: 255, 24: 14, 25: 255, 26: 255,
                   27: 255, 28: 15, 29: 255, 30: 255, 31: 255, 32: 255, 33: 16, 34: 17, 
                   35: 255, 36: 18, 37: 255, 38: 255, 39: 19, 40: 255, 255:255}

    def map_func(self, label):
        lut = torch.zeros((256)).long()
        for key, data in self.map.items():
            lut[key] = data
        # do the mapping
        return lut[label]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        label = self.map_func(label)

        return image, label
