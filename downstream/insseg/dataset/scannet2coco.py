import os
import argparse
import datetime
import csv
import torch
import numpy as np
import json

from PIL import Image
from random import random
from pycococreatortools import pycococreatortools
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

LEARNING_MAP = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6,  9: 7,
                10: 8, 11: 9, 12: 10, 13: 0, 14: 11, 15: 0, 16: 12, 17: 0,
                18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 13, 25: 0, 26: 0,
                27: 0, 28: 14, 29: 0, 30: 0, 31: 0, 32: 0, 33: 15, 34: 16, 
                35: 0, 36: 17, 37: 0, 38: 0, 39: 18, 40: 0}

NAME_MAP = { 1: 'cabinet', 2: 'bed', 3: 'chair', 4: 'sofa', 5: 'table', 6: 'door', 7: 'window',
             8: 'bookshelf', 9: 'picture', 10: 'counter', 11: 'desk', 12: 'curtain', 13: 'refridgerator',
             14: 'shower curtain', 15: 'toilet', 16: 'sink', 17:' bathtub',  18: 'otherfurniture'}



SPLITS = {
    'train':   'scannet_splits/scannetv2_train.txt',
    'val':     'scannet_splits/scannetv2_val.txt',
    'train20': 'scannet_splits/data_efficient_by_images/scannetv2_train_20.txt',
    'train40': 'scannet_splits/data_efficient_by_images/scannetv2_train_40.txt',
    'train60': 'scannet_splits/data_efficient_by_images/scannetv2_train_60.txt',
    'train80': 'scannet_splits/data_efficient_by_images/scannetv2_train_80.txt',
}

INFO = {
    "description": "ScanNet Dataset",
    "url": "https://github.com/sekunde",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "Ji Hou",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in NAME_MAP.items() 
]

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines

def convert_scannet_to_coco(path, phase):
    transform = Resize([240,320], Image.NEAREST)
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # get list
    scene_ids = read_txt(SPLITS[phase])
    image_ids = []
    for scene_id in scene_ids:
        for image_id in os.listdir(os.path.join(path, scene_id, 'color')):
            image_ids.append(os.path.join(scene_id, image_id.split('.')[0]))
    print("images number in {}: {}".format(path, len(image_ids)))

    coco_image_id = 1
    coco_ann_id = 1
    for index in range(len(image_ids)):
        print("{}/{}".format(index, len(image_ids)), end='\r')

        scene_id = image_ids[index].split('/')[0]
        image_id = image_ids[index].split('/')[1]

        ann_path = os.path.join(path, scene_id, 'instance', image_id + '.png')
        ann_map = Image.open(ann_path)
        image_size = ann_map.size
        #ann_map = transform(ann_map)
        ann_map = np.array(ann_map)

        ann_ids = np.unique(ann_map)
        has_ann = False
        for ann_id in ann_ids:
            label_id = LEARNING_MAP[int(ann_id / 1000)]
            inst_id = int(ann_id % 1000)
            if label_id == 0:
                continue

            category_info = {'id': label_id, 'is_crowd': 0}
            binary_mask = (ann_map == ann_id).astype(np.uint8)
            mask_size = binary_mask.sum()

            if mask_size < 1000:
                continue

            ann_info = pycococreatortools.create_annotation_info(
                coco_ann_id, coco_image_id, category_info, binary_mask,
                image_size, tolerance=0)

            if ann_info is not None:
                coco_output['annotations'].append(ann_info)
                has_ann = True
                coco_ann_id += 1

        if has_ann:
            image_filename = os.path.join(scene_id, 'color', image_id + '.jpg')
            image_info = pycococreatortools.create_image_info(coco_image_id, image_filename, image_size)
            coco_output['images'].append(image_info)
            coco_image_id += 1

    json.dump(coco_output, open(f'scannet_{phase}.coco.json','w'))


def config():
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannet_path', default='/rhome/jhou/data/dataset/scannet/rgbd')
    parser.add_argument('--phase', default='train')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = config()
    convert_scannet_to_coco(opt.scannet_path, opt.phase)
