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

#(0) background, (1) bed, (2) books, (3) ceiling, (4) chair, 
#(5) floor, (6) furniture, (7) objects, (8) painting, (9) sofa, 
#(10) table, (11) tv, (12) wall, (13) window
NAME_MAP = {1: 'bed', 2: 'books', 3: 'chair', 4: 'furniture', 5: 'painting', 
            6: 'sofa', 7: 'table', 8: 'tv', 9: 'window'}

LEARNING_MAP = {0:0, 1:1, 2:2, 3:0, 4:3, 5:0, 6:4, 7:0, 8:5, 9:6, 10: 7, 11: 8, 12:0, 13: 9}



INFO = {
    "description": "NYUv2 Dataset",
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

def convert_mat_to_png(path='/checkpoint/jihou/data/nyuv2/nyu_depth_v2_labeled.mat'):
    import mat73
    data = mat73.loadmat(path)
    num_images = data['instances'].shape[2]
    for i in range(num_images):
        print(i)
        instance = data['instances'][:,:,i]
        instance = Image.fromarray(instance)
        instance.save("instance/{:04d}.png".format(i+1))

def visualize_instance_mask(label_path='test.png', instance_path='test1.png'):
    instance = Image.open(instance_path)
    instance = np.array(instance)
    label = Image.open(label_path)
    label = np.array(label)

    from io2d import write_to_label
    instance = label*1000 + instance
    instance_ids = np.unique(instance)

    for instance_id in instance_ids:
        new_array = np.array(instance, copy=True)
        mask = instance != instance_id
        new_array[mask] = 0
        label_id = int(instance_id  / 1000)
        write_to_label('{}.png'.format(instance_id), new_array)

def split_train_val(ref_path='/checkpoint/jihou/data/nyuv2/train/color/', input_path='./instance'):
    output_path = 'instance_'
    for image_id in os.listdir(ref_path):
        print(image_id)
        image_id = image_id.split('_')[-1].split('.')[0]
        os.system('mv {} {}'.format(os.path.join(input_path, image_id + '.png'), output_path))


def convert_nyu_to_coco(path, phase):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    image_ids = []
    for image_id in os.listdir(os.path.join(path, phase, 'instance')):
        image_ids.append(image_id)

    print("images number in {}: {}".format(path, len(image_ids)))

    coco_image_id = 1
    coco_ann_id = 1
    for index in range(len(image_ids)):
        print("{}/{}".format(index, len(image_ids)), end='\r')

        instance_path = os.path.join(path, phase, 'instance', image_ids[index])
        instance_map = Image.open(instance_path)
        image_size = instance_map.size
        instance_map = np.array(instance_map)

        label_path = os.path.join(path, phase, 'label', 'new_nyu_class13_' + image_ids[index])
        label_map = Image.open(label_path)
        label_map = np.array(label_map)

        ann_map = label_map*1000+instance_map
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
            image_filename = os.path.join(phase, 'color', 'nyu_rgb_' + image_ids[index])
            image_info = pycococreatortools.create_image_info(coco_image_id, image_filename, image_size)
            coco_output['images'].append(image_info)
            coco_image_id += 1

    json.dump(coco_output, open(f'nyu_{phase}.coco.json','w'))


def config():
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--nyu_path', default='/rhome/jhou/data/dataset/nyuv2')
    parser.add_argument('--phase', default='train')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = config()
    convert_nyu_to_coco(opt.nyu_path, opt.phase)
    #convert_mat_to_png()
    #visualize_instance_mask()
    #split_train_val()
