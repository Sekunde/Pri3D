import torch
from torch.utils.data import Dataset
from dataset.scannet import ScanNet
from dataset.nyuv2 import NYUv2
from dataset.cityscapes import Cityscapes
from dataset.kitti import KITTI

DATASETS = [ScanNet, NYUv2, Cityscapes, KITTI]

def load_dataset(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  mdict = {dataset.__name__: dataset for dataset in DATASETS}
  if name not in mdict:
    print('Invalid dataset index. Options are:')
    # Display a list of valid dataset names
    for dataset in DATASETS:
      print('\t* {}'.format(dataset.__name__))
    raise ValueError(f'Dataset {name} not defined')
  DatasetClass = mdict[name]

  return DatasetClass
