import torch
import numpy as np
from torch.utils.data import Dataset

def checktype(obj):
  return bool(obj) and all(isinstance(elem, str) for elem in obj)

class collate_fn_factory:
  """Generates collate function for coords, feats, labels.
    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                       size so that the number of input coordinates is below limit_numpoints.
  """

  def __init__(self, keywords=['xyz']):
    self.keywords = keywords

  def __call__(self, list_data):
    data_dict = {}
    for batch_id, sample in enumerate(list_data):
      for name in sample:
        data = sample[name]
        if name in self.keywords:
          num_points = data.shape[0]
          batch_ids = torch.ones(num_points, 1) * batch_id
          data = np.concatenate([batch_ids, data], -1)

        if name not in data_dict:
          data_dict[name] = []
        data_dict[name].append(data)

    # Concatenate all lists
    for name in data_dict:
      if checktype(data_dict[name]):
        continue
      if name in self.keywords:
        data_dict[name] = torch.from_numpy(np.concatenate(data_dict[name], 0))
      elif name == 'id':
        data_dict[name] = data_dict[name]
      else:
        data_dict[name] = torch.from_numpy(np.stack(data_dict[name], 0))

    return data_dict

class collate_fn_factory_triplet:
  """Generates collate function for coords, feats, labels.
    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                       size so that the number of input coordinates is below limit_numpoints.
  """

  def __init__(self, keywords=['xyz']):
    self.keywords = keywords

  def __call__(self, list_data):
    data_dict = {'color1': [], 'color2': [], 'depth1': [], 'depth2': [], 
                 'id1': [], 'id2': [], 'camera2world1': [], 'camera2world2': []}
    for batch_id, sample in enumerate(list_data):
        data_dict['color1'].append(sample['color1'])
        data_dict['color2'].append(sample['color2'])
        data_dict['depth1'].append(sample['depth1'])
        data_dict['depth2'].append(sample['depth2'])
        data_dict['id1'].append(sample['id1'])
        data_dict['id2'].append(sample['id2'])
        data_dict['camera2world1'].append(sample['camera2world1'])
        data_dict['camera2world2'].append(sample['camera2world2'])

        data_dict['color1'].append(sample['color1'])
        data_dict['color2'].append(sample['color3'])
        data_dict['depth1'].append(sample['depth1'])
        data_dict['depth2'].append(sample['depth3'])
        data_dict['id1'].append(sample['id1'])
        data_dict['id2'].append(sample['id3'])
        data_dict['camera2world1'].append(sample['camera2world1'])
        data_dict['camera2world2'].append(sample['camera2world3'])

        data_dict['color1'].append(sample['color2'])
        data_dict['color2'].append(sample['color3'])
        data_dict['depth1'].append(sample['depth2'])
        data_dict['depth2'].append(sample['depth3'])
        data_dict['id1'].append(sample['id2'])
        data_dict['id2'].append(sample['id3'])
        data_dict['camera2world1'].append(sample['camera2world2'])
        data_dict['camera2world2'].append(sample['camera2world3'])

    # Concatenate all lists
    for name in data_dict:
      if checktype(data_dict[name]):
        continue
      if name in self.keywords:
        data_dict[name] = torch.from_numpy(np.concatenate(data_dict[name], 0))
      elif name == 'id':
        data_dict[name] = data_dict[name]
      else:
        data_dict[name] = torch.from_numpy(np.stack(data_dict[name], 0))

    return data_dict