import numpy as np
import torch

class maeEval(object):
  def __init__(self, invalid_value=0):
      self.invalid_value = invalid_value
      self.reset()

  def reset(self):
      self.sum = 0
      self.num = 0

  def update(self, pred, gt): 
    # sizes should be "batch_size x 1 x H x W"
    batch_size = gt.shape[0]
    self.num += batch_size
    for i in range(batch_size):
      pred_batch = pred[i].reshape(-1)  # de-batchify
      gt_batch = gt[i].reshape(-1)    # de-batchify
      mask = (gt_batch != self.invalid_value)
      mae = torch.abs(pred_batch[mask] - gt_batch[mask]).sum() / mask.sum()
      self.sum += mae.item()

  def evaluate(self):
    # remove fp and fn from confusion on the ignore classes cols and rows
    return self.sum / self.num  # returns "iou mean", "iou per class" ALL CLASSES


