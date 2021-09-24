#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.
import torch

class iouEval(object):
  def __init__(self, n_classes, ignore=None):
    self.n_classes = n_classes
    # if ignore is larger than n_classes, consider no ignoreIndex
    self.ignore = torch.tensor(ignore).long()
    self.include = torch.tensor([n for n in range(self.n_classes) if n not in self.ignore]).long()
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)
    self.reset()

  def reset(self):
    self.conf_matrix = torch.zeros((self.n_classes, self.n_classes)).long()
    self.ones = None
    self.last_scan_size = None  # for when variable scan size is used

  def update(self, pred, gt): 
    # sizes should be "batch_size x H x W"
    row = pred.reshape(-1)  # de-batchify
    col = gt.reshape(-1)    # de-batchify

    # idxs are labels and predictions
    idxs = torch.stack([row, col], dim=0)
    ones = torch.ones((idxs.shape[-1])).long()

    # make confusion matrix (rows = pred, cols = gt)
    self.conf_matrix = self.conf_matrix.index_put_(tuple(idxs), ones, accumulate=True)

  def getStats(self):
    # remove fp and fn from confusion on the ignore classes cols and rows
    conf = self.conf_matrix.clone().double()
    conf[self.ignore] = 0
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diag()
    fp = conf.sum(dim=1) - tp
    fn = conf.sum(dim=0) - tp
    return tp, fp, fn

  def evaluate(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean#, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"




if __name__ == "__main__":
  # make evaluator
  eval = iouEval(nclasses, ignore)
  # run
  eval.update(pred, gt)
  m_iou, iou = eval.getIoU()
  m_acc = eval.getacc()