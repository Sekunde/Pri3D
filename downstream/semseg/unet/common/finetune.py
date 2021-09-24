#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import time
import copy
import cv2
import os
import wandb
import numpy as np
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler
from torch.serialization import default_restore_location

import MinkowskiEngine as ME

from dataset import load_dataset
from model import load_model
from common.io2d import write_to_depth, write_to_rgb, write_to_label
from common.iou import iouEval
from common.mae import maeEval
from common.avgmeter import AverageMeter
from common.solver import initialize_optimizer, initialize_scheduler, initialize_bnm_scheduler

from common.distributed import is_master_proc

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_state_with_same_shape(model, weights):
    model_state = model.state_dict()
    filtered_weights = {
          k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
      }
    return filtered_weights

class Finetune():
  def __init__(self, config):
    # parameters
    self.config = config
    self.is_master = is_master_proc(config.distributed.num_gpus)
    # put logger where it belongs
    if self.is_master:
      logging.info(config.pretty())
      wandb.init(project="pri3d", name=config.train.exp_name, config=config, reinit=True)
    
    self.init_dataset()
    self.init_model()
    self.init_optimizer()

    if self.is_master:
      self.init_meter()

  def init_optimizer(self):
    self.optimizer = initialize_optimizer(self.model.parameters(), self.config)
    self.scheduler = initialize_scheduler(self.optimizer, self.config, len(self.trainloader))
    self.bnm_scheduler = initialize_bnm_scheduler(self.model)

  def init_meter(self):
    self.loss = AverageMeter()
    self.iou =  AverageMeter()

    # for validation
    self.best_iou = -1e9

  def init_dataset(self):
    # get the data
    DatasetClass = load_dataset(self.config.dataset.name)
    train_dataset = DatasetClass(config=self.config.dataset, phase=self.config.train.phase)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.config.distributed.num_gpus > 1 else None
    self.trainloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=self.config.train.batch_size,
                                                shuffle=(train_sampler is None),
                                                sampler=train_sampler,
                                                num_workers=self.config.train.workers,
                                                #num_workers=0,
                                                collate_fn=train_dataset.collate_fn)

    val_dataset = DatasetClass(config=self.config.dataset, phase='val')
    self.valloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.config.train.workers,
                                                 collate_fn=train_dataset.collate_fn)

    # weights for loss (and bias)
    content = torch.zeros(self.trainloader.dataset.nclasses, dtype=torch.float)
    for cl, freq in self.trainloader.dataset.CONTENT.items():
      x_cl = self.trainloader.dataset.LEARNING_MAP[cl]  # map actual class to xentropy class
      content[x_cl] += freq
    self.loss_w = 1 / (content + 1e-3)   # get weights
    for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
      if train_dataset.LEARNING_IGNORE[x_cl]:
        # don't weigh
        self.loss_w[x_cl] = 0
    logging.info("Loss weights from content: {}".format(self.loss_w.data))
    self.loss_w = self.loss_w.cuda()
    self.ignore_class = []
    for i, w in enumerate(self.loss_w):
      if w < 1e-10:
        self.ignore_class.append(i)
        logging.info("Ignoring class {} in IoU evaluation".format(i))
  
  def init_model(self):
    # concatenate the encoder and the head
    ModelClass = load_model(self.config.finetune.model)
    self.model = ModelClass(self.config, self.trainloader.dataset.nclasses, self.loss_w)
    logging.info('===> Number of trainable parameters: {}: {}'.format(ModelClass.__name__, count_parameters(self.model)))
    # load pretrained model here
    if self.config.finetune.pretrain != 'imagenet' and self.config.finetune.pretrain != 'scratch':
        logging.info('==========> Loading weights: ' + self.config.finetune.pretrain)
        state = torch.load(self.config.finetune.pretrain)

        if 'model' in state.keys():
          matched_weights = load_state_with_same_shape(self.model, state['model'])
        else:
          matched_weights = load_state_with_same_shape(self.model, state)
          
        self.model.load_state_dict(matched_weights, strict=False)
        if bool(matched_weights):
          logging.info("===> Loaded weights: {}".format(self.config.finetune.pretrain))

    # GPU
    self.model.cuda()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training in device: {}".format(self.device))

    if self.config.distributed.num_gpus > 1:
      self.cur_device = torch.cuda.current_device()
      self.model = torch.nn.parallel.DistributedDataParallel(
        module=self.model, device_ids=[self.cur_device], 
        output_device=self.cur_device) 

      if '3D' in self.config.finetune.backbone:
        self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model)

  def train(self):
    # train for n epochs
    for epoch in range(self.config.scheduler.max_epochs):
      # train for 1 epoch
      self.train_epoch(epoch=epoch)

      if epoch % self.config.train.report_epoch == 0 and self.is_master:
        iou, loss = self.validate()
        wandb.log({'val/loss': loss, 
                   'val/iou': iou}, step=(epoch+1)*len(self.trainloader))

        if iou > self.best_iou:
          self.best_iou = iou
          self.save_checkpoint((epoch+1)*len(self.trainloader))

    logging.info('Finished Training')
  
  def save_checkpoint(self, step):
    # Save the weights
    os.makedirs('models', exist_ok=True)
    logging.info('---> save checkpoint')
    save_dict = {'step': step, 'best_iou': self.best_iou}

    if self.config.distributed.num_gpus > 1:
      save_dict['model'] = self.model.module.state_dict()
      torch.save(save_dict, 'models/checkpoint_finetune.pth')
    else:
      save_dict['model'] = self.model.state_dict()
      torch.save(save_dict, 'models/checkpoint_finetune.pth')

  def save2d(self, color, output, target, iteration):
    os.makedirs('visuals', exist_ok=True)
    _imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    batch_size = color.shape[0]
    for i in range(batch_size):
      color_write = color[i].permute(1,2,0).cpu().numpy()
      for ii in range(3):
        color_write[:,:,ii] = color_write[:,:,ii] * _imagenet_stats['std'][ii] + _imagenet_stats['mean'][ii]
      write_to_rgb('visuals/{}_rgb.png'.format(iteration*batch_size+i), color_write)
      write_to_label('visuals/{}_gt.png'.format(iteration*batch_size+i), target[i])
      write_to_label('visuals/{}_pred.png'.format(iteration*batch_size+i), output[i])
  
  def train_epoch(self, epoch):
    # empty the cache to train now
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    evaluator = iouEval(self.trainloader.dataset.nclasses, 0)

    # switch to train mode
    self.model.train()
    end = time.time()
    if self.config.distributed.num_gpus > 1:
      logging.info('reset sampler here')
      self.trainloader.sampler.set_epoch(epoch)
    for i, data in enumerate(self.trainloader):
      # measure data loading time
      data_time.update(time.time() - end)

      # train model
      output, label, loss = self.model(data)

      # compute gradient and do SGD step
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      self.scheduler.step()
      self.bnm_scheduler.step() # decay BN momentum

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if self.is_master:
        evaluator.reset()
        evaluator.update(output.cpu(), label.cpu())

        self.iou.update(evaluator.evaluate())
        self.loss.update(loss.item())

        if i % self.config.train.report_batch == 0:
          logging.info('Lr: {lr:.3e} | '
            'Epoch: [{0}][{1}/{2}] | '
            'Time {batch_time:.3f} | '
            'Data {data_time:.3f} | '
            'Loss {loss:.4f} | '
            'IoU {iou:.3f}'.format(
                epoch, i, len(self.trainloader), batch_time=batch_time.avg,
                data_time=data_time.avg, loss=self.loss.avg,
                iou=self.iou.avg, lr=self.scheduler.get_last_lr()[0]))

          wandb.log({'train/loss': self.loss.avg, 
                    'train/iou': self.iou.avg,
                    'train/lr': self.scheduler.get_last_lr()[0]},
                    step=epoch*len(self.trainloader)+i)


  def validate(self):
    batch_time = AverageMeter()
    loss_avg = AverageMeter()
    evaluator = iouEval(self.trainloader.dataset.nclasses, 0)

    # switch to evaluate mode
    self.model.eval()

    # empty the cache to infer in high res
    torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      for i, data in enumerate(self.valloader):
        logging.info("{}/{}".format(i, len(self.valloader)))
        output, label, loss = self.model(data)
        loss_avg.update(loss.item())
        evaluator.update(output.cpu(), label.cpu())

        if self.config.train.write_result:
          self.save2d(color, output, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

      iou = evaluator.evaluate()
      logging.info('Validation set:\n'
            'Time avg per batch {batch_time:.3f}\n'
            'Loss avg {loss:.4f}\n'
            'IoU avg {iou:.3f}'.format(batch_time=batch_time.avg,
                                           loss=loss_avg.avg, 
                                           iou=iou))
    
    return iou, loss_avg.avg


  def test(self):
    # switch to evaluate mode
    self.model.eval()
    evaluator = iouEval(self.valloader.dataset.nclasses, 0)

    with torch.no_grad():
      for i, data in enumerate(self.valloader):
        logging.info("{}/{}".format(i, len(self.valloader)))
        output, label, loss = self.model(data)
        evaluator.update(output.cpu(), label.cpu())

      iou = evaluator.evaluate()
      logging.info('Validation set:\n'
                   'IoU avg {iou:.3f}'.format(iou=iou))
        #self.save2d(data['color'], output, label, i)
