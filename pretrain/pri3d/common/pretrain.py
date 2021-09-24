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
from common.avgmeter import AverageMeter
from common.solver import initialize_optimizer, initialize_scheduler, initialize_bnm_scheduler

from common.distributed import is_master_proc, get_world_size

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_state_with_same_shape(model, weights):
    logging.info("Loading weights:" + ', '.join(weights.keys()))
    model_state = model.state_dict()
    filtered_weights = {
          k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
      }
    logging.info("Loaded weights:" + ', '.join(filtered_weights.keys()))
    return filtered_weights

class Pretrain():
  def __init__(self, config):
    # parameters
    self.config = config
    self.is_master = is_master_proc(config.distributed.num_gpus)
    self.initialized = False
    self.epoch = 0

    # put logger where it belongs
    if self.is_master:
      logging.info(config.pretty())
      wandb.init(project="Pri3D", name=config.train.exp_name, config=config)
    
    self.init_dataset()
    self.init_model()
    self.init_optimizer()
    self.resume_checkpoint()

  
  def resume_checkpoint(self, checkpoint_filename='models/checkpoint.pth'):
    if os.path.isfile(checkpoint_filename):
        logging.info('===> Loading existing checkpoint')
        state = torch.load(checkpoint_filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))

        # load weights
        model = self.model
        if get_world_size() > 1:
          model = self.model.module
        matched_weights = load_state_with_same_shape(model, state['model'])
        model.load_state_dict(matched_weights, strict=False)

        self.epoch = state['step']
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])

  def init_optimizer(self):
    self.optimizer = initialize_optimizer(self.model.parameters(), self.config)
    self.scheduler = initialize_scheduler(self.optimizer, self.config, len(self.trainloader)/self.config.optimizer.accumulate_step)
    self.bnm_scheduler = initialize_bnm_scheduler(self.model)

  def init_dataset(self):
    # get the data
    DatasetClass = load_dataset(self.config.dataset.name)
    train_dataset = DatasetClass(config=self.config.dataset, phase=self.config.train.phase, chunk=self.config.pretrain.geometric_prior)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if self.config.distributed.num_gpus > 1 else None
    self.trainloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=self.config.train.batch_size // self.config.distributed.num_gpus,
                                                shuffle=(train_sampler is None),
                                                #shuffle=False,
                                                sampler=train_sampler,
                                                num_workers=self.config.train.batch_size // self.config.distributed.num_gpus,
                                                #num_workers=0,
                                                collate_fn=train_dataset.collate_fn,
                                                drop_last=True)

  def init_model(self):
    # concatenate the encoder and the head
    ModelClass = load_model(self.config.pretrain.model)
    self.model = ModelClass(self.config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(ModelClass.__name__, count_parameters(self.model)))
    # GPU
    self.cur_device = torch.cuda.current_device()
    logging.info("Training in device: {}".format(self.cur_device))
    self.model.to(self.cur_device)

    if self.config.distributed.num_gpus > 1:
      #self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.cur_device)
      self.model = torch.nn.parallel.DistributedDataParallel(
        module=self.model, device_ids=[self.cur_device], output_device=self.cur_device) 

  def train(self):
    for epoch in range(self.epoch, self.config.scheduler.max_epochs):
      # train for 1 epoch
      self.trainloader.dataset.build_dataset()
      self.train_epoch(epoch=epoch)
      if epoch % self.config.train.report_epoch == 0 and self.is_master:
        self.save_checkpoint(epoch+1)
    logging.info('Finished Training')
  
  @staticmethod
  def logging_string(string_, loss):
    for key in loss:
      string_ += ' | {0} {1:.3f}'.format(key, loss[key].avg)
    return string_

  @staticmethod
  def update_wandb_dict(phase, dict_, loss):
    for key in loss:
      dict_['{}/{}'.format(phase, key)] =  loss[key].avg
    return dict_

  @staticmethod
  def update_loss(loss, losses):
    for key in loss:
      losses[key].update(loss[key])
  
  @staticmethod
  def init_loss(loss_keys):
    # accuracy and IoU stuff
    loss = {}
    for key in loss_keys:
      loss[key] = AverageMeter()
    return loss
  
  def save_checkpoint(self, step):
    # Save the weights
    os.makedirs('models', exist_ok=True)
    logging.info('---> save checkpoint')
    save_dict = {'step': step,
                 'optimizer': self.optimizer.state_dict(), 
                 'scheduler': self.scheduler.state_dict()}

    if self.config.distributed.num_gpus > 1:
      save_dict['model'] = self.model.module.state_dict()
      torch.save(save_dict, 'models/checkpoint{}.pth'.format(step))
      if os.path.exists("models/checkpoint.pth"):
          os.remove("models/checkpoint.pth")
      os.system("ln -s checkpoint{}.pth models/checkpoint.pth".format(step))
    else:
      save_dict['model'] = self.model.state_dict()
      torch.save(save_dict, 'models/checkpoint{}.pth'.format(step))
      if os.path.exists("models/checkpoint.pth"):
          os.remove("models/checkpoint.pth")
      os.system("ln -s checkpoint{}.pth models/checkpoint.pth".format(step))

  
  def train_epoch(self, epoch):
    # empty the cache to train now
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    self.model.train()
    self.model.zero_grad()

    end = time.time()
    data_time_batch = 0
    total_time_batch = 0

    if self.config.distributed.num_gpus > 1:
      self.trainloader.sampler.set_epoch(epoch)

    batch_loss = {}
    for i, data in enumerate(self.trainloader):
      # measure data loading time
      data_time_batch += (time.time() - end)
      if (i+1) % self.config.optimizer.accumulate_step == 0:
          data_time.update(data_time_batch)
          data_time_batch = 0

      # train model
      _, loss = self.model(data)
      total_loss = loss['total_loss'] / self.config.optimizer.accumulate_step
      total_loss.backward()
      for key in loss:
          if key not in batch_loss:
              batch_loss[key] = 0.0
          batch_loss[key] += (loss[key].item() / self.config.optimizer.accumulate_step)

      if (i+1) % self.config.optimizer.accumulate_step == 0:
          self.optimizer.step()
          self.model.zero_grad()
          self.scheduler.step()
          self.bnm_scheduler.step() # decay BN momentum

      # measure elapsed time
      total_time_batch += time.time() - end
      if (i+1) % self.config.optimizer.accumulate_step == 0:
          batch_time.update(total_time_batch)
          total_time_batch = 0

      end = time.time()

      if self.is_master and (i+1) % self.config.optimizer.accumulate_step == 0:
        if not self.initialized:
          self.loss = Pretrain.init_loss(loss.keys())
          self.initialized = True

        Pretrain.update_loss(batch_loss, self.loss)
        batch_loss = {}

        if ((i+1)/self.config.optimizer.accumulate_step) % self.config.train.report_batch == 0:
          #logging
          string_ = 'LR: {lr:.3e} | Epoch: [{0}][{1}/{2}] |\
                    Time {batch_time:.3f} | Data {data_time:.3f}'.format(
                    epoch, int((i+1) / self.config.optimizer.accumulate_step),
                    int(len(self.trainloader) / self.config.optimizer.accumulate_step),
                    batch_time=batch_time.avg, data_time=data_time.avg, lr=self.scheduler.get_last_lr()[0])
          string_ = Pretrain.logging_string(string_, self.loss)
          logging.info(string_)
          #logging.info(data['id'])

          # wandb
          dict_ = {'train/lr': self.scheduler.get_last_lr()[0]}
          self.update_wandb_dict('train', dict_, self.loss)
          wandb.log(dict_, step=int((epoch*len(self.trainloader)+(i+1)) / self.config.optimizer.accumulate_step))

