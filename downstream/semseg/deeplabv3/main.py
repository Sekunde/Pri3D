import hydra
import util
import os
import random
import argparse
import numpy as np
import logging
import wandb

from model.deeplab import deeplabv3_resnet50, deeplabv3plus_resnet50
from torch.utils import data
from util import transform
from util import StreamSegMetrics
from dataset import ScanNet

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

def get_dataset(config):
    """ Dataset And Augmentation
    """

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Resize((240, 320)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    val_transform = transform.Compose([
        transform.Resize((240, 320)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    train_dst = ScanNet(data_root=config.dataset.data_root, split='train', transform=train_transform)
    val_dst = ScanNet(data_root=config.dataset.data_root, split='val', transform=val_transform)
    return train_dst, val_dst

def validate(config, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            logging.info("{}/{}".format(i, len(loader)))
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

        result = metrics.get_results()

    
    return result['Mean IoU']

def save_ckpt(path, cur_itrs, model, optimizer, scheduler, best_miou):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_miou": best_miou,
    }, path)
    logging.info("Model saved as %s" % path)

@hydra.main(config_path='config', config_name='default.yaml')
def main(config):
    # Setup visualization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    train_dst, val_dst = get_dataset(config)

    train_loader = data.DataLoader(
        train_dst, batch_size=config.train.batch_size, shuffle=True, num_workers=8)
    val_loader = data.DataLoader(val_dst, batch_size=config.train.batch_size, shuffle=False, num_workers=8)
    logging.info("Train set: %d, Val set: %d" % (len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3': deeplabv3_resnet50,
        'deeplabv3plus': deeplabv3plus_resnet50,
    }

    model = model_map[config.train.model](num_classes=config.dataset.classes, output_stride=config.train.output_stride, pretrained_backbone=config.train.weight)
    util.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(config.dataset.classes)

    wandb.init(project="pri3d", name=config.train.exp_name, config=config, reinit=True)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*config.train.lr},
        {'params': model.classifier.parameters(), 'lr': config.train.lr},
    ], lr=config.train.lr, momentum=0.9, weight_decay=config.train.weight_decay)

    if config.train.lr_policy=='poly':
        scheduler = util.PolyLR(optimizer, config.train.total_itrs, power=0.9)
    elif config.train.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train.step_size, gamma=0.1)

    # Set up criterion
    if config.train.loss_type == 'focal_loss':
        criterion = util.FocalLoss(ignore_index=255, size_average=True)
    elif config.train.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    util.mkdir('checkpoints')
    # Restore
    best_miou = 0.0
    cur_itrs = 0
    cur_epochs = 0
    model = nn.DataParallel(model)
    model.to(device)

    interval_loss = 0
    while True: #cur_itrs < total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                logging.info("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, config.train.total_itrs, interval_loss))
                wandb.log({'train/loss': interval_loss}, step=cur_itrs)
                interval_loss = 0.0

            if (cur_itrs) % config.train.val_interval == 0:
                logging.info("validation...")
                model.eval()
                miou = validate(config, model=model, loader=val_loader, device=device, metrics=metrics)
                logging.info('miou: {}'.format(miou))
                wandb.log({'val/iou': miou}, step=cur_itrs)
                if miou > best_miou:  # save best model
                    best_miou = miou
                    save_ckpt('checkpoints/best_miou.pth', cur_itrs, model, optimizer, scheduler, best_miou)
                model.train()
            scheduler.step()  

            if cur_itrs >=  config.train.total_itrs:
                return

if __name__ == '__main__':
    main()
