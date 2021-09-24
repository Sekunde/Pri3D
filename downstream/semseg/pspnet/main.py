import os
import random
import time
import cv2
import numpy as np
import logging
import wandb
import argparse
import hydra

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data

from dataset.scannet import ScanNet
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
from util import transform

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def check(config):
    assert config.dataset.classes > 1
    assert config.train.zoom_factor in [1, 2, 4, 8]
    #assert (config.train.train_h - 1) % 8 == 0 and (config.train.train_w - 1) % 8 == 0

@hydra.main(config_path='config', config_name='default.yaml')
def main(config):
    check(config)

    criterion = nn.CrossEntropyLoss(ignore_index=config.train.ignore_label)

    from model.pspnet import PSPNet
    model = PSPNet(layers=config.train.layers, classes=config.dataset.classes, zoom_factor=config.train.zoom_factor, criterion=criterion, pretrained=config.train.weight)
    modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    modules_new = [model.ppm, model.cls, model.aux]

    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=config.train.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=config.train.base_lr * 10))
    optimizer = torch.optim.SGD(params_list, lr=config.train.base_lr, momentum=config.train.momentum, weight_decay=config.train.weight_decay)

    logging.info(config)
    logging.info("=> creating model ...")
    logging.info("Classes: {}".format(config.dataset.classes))
    logging.info(model)
    wandb.init(project="pri3d", name=config.train.exp_name, config=config, reinit=True)

    model = torch.nn.DataParallel(model.cuda())

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Resize((config.train.train_h, config.train.train_w)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    train_data = ScanNet(split='train', data_root=config.dataset.data_root, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.workers, pin_memory=True, drop_last=True)

    val_transform = transform.Compose([
        transform.Resize((config.train.train_h, config.train.train_w)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    val_data = ScanNet(split='val', data_root=config.dataset.data_root, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.train.batch_size_val, shuffle=False, num_workers=config.train.workers, pin_memory=True)

    best_miou = -1.0
    for epoch in range(config.train.epochs):
        print ('epoch_worker',epoch)
        train(train_loader, model, optimizer, epoch, config)

        if (epoch+1) % config.train.val_freq == 0:
            loss_val, mIoU_val = validate(val_loader, model, criterion, config)
            wandb.log({'val/loss': loss_val, 
                       'val/iou': mIoU_val}, step=(epoch+1)*len(train_loader))

            if mIoU_val > best_miou:
                best_miou = mIoU_val
                filename = 'best_miou.pth'
                logging.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iou':best_miou}, filename)


def train(train_loader, model, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    model.train()

    end = time.time()
    max_iter = config.train.epochs * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if config.train.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * config.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * config.zoom_factor + 1)
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output, main_loss, aux_loss = model(input, target)
        main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        loss = main_loss + config.train.aux_weight * aux_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = input.size(0)
        intersection, union, _ = intersectionAndUnionGPU(output, target, config.dataset.classes, config.train.ignore_label)
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(config.train.base_lr, current_iter, max_iter, power=config.train.power)
        for index in range(0, config.train.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(config.train.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10

        if (i + 1) % config.train.print_freq == 0:
            logging.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} '.format(epoch+1, config.train.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          loss_meter=loss_meter))
            wandb.log({'train/loss': loss_meter.val, 'train/mIoU': np.mean(intersection / (union + 1e-10))}, step=current_iter)

def validate(val_loader, model, criterion, config):
    logging.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    model.eval()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if config.train.zoom_factor != 8:
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        n = input.size(0)
        loss = torch.mean(loss)
        output = output.max(1)[1]

        intersection, union, _ = intersectionAndUnionGPU(output, target, config.dataset.classes, config.train.ignore_label)
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union)

        loss_meter.update(loss.item(), input.size(0))
        logging.info('Test: [{}/{}] '.format(i + 1, len(val_loader)))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    logging.info('Val result: mIoU {:.4f}.'.format(mIoU))

    for i in range(config.dataset.classes):
        logging.info('Class_{} Result: iou {:.4f}.'.format(i, iou_class[i]))
    logging.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]="7"
    main()
