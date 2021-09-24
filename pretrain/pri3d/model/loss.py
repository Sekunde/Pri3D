# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out

class DepthLoss(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, invalid_value=None):
        super(DepthLoss, self).__init__()
        self.cos_loss = torch.nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradient = Sobel().cuda()
        self.criterionL1 = torch.nn.functional.l1_loss
        self.invalid_value = invalid_value

    def forward(self, features, depth_target):
        depth_target = depth_target.unsqueeze(1).contiguous()
        grad_target = self.get_gradient(depth_target)
        grad_pred = self.get_gradient(features)

        grad_target_dx = grad_target[:, 0, :, :].contiguous().view_as(depth_target)
        grad_target_dy = grad_target[:, 1, :, :].contiguous().view_as(depth_target)
        grad_pred_dx = grad_pred[:, 0, :, :].contiguous().view_as(depth_target)
        grad_pred_dy = grad_pred[:, 1, :, :].contiguous().view_as(depth_target)

        ones = torch.ones(depth_target.size(0), 1, depth_target.size(2), depth_target.size(3)).float().cuda()
        normal_target = torch.cat((-grad_target_dx, -grad_target_dy, ones), 1)
        normal_pred = torch.cat((-grad_pred_dx, -grad_pred_dy, ones), 1)

        # filter out invalid
        mask = torch.ones_like(depth_target).bool().cuda()
        if self.invalid_value is not None:
            mask = (depth_target != self.invalid_value).bool().cuda()

        loss_depth = torch.log(torch.abs(depth_target[mask] - features[mask]) + 0.5).mean()

        loss_dx = torch.log(torch.abs(grad_target_dx[mask] - grad_pred_dx[mask]) + 0.5).mean()
        loss_dy = torch.log(torch.abs(grad_target_dy[mask] - grad_pred_dy[mask]) + 0.5).mean()
        loss_gradient = loss_dx + loss_dy

        loss_normal = torch.abs(1 - self.cos_loss(normal_pred, normal_target))
        loss_normal = loss_normal[mask[:,0,:,:]].mean()

        #losses = {'l1': loss_depth, 'normal': loss_normal, 'gradient': loss_gradient}
        loss = (loss_depth + loss_normal + loss_gradient) / 3

        return loss

#---------------------------------------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, features, target):
        loss = torch.log(torch.abs(target - features) + 0.5).mean()

        return loss

#---------------------------------------------------------------------------------------------------

class SemanticLoss(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, loss_reweight):
        super(SemanticLoss, self).__init__()
        # semantic segmentation
        self.loss_reweight = loss_reweight

    def forward(self, features, target):
        output = F.log_softmax(features, dim=1)
        labels = target.cuda().long()
        loss = F.nll_loss(output, labels, weight=self.loss_reweight)
        return loss

#---------------------------------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, nceT):
        super(ContrastiveLoss, self).__init__()
        # semantic segmentation
        self.T = nceT
        self.criterion = nn.CrossEntropyLoss()
        self.LARGE_NUM = 1e9

    def forward(self, q, k, mask=None):
        npos = q.shape[0] 
        q = q / torch.norm(q, p=2, dim=1, keepdim=True)
        k = k / torch.norm(k, p=2, dim=1, keepdim=True)
        logits = torch.mm(q, k.transpose(1, 0)) # npos by npos
        out = torch.div(logits, self.T)

        if mask != None:
            out = out - self.LARGE_NUM * mask.float()

        labels = torch.arange(npos).cuda().long()
        loss1 = self.criterion(out, labels)
        loss2 = self.criterion(out.transpose(1, 0), labels)
        return (loss1 + loss2) / 2