# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from collections import OrderedDict

from . import fpn as fpn_module
from .resnet import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101
from .resunet import Res50UNet, Res18UNet, Res10UNet, Res18UNetMultiRes
from .pointnet2 import PointNet2
from .enet import ENet
from .resunet3d import MinkUNet34C as ResUNet3D

def build_backbone(name, nclasses, pretrained=False):
    modelClass = globals()[name]
    model = modelClass(nclasses, pretrained=pretrained)
    return model
