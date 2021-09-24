# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from collections import OrderedDict
from .resunet import Res50UNet, Res18UNet

def build_backbone(name, nclasses, pretrained=False):
    #model = build_resnet_fpn_backbone(model)
    modelClass = globals()[name]
    model = modelClass(nclasses, pretrained=pretrained)
    return model
