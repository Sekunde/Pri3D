import numpy as np
from torch import nn
from torch.nn import functional as F

from model.backbone import build_backbone
from model.loss import SemanticLoss
from common.io2d import write_to_label

class Semantic2D(nn.Module):
    def __init__(self, config, nclasses, loss_reweight=None):
        super(Semantic2D, self).__init__()
        self.backbone = build_backbone(config.finetune.backbone, nclasses, config.finetune.pretrain)
        self.loss = SemanticLoss(loss_reweight)

    def forward(self, data):
        color, label2d = data['color'], data['label2d']
        feature = self.backbone(color.cuda())
        output = nn.functional.interpolate(feature, size=[label2d.shape[1], label2d.shape[2]], mode='bilinear', align_corners=True)
        loss = self.loss(output, label2d.cuda())
        output = output.argmax(1).cpu().long()
        return output, label2d, loss

