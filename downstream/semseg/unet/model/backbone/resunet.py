
import torch
import numpy as np

from torch import nn
from typing import Tuple
from torch.nn import functional as F

from model.backbone.resnet import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101


class Encoder(nn.Module):
    def __init__(self, original_model, num_features=2048):
        super(Encoder, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

        self.out_channels = [self.layer1[-1].out_channels, self.layer2[-1].out_channels,
                             self.layer3[-1].out_channels, self.layer4[-1].out_channels]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4


class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)

        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)
        return out

class Decoder(nn.Module):
    def __init__(self, block_channel, output_channel=1):
        super(Decoder, self).__init__()

        num_features=block_channel[-1]

        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        # last part
        self.conv0 = nn.Conv2d(
            num_features, output_channel, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):

        x_block1, x_block2, x_block3, x_block4 = x

        x_d1 = self.up1(x_block4, [x_block3.size(2), x_block3.size(3)])
        x_d1 = x_d1 + x_block3

        x_d2 = self.up2(x_d1,     [x_block2.size(2), x_block2.size(3)])
        x_d2 = x_d2 + x_block2

        x_d3 = self.up3(x_d2,     [x_block1.size(2), x_block1.size(3)])
        x_d3 = x_d3 + x_block1

        x_d4 = self.up4(x_d3,     [x_block1.size(2)*2, x_block1.size(3)*2])

        x_d5 = self.conv0(x_d4)

        return x_d5

class DecoderMultiRes(nn.Module):
    def __init__(self, block_channel, output_channel=1):
        super(DecoderMultiRes, self).__init__()

        num_features=block_channel[-1]

        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        # last part
        self.conv1 = nn.Conv2d(
            num_features, output_channel, kernel_size=1, stride=1, padding=0, bias=True)

        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        # last part
        self.conv2 = nn.Conv2d(
            num_features, output_channel, kernel_size=1, stride=1, padding=0, bias=True)

    
    def forward(self, x):

        x_block1, x_block2, x_block3, x_block4 = x

        x_d1 = self.up1(x_block4, [x_block3.size(2), x_block3.size(3)])

        x_d1 = x_d1 + x_block3
        x_d2 = self.up2(x_d1,     [x_block2.size(2), x_block2.size(3)])

        x_d2 = x_d2 + x_block2
        x_d3 = self.up3(x_d2,     [x_block1.size(2), x_block1.size(3)])
        x_d3_out = self.conv1(x_d3)

        x_d3 = x_d3 + x_block1
        x_d4 = self.up4(x_d3,     [x_block1.size(2)*2, x_block1.size(3)*2])
        x_d4_out = self.conv2(x_d4)

        return x_d4_out, x_d3_out

class Res50UNet(nn.Module):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self, output_channel, pretrained='imagenet'):
        super().__init__()
        resnet_backbone = ResNet50(nclasses=None, pretrained=pretrained)
        block_channel = [256, 512, 1024, 2048]

        self.encoder = Encoder(resnet_backbone)
        self.decoder = Decoder(block_channel, output_channel=output_channel)

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features)
        return features

class Res18UNet(nn.Module):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self, output_channel, pretrained=False):
        super().__init__()
        
        block_channel = [64, 128, 256, 512]
        resnet_backbone = ResNet18(nclasses=None, pretrained=pretrained)

        self.encoder = Encoder(resnet_backbone)
        self.decoder = Decoder(block_channel, output_channel=output_channel)

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features)
        return features


class Res18UNetMultiRes(nn.Module):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self, output_channel, pretrained=False):
        super().__init__()
        
        block_channel = [64, 128, 256, 512]
        resnet_backbone = ResNet18(nclasses=None, pretrained=pretrained)

        self.encoder = Encoder(resnet_backbone)
        self.decoder = DecoderMultiRes(block_channel, output_channel=output_channel)

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features)
        return features


class Res10UNet(nn.Module):
    """Generate the ENet model.

    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.

    """

    def __init__(self, output_channel, pretrained=False):
        super().__init__()
        
        block_channel = [64, 128, 256, 512]
        resnet_backbone = ResNet10(nclasses=None, pretrained=pretrained)

        self.encoder = Encoder(resnet_backbone)
        self.decoder = Decoder(block_channel, output_channel=output_channel)

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features)
        return features
