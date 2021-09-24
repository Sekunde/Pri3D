import os
import torch
import numpy as np
import argparse
from PIL import Image
from torch import nn

def _is_pil_image(img):
    return isinstance(img, Image.Image)

class Resize:
    def __init__(self, size, mode=Image.NEAREST):
        self.size = size
        self.mode = mode

    def __call__(self, image):
        ow, oh = self.size
        image = image.resize((ow, oh), self.mode)

        return image

class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, image):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
        image = self.to_tensor(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=True))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=True))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

class Unproject(nn.Module):
    # 2d to 3d
    # process translation from camera coordinates to grid space on the fly
    def __init__(self, image_size, intrinsic_inv, device="cuda"):
        """
        depth_min: 0.6
        depth_max: 3.8
        volume_size: [x, y, z]
        image_size: [height, width]
        """
        super(Unproject, self).__init__()
        self.image_size = image_size 
        self.intrinsic_inv = intrinsic_inv

        self.depth_min = 0.4
        self.depth_max = 6.0

        # 2d to 3d
        width = image_size[0]
        height = image_size[1]
        xv, yv = torch.meshgrid([torch.arange(0,width), torch.arange(0,height)])
        device = intrinsic_inv.device
        self.coords2d = torch.stack([xv, yv], 2).long().view(-1,2).to(device)

    @staticmethod
    def coords_multiplication(matrix, points):
        """
        matrix: 4x4
        points: nx3
        """
        if isinstance(matrix, torch.Tensor):
            device = torch.device("cuda" if matrix.get_device() != -1 else "cpu")
            points = torch.cat([points.t(), torch.ones((1, points.shape[0]), device=device)])
            return torch.mm(matrix, points).t()[:, :3]
        elif isinstance(matrix, np.ndarray):
            points = np.concatenate([np.transpose(points), np.ones((1, points.shape[0]))])
            return np.transpose(np.dot(matrix, points))[:, :3]

    def _backproject(self, depth_map, extrinsic=None):
        dtype, device = depth_map.dtype, depth_map.device
        batch_size = depth_map.shape[0]
        matrix = torch.mm(extrinsic, self.intrinsic_inv) if extrinsic is not None else self.intrinsic_inv

        coords3d = []
        for idx in range(batch_size):
            depth_value = depth_map[idx,0][self.coords2d[:,1].long(), self.coords2d[:,0].long()]
            points = torch.cat([self.coords2d.float(), depth_value[:,None].float()], 1)
            points[:,0] *= depth_value
            points[:,1] *= depth_value
            points = Unproject.coords_multiplication(matrix, points)
            coords3d.append(points)

        return torch.stack(coords3d)

    def forward(self, depth_map, extrinsic=None):
        coords3d = self._backproject(depth_map, extrinsic)
        return coords3d

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):

    if intrinsic_image_dim == image_dim:
        return intrinsic

    intrinsic_return = np.copy(intrinsic)

    height_after = image_dim[1]
    height_before = intrinsic_image_dim[1]
    height_ratio = height_after / height_before

    width_after = image_dim[0]
    width_before = intrinsic_image_dim[0]
    width_ratio = width_after / width_before

    if width_ratio >= height_ratio:
        resize_height = height_after
        resize_width = height_ratio * width_before

    else:
        resize_width = width_after
        resize_height = width_ratio * height_before

    intrinsic_return[0,0] *= float(resize_width)/float(width_before)
    intrinsic_return[1,1] *= float(resize_height)/float(height_before)
    # account for cropping/padding here
    intrinsic_return[0,2] *= float(resize_width-1)/float(width_before-1)
    intrinsic_return[1,2] *= float(resize_height-1)/float(height_before-1)

    return intrinsic_return

def get_scannet_intrinsic(image_size):

    scannet_intrinsic = np.array([[288.9353025,   0.       , 159.5, 0.],
                                  [  0.       , 288.9353025, 119.5, 0.],
                                  [  0.       ,   0.       ,   1. , 0.],
                                  [  0.       ,   0.       ,   0. , 1.]])

    return adjust_intrinsic(scannet_intrinsic, [320, 240], image_size)
