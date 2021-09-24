import numpy as np
from copy import copy
import math

# suncg intrinsic 320x240
suncg_intrinsic = np.array([[277.1281435,   0.       , 159.5,  0.],
                            [  0.       , 277.1281435, 119.5,  0.],
                            [  0.       ,   0.       ,   1. ,  0.],
                            [  0.       ,   0.       ,   0. ,  1.]])

scannet_intrinsic = np.array([[288.9353025,   0.       , 159.5, 0.],
                              [  0.       , 288.9353025, 119.5, 0.],
                              [  0.       ,   0.       ,   1. , 0.],
                              [  0.       ,   0.       ,   0. , 1.]])
#suncg_intrinsic = np.array([[554.256, 0.0, 319.5, 0.0],
#                            [0.0, 554.256, 239.5, 0.0],
#                            [0.0, 0.0, 1.0, 0.0],
#                            [0.0, 0.0, 0.0, 1.0]])

#reate camera intrinsics
# no content lost and keep aspect ratio
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

def intrinsic_by_crop(intrinsic, crop_point):
    intrinsic = intrinsic.clone()
    intrinsic[0, 2] -= crop_point[0]
    intrinsic[1, 2] -= crop_point[1]

    return intrinsic

def intrinsic_by_scale(intrinsic, scale):
    intrinsic = intrinsic.clone()
    intrinsic[0, 0] *= scale[0]
    intrinsic[1, 1] *= scale[1]
    intrinsic[0, 2] *= scale[0]
    intrinsic[1, 2] *= scale[1]

    return intrinsic

def get_suncg_intrinsic(image_size):
    return adjust_intrinsic(suncg_intrinsic, [320, 240], image_size)

def get_scannet_intrinsic(image_size):
    return adjust_intrinsic(scannet_intrinsic, [320, 240], image_size)

