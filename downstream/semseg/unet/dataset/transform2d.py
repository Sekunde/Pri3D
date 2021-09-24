import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        for a in self.augmentations:
            img, mask = a(img, mask)
        return img, mask

class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (tf.hflip(img), tf.hflip(mask))
        return img, mask


class RandomRotation(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=2,
                fillcolor=(0, 0, 0),
                shear=0.0,
            ),
            tf.affine(
                mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=0,
                fillcolor=250,
                shear=0.0,
            ),
        )


