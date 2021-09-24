import numpy as np
import random
import scipy

from scipy.interpolate import RegularGridInterpolator


# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Color Transformations
##############################

class ChromaticTranslation(object):
  """Add random color to the image, input must be an array in [0,255] or a PIL image"""
  def __init__(self, trans_range_ratio=1e-1):
    """
    trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
    """
    self.trans_range_ratio = trans_range_ratio

  def __call__(self, rgb):
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
      rgb[:, :3] = np.clip(tr + rgb[:, :3], 0, 255)
    return rgb

class ChromaticAutoContrast(object):
  def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
    self.randomize_blend_factor = randomize_blend_factor
    self.blend_factor = blend_factor

  def __call__(self, rgb):
    if random.random() < 0.2:
      lo = rgb[:, :3].min(0, keepdims=True)
      hi = rgb[:, :3].max(0, keepdims=True)
      assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"
      scale = 255 / (hi - lo)
      contrast_rgb = (rgb[:, :3] - lo) * rgb
      blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
      feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats
    return feats

class ChromaticJitter(object):
  def __init__(self, std=0.01):
    self.std = std

  def __call__(self, rgb):
    if random.random() < 0.95:
      noise = np.random.randn(rgb.shape[0], 3)
      noise *= self.std * 255
      rgb[:, :3] = np.clip(noise + rgb[:, :3], 0, 255)
    return rgb


class HueSaturationTranslation(object):

  @staticmethod
  def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

  @staticmethod
  def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

  def __init__(self, hue_max, saturation_max):
    self.hue_max = hue_max
    self.saturation_max = saturation_max

  def __call__(self, rgb):
    # Assume feat[:, :3] is rgb
    hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
    hue_val = (random.random() - 0.5) * 2 * self.hue_max
    sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
    hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
    hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
    rgb[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

    return rgb


##############################
# Coordinate Transformations
##############################
class RandomDropout(object):
  def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.dropout_ratio = dropout_ratio
    self.dropout_application_ratio = dropout_application_ratio

  def __call__(self, coords):
    N = len(coords)
    inds = np.arange(N)
    if random.random() < self.dropout_ratio:
      inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
    return inds
    
class RandomHorizontalFlip(object):

  def __init__(self, upright_axis):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.D = 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])

  def __call__(self, xyz):
    if random.random() < 0.95:
      for curr_ax in self.horz_axes:
        if random.random() < 0.5:
          coord_max = np.max(xyz[:, curr_ax])
          xyz[:, curr_ax] = coord_max - xyz[:, curr_ax]
    return xyz

class RandomTranslation(object):
  def __init__(self, translate_range=0.5):
    self.translate_range = translate_range

  def __call__(self, xyz):
    if random.random() < 0.5:
      xyz_factor = np.random.choice(np.arange(-self.translate_range, self.translate_range, 0.001), size=3)
      xyz += xyz_factor
    return xyz

class RandomRotation(object):
  def __init__(self):
    self.rot_bound = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))

  def __call__(self, xyz):
    if random.random() < 0.5:
      # x rotation matrix
      theta = np.random.uniform(low=self.rot_bound[0][0], high=self.rot_bound[0][1], size=1)[0]
      Rx = np.array(
          [[1, 0, 0],
           [0, np.cos(theta), -np.sin(theta)],
           [0, np.sin(theta), np.cos(theta)]]
      )

      # y rotation matrix
      theta = np.random.uniform(low=self.rot_bound[1][0], high=self.rot_bound[1][1], size=1)[0]
      Ry = np.array(
          [[np.cos(theta), 0, np.sin(theta)],
           [0, 1, 0],
           [-np.sin(theta), 0, np.cos(theta)]]
      )

      # z rotation matrix
      theta = np.random.uniform(low=self.rot_bound[2][0], high=self.rot_bound[2][1], size=1)[0]
      Rz = np.array(
          [[np.cos(theta), -np.sin(theta), 0],
           [np.sin(theta), np.cos(theta), 0],
           [0, 0, 1]]
      )

      # rotate
      R = np.matmul(np.matmul(Rz, Ry), Rx)
      center = np.mean(xyz, axis=0)
      xyz -= center
      xyz = np.matmul(R, xyz.T).T
      xyz += center

    return xyz

class RandomScale(object):
  def __init__(self, bound=0.05):
    self.bound = bound

  def __call__(self, xyz):
    if random.random() < 0.5:
      factor = np.random.choice(np.arange(1.0-self.bound, 1.0+self.bound, 0.001), size=1)[0]
      xyz *= [factor, factor, factor]
    return xyz

class ElasticDistortion:

  def __init__(self, distortion_params):
    self.distortion_params = distortion_params

  def elastic_distortion(self, xyz, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

      pointcloud: numpy array of (number of points, at least 3 spatial dims)
      granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
      magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    xyz_min = xyz.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((xyz - xyz_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
      noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
      noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(xyz_min - granularity, xyz_min + granularity *
                                   (noise_dim - 2), noise_dim)
    ]
    interp = RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    xyz += interp(xyz) * magnitude
    return xyz

  def __call__(self, xyz):
    if self.distortion_params is not None:
      if random.random() < 0.95:
        for granularity, magnitude in self.distortion_params:
          xyz = self.elastic_distortion(xyz, granularity, magnitude)
    return xyz

