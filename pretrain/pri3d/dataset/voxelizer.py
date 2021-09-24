import numpy as np
import torch
import MinkowskiEngine as ME

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

class Voxelizer:
  def __init__(self,
               voxel_size=1,
               ignore_label=255):
    """
    Args:
      voxel_size: side length of a voxel
      ignore_label: label assigned for ignore (not a training label).
    """
    self.voxel_size = voxel_size
    self.ignore_label = ignore_label

    scale = 1 / self.voxel_size
    self.voxelization_matrix = torch.eye(4)
    self.voxelization_matrix[:3,:3].fill_diagonal_(scale)

  def voxelize(self, coords, labels=None):
    assert coords.shape[1] == 3
    coords = coords_multiplication(self.voxelization_matrix, coords).contiguous().long()
    if labels is not None:
        coords, _, inds = ME.utils.sparse_quantize(coords, labels=labels, return_index=True, ignore_label=self.ignore_label)
    else:
        coords, inds = ME.utils.sparse_quantize(coords, return_index=True)

    return coords, inds, self.voxelization_matrix
