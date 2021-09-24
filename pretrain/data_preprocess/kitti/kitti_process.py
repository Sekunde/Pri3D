# Basic libs
import os, time, glob, random, pickle, copy, torch, argparse
import numpy as np
import trimesh
from collections import namedtuple
from PIL import Image

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

def odometry_to_positions(odometry):
    T_w_cam0 = odometry.reshape(3, 4)
    cam02world = np.vstack((T_w_cam0, [0, 0, 0, 1]))
    return cam02world

def warp(depth_map0, depth_map1, calib, positions):

    K_cam2 = calib.K_cam2
    K_cam2_inv = np.linalg.inv(K_cam2)

    y0, x0 = (depth_map0 != - 1).nonzero()
    d0 = depth_map0[y0,x0]
    xyz0 = np.stack([x0*d0, y0*d0, d0], 0)
    xyz0 = K_cam2_inv @ xyz0
    xyz0 = xyz0.T

    y1, x1 = (depth_map1 != - 1).nonzero()
    d1 = depth_map1[y1,x1]
    xyz1 = np.stack([x1*d1, y1*d1, d1], 0)
    xyz1 = K_cam2_inv @ xyz1
    xyz1 = xyz1.T

    velo2cam2 = calib.velo2cam2
    velo2cam0 = calib.velo2cam0
    cam02velo = np.linalg.inv(velo2cam0)
    cam22velo = np.linalg.inv(velo2cam2)

    cam02world0 = positions[0]
    cam02world1 = positions[1]
    world2cam01 = np.linalg.inv(cam02world1)

    M = velo2cam2 @ cam02velo @ world2cam01 @ cam02world0 @ velo2cam0 @ cam22velo 
    xyz0_t = apply_transform(xyz0, M)


    xyz0_in_cam1 = (K_cam2 @ xyz0_t.T).T
    x0_in_cam1 = xyz0_in_cam1[:,0] / xyz0_in_cam1[:,2]
    y0_in_cam1 = xyz0_in_cam1[:,1] / xyz0_in_cam1[:,2]
    x0_in_cam1 = x0_in_cam1.astype(np.int32)
    y0_in_cam1 = y0_in_cam1.astype(np.int32)
    z0_in_cam1 = xyz0_in_cam1[:,2]
    mask_boundary = (x0_in_cam1 >= 0) & (y0_in_cam1 >= 0) & (x0_in_cam1 < depth_map1.shape[1]) & (y0_in_cam1 < depth_map1.shape[0])

    # update once
    y0 = y0[mask_boundary]
    x0 = x0[mask_boundary]
    y0_in_cam1 = y0_in_cam1[mask_boundary]
    x0_in_cam1 = x0_in_cam1[mask_boundary]
    z0_in_cam1 = z0_in_cam1[mask_boundary]

    z1 = depth_map1[y0_in_cam1, x0_in_cam1]
    mask_z1 = (z1 != -1)

    y0 = y0[mask_z1]
    x0 = x0[mask_z1]
    y0_in_cam1 = y0_in_cam1[mask_z1]
    x0_in_cam1 = x0_in_cam1[mask_z1]
    z0_in_cam1 = z0_in_cam1[mask_z1]
    z1 = z1[mask_z1]

    mask_depth = np.abs(z1 - z0_in_cam1) < 0.5
    x0 = x0[mask_depth]
    y0 = y0[mask_depth]
    x0_in_cam1 = x0_in_cam1[mask_depth]
    y0_in_cam1 = y0_in_cam1[mask_depth]


    matches = np.stack([x0, y0, x0_in_cam1, y0_in_cam1], 1)
    #matches = np.stack([y0, x0, y0_in_cam1, x0_in_cam1], 1)

    return matches


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    #velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    data['Tr'] = velo2cam

    return data

def load_calib(sequence_path):
      """Load and compute intrinsic and extrinsic calibration parameters."""
      # We'll build the calibration parameters as a dictionary, then
      # convert it to a namedtuple to prevent it from being modified later
      data = {}

      # Load the calibration file
      calib_filepath = os.path.join(sequence_path, 'calib.txt')
      filedata = read_calib_file(calib_filepath)

      # Create 3x4 projection matrices
      P_rect_00 = np.reshape(filedata['P0'], (3, 4))
      P_rect_10 = np.reshape(filedata['P1'], (3, 4))
      P_rect_20 = np.reshape(filedata['P2'], (3, 4))
      P_rect_30 = np.reshape(filedata['P3'], (3, 4))

      data['P_rect_00'] = P_rect_00
      data['P_rect_10'] = P_rect_10
      data['P_rect_20'] = P_rect_20
      data['P_rect_30'] = P_rect_30

      # Compute the rectified extrinsics from cam0 to camN
      T1 = np.eye(4)
      T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
      T2 = np.eye(4)
      T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
      T3 = np.eye(4)
      T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

      # Compute the velodyne to rectified camera coordinate transforms
      data['velo2cam0'] = np.reshape(filedata['Tr'], (3, 4))
      data['velo2cam0'] = np.vstack([data['velo2cam0'], [0, 0, 0, 1]])
      data['velo2cam1'] = T1.dot(data['velo2cam0'])
      data['velo2cam2'] = T2.dot(data['velo2cam0'])
      data['velo2cam3'] = T3.dot(data['velo2cam0'])

      # Compute the camera intrinsics
      data['K_cam0'] = P_rect_00[0:3, 0:3]
      data['K_cam1'] = P_rect_10[0:3, 0:3]
      data['K_cam2'] = P_rect_20[0:3, 0:3]
      data['K_cam3'] = P_rect_30[0:3, 0:3]

      # Compute the stereo baselines in meters by projecting the origin of
      # each camera frame into the velodyne frame and computing the distances
      # between them
      p_cam = np.array([0, 0, 0, 1])
      p_velo0 = np.linalg.inv(data['velo2cam0']).dot(p_cam)
      p_velo1 = np.linalg.inv(data['velo2cam1']).dot(p_cam)
      p_velo2 = np.linalg.inv(data['velo2cam2']).dot(p_cam)
      p_velo3 = np.linalg.inv(data['velo2cam3']).dot(p_cam)

      data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
      data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

      calib = namedtuple('CalibData', data.keys())(*data.values())

      return calib


class KITTIPair(object):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """

    def __init__(self, data_path):
        super(KITTIPair, self).__init__()

        self.path_odometry = os.path.join(data_path, 'odometry/data_odometry_velodyne/dataset/poses/')
        self.path_color = os.path.join(data_path, 'odometry/data_odometry_color/dataset/sequences/')
        self.path_depth = os.path.join(data_path, '/rhome/jhou/data/dataset/kitti/depth/data_depth_annotated/')
        self.output_path = os.path.join(data_path, 'preprocessed')

        self.mapping = {'2011_10_03_drive_0027_sync': '00',
                        '2011_10_03_drive_0042_sync': '01',
                        '2011_10_03_drive_0034_sync': '02',
                        '2011_09_30_drive_0016_sync': '04',
                        '2011_09_30_drive_0018_sync': '05',
                        '2011_09_30_drive_0020_sync': '06',
                        '2011_09_30_drive_0027_sync': '07',
                        '2011_09_30_drive_0028_sync': '08',
                        '2011_09_30_drive_0033_sync': '09',
                        '2011_09_30_drive_0034_sync': '10' }

        self.overlap = []
        self.files = {}
        for split in os.listdir(self.path_depth):
            for drive_id in os.listdir(os.path.join(self.path_depth, split)):
                if drive_id in self.mapping:
                    sequence_id = self.mapping[drive_id]
                    # get one-to-one distance by comparing the translation vector
                    all_odo = self.get_odometry(sequence_id)
                    max_num = all_odo.shape[0]
                    self.files[drive_id] = []

                    fnames = glob.glob(self.path_depth + '/train/{}/proj_depth/groundtruth/image_02/*.png'.format(drive_id))
                    fnames_val = glob.glob(self.path_depth + '/val/{}/proj_depth/groundtruth/image_02/*.png'.format(drive_id))
                    fnames.extend(fnames_val)

                    for fname in fnames:
                        idx = int(os.path.split(fname)[-1][:-4])
                        if idx < max_num:
                            self.files[drive_id].append(fname)



    def get_odometry(self, sequence_id, indices=None):
        data_path = os.path.join(self.path_odometry, '{}.txt'.format(sequence_id))
        poses = np.genfromtxt(data_path)
        if indices == None:
            return poses 
        else:
            return poses[indices]

    def run(self):
        for drive_id in self.files:
            sequence_id = self.mapping[drive_id]

            #------------pre load depth maps-------------------
            depth_map = {}
            print('reading depth maps.....')
            for i in range(len(self.files[drive_id])):
                print('{}/{}'.format(i, len(self.files[drive_id])), end='\r')
                fname = self.files[drive_id][i]
                depth_map[os.path.split(fname)[-1]] = depth_read(fname)
            #--------------------------------------------------

            for i in range(len(self.files[drive_id])):
                print('{}/{}'.format(i, len(self.files[drive_id])), end='\r')
                for j in range(i+1, len(self.files[drive_id])):

                    fname0 = self.files[drive_id][i]
                    fname1 = self.files[drive_id][j]
                    t0 = int(os.path.split(fname0)[-1][:-4])
                    t1 = int(os.path.split(fname1)[-1][:-4])
                    if abs(t0-t1) > 100:
                        continue
                    all_odometry = self.get_odometry(sequence_id, [t0, t1])
                    #positions is cam2world
                    positions = [odometry_to_positions(odometry) for odometry in all_odometry]
                    #intrinsic
                    calib = load_calib(os.path.join(self.path_color, sequence_id))

                    stereo_depth_map0 = depth_map[os.path.split(fname0)[-1]]
                    stereo_depth_map1 = depth_map[os.path.split(fname1)[-1]]

                    stereo_matches = warp(stereo_depth_map0, stereo_depth_map1, calib, positions)
                    num0 = (stereo_depth_map0 != -1).sum()
                    num1 = (stereo_depth_map1 != -1).sum()
                    ratio = len(stereo_matches) / (num0 + num1 - len(stereo_matches))
                    if ratio > 0.05:
                        # save mapping
                        os.makedirs(os.path.join(self.output_path,'mapping', drive_id), exist_ok=True)
                        split = fname0.split('/')[-6]
                        fname0 = fname0.split('/')[-1].split('.')[0]
                        fname1 = fname1.split('/')[-1].split('.')[0]
                        save_name = '_'.join([split, fname0, fname1])
                        np.save(os.path.join(self.output_path, 'mapping', drive_id, save_name), stereo_matches)
                        # save txt
                        self.overlap.append(' '.join([drive_id, split, fname0, fname1, str(ratio)])+'\n')
                break

        filename = os.path.join(self.output_path, 'overlap.txt')
        open(filename, 'w').writelines(self.overlap)

def parse_config():
    parser = argparse.ArgumentParser(description='Render directory')
    parser.add_argument('--input', required=True, help='kitti dataset location')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    kitti_pair = KITTIPair(args.input)
    kitti_pair.run()


