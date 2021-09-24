import os
import torch
import argparse
from PIL import Image
import numpy as np
from chunk_utils import *


def process(opt):
    image_size = [320, 240]
    intrinsics = get_scannet_intrinsic(image_size)
    intrinsics_inv = torch.from_numpy(np.linalg.inv(intrinsics)).float()
    unproject = Unproject(image_size, intrinsics_inv)
    scene_ids = os.listdir(opt.base)

    for idx, scene_id in enumerate(scene_ids):
        print('{}/{}'.format(idx, len(scene_ids)), end='\r')
        if scene_id == 'splits':
            continue
        os.makedirs(os.path.join(opt.base, scene_id, 'world'), exist_ok=True)
        for frame_id in os.listdir(os.path.join(opt.base, scene_id, 'depth')):
            frame_id = frame_id.split('.')[0]
            pose = torch.from_numpy(np.loadtxt(os.path.join(opt.base, scene_id, 'pose', frame_id+'.txt'))).float()
            depth = Image.open(os.path.join(opt.base, scene_id, 'depth', frame_id+'.png'))
            depth = ToTensor()(Resize(image_size)(depth)) / 1000.0
            points_world = unproject(depth[None, :, :], extrinsic=pose)
            points_world = points_world[0].view(image_size[0], image_size[1], 3).transpose(0,1).contiguous()

            minx = torch.floor(points_world[:,:,0].min()).item()
            miny = torch.floor(points_world[:,:,1].min()).item()
            minz = torch.floor(points_world[:,:,2].min()).item()
            maxx = torch.ceil(points_world[:,:,0].max()).item()
            maxy = torch.ceil(points_world[:,:,1].max()).item()
            maxz = torch.ceil(points_world[:,:,2].max()).item()
            torch.save([minx, miny, minz, maxx, maxy, maxz], os.path.join(opt.base, scene_id, 'world', frame_id))


def config():
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='/rhome/jhou/data/dataset/scannet/partial_frames/')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = config()
    process(opt)

