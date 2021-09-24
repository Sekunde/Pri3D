import torch
import os
import numpy as np
from PIL import Image, ImageFilter
from torch import nn
from typing import Tuple
from torch.nn import functional as F
import MinkowskiEngine as ME

from model.backbone import build_backbone
from model.loss import DepthLoss, SemanticLoss, ContrastiveLoss, ReconstructionLoss
from common.io2d import write_to_label
from common.io3d import write_triangle_mesh, create_color_palette

#---------------------------------- Utils ---------------------------------------
class EmptyTensorError(Exception):
    pass

def grid_positions(h, w, device):
    lines = torch.arange(0, h, device=device).view(-1, 1).float().repeat(1, w)
    columns = torch.arange(0, w, device=device).view(1, -1).float().repeat(h, 1)
    return torch.cat([lines.view(1, -1), columns.view(1, -1)], dim=0)

def uv_to_pos(uv):
    return torch.stack([uv[1, :], uv[0, :]], dim=1)

def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = (pos - 0.5) / 2
    return pos

# -------------------------------- Pretraining -------------------------------------

class Pri3D(nn.Module):
    def __init__(self, config):
        super(Pri3D, self).__init__()
        self.config = config

        backbone_output_channels = {
            'Res18UNet': 32,
            'Res50UNet': 128,
            'ResNet18': 32,
            'Res18UNetMultiRes': 32,
        }

        self.backbone = build_backbone(config.pretrain.backbone, 
                                       backbone_output_channels[config.pretrain.backbone],
                                       config.pretrain.pretrained)
        self.contrastive_loss = ContrastiveLoss(config.pretrain.nceT)

        h = config.dataset.size[0]
        w = config.dataset.size[1]
        self.pos = grid_positions(h, w, 'cuda').long().transpose(0,1)

        if self.config.pretrain.geometric_prior:
            self.backbone3d = build_backbone('ResUNet3D', 96, False)
            self.w2d = nn.Linear(backbone_output_channels[config.pretrain.backbone], 64)
            self.w3d = nn.Linear(96, 64)

        if self.config.pretrain.depth:
            self.headnet = nn.Conv2d(backbone_output_channels[config.pretrain.backbone],
                                     1, kernel_size=1, stride=1, padding=0, bias=True)
            self.headloss = DepthLoss(invalid_value=0)

    
    def forward(self, data):
        output = {}
        losses = {'contrastive_loss': torch.FloatTensor([0.0]).cuda()}

        color1, depth1, world2camera1, intrinsics1, bbox1 = \
                data['color1'], data['depth1'], data['world2camera1'], data['intrinsics1'], data['bbox1']

        color2, depth2, world2camera2, intrinsics2, bbox2 = \
                data['color2'], data['depth2'], data['world2camera2'], data['intrinsics2'], data['bbox2']
        batch_size = color1.shape[0]

        # get features
        feature2d1 = self.backbone(color1.cuda())
        feature2d2 = self.backbone(color2.cuda())

        if self.config.pretrain.geometric_prior:
            pointcloud1, feature3d1, world2grid1 = data['pointcloud1'], data['feature3d1'], data['world2grid1']
            pointcloud2, feature3d2, world2grid2 = data['pointcloud2'], data['feature3d2'], data['world2grid2']
            input1 = ME.SparseTensor(features=feature3d1[:,1:], coordinates=pointcloud1.int(), device='cuda')
            input2 = ME.SparseTensor(features=feature3d2[:,1:], coordinates=pointcloud2.int(), device='cuda')
            feature3d1 = self.backbone3d(input1)
            feature3d2 = self.backbone3d(input2)
            coords3d1 = feature3d1.C
            coords3d2 = feature3d2.C
            feature3d1 = feature3d1.F
            feature3d2 = feature3d2.F


        if self.config.pretrain.depth:
            depth_pred1 = self.headnet(feature2d1)
            depth_pred2 = self.headnet(feature2d2)
            depth_pred1 = nn.functional.interpolate(depth_pred1, size=depth1.shape[1:], mode='bilinear', align_corners=True)
            depth_pred2 = nn.functional.interpolate(depth_pred2, size=depth2.shape[1:], mode='bilinear', align_corners=True)
            headloss1 = self.headloss(depth_pred1, depth1.cuda())
            headloss2 = self.headloss(depth_pred2, depth2.cuda())
            losses['depth_loss'] = (headloss1 + headloss2) / 2.0

        for batch_id in range(batch_size):
            if self.config.pretrain.view_invariant:
                try:
                    pos1, pos2 = self.warp(depth1[batch_id].cuda(), intrinsics1[batch_id].cuda(),
                                           world2camera1[batch_id].cuda(), bbox1[batch_id].cuda(), 
                                           depth2[batch_id].cuda(), intrinsics2[batch_id].cuda(),
                                           world2camera2[batch_id].cuda(), bbox2[batch_id].cuda(), self.config.pretrain.thresh)

                except EmptyTensorError:
                    continue

                if self.config.pretrain.sample_points:
                    replace = self.config.pretrain.sample_points > pos1.shape[0] 
                    choice = torch.from_numpy(np.random.choice(pos1.shape[0], self.config.pretrain.sample_points, replace=replace)).long()
                    pos1 = pos1[choice]
                    pos2 = pos2[choice]

                pos1 = torch.round(downscale_positions(pos1, 1))
                pos2 = torch.round(downscale_positions(pos2, 1))
                contrastive2d1 = feature2d1[batch_id][:, pos1[:,0].long(), pos1[:,1].long()].transpose(0,1)
                contrastive2d2 = feature2d2[batch_id][:, pos2[:,0].long(), pos2[:,1].long()].transpose(0,1)
                contrastive_loss = self.contrastive_loss(contrastive2d1, contrastive2d2)
                losses['contrastive_loss'] += contrastive_loss / batch_size

            if self.config.pretrain.geometric_prior:
                mask3d = coords3d1[:,0] == batch_id
                coords3d = coords3d1[mask3d,1:]
                chunk_loss = self.chunk_loss(coords3d, depth1[batch_id], intrinsics1[batch_id],
                                             world2camera1[batch_id], world2grid1[batch_id], 
                                             feature2d1[batch_id], feature3d1[mask3d])
                losses['contrastive_loss'] += chunk_loss / (batch_size * 2)

                mask3d = coords3d2[:,0] == batch_id
                coords3d = coords3d2[mask3d,1:]
                chunk_loss = self.chunk_loss(coords3d, depth2[batch_id], intrinsics2[batch_id],
                                             world2camera2[batch_id], world2grid2[batch_id], 
                                             feature2d2[batch_id], feature3d2[mask3d])
                losses['contrastive_loss'] += chunk_loss / (batch_size * 2)

        total_loss = 0.0
        for key in losses:
            total_loss += losses[key]
        losses['total_loss'] = total_loss
        return output, losses

    def warp(self, depth1, intrinsics1, world2camera1, bbox1,
                   depth2, intrinsics2, world2camera2, bbox2, threshold=0.05):

        device = self.pos.device
        
        Z1 = depth1[self.pos[:,0].long(), self.pos[:,1].long()]
        inds = torch.arange(0, self.pos.size(0), device=device)
        valid_depth_mask = (Z1 != -1)
        Z1 = Z1[valid_depth_mask]
        inds = inds[valid_depth_mask]

        pos1 = self.pos[inds]
        # inds == pos1
        # COLMAP convention
        u1 = pos1[:,1] + bbox1[1] + .5
        v1 = pos1[:,0] + bbox1[0] + .5

        X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
        Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

        XYZ1_hom = torch.cat([
            X1.view(1, -1),
            Y1.view(1, -1),
            Z1.view(1, -1),
            torch.ones(1, Z1.size(0), device=device)], dim=0)
        XYZ2_hom = torch.chain_matmul(world2camera2, torch.inverse(world2camera1.cpu()).cuda(), XYZ1_hom)
        XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

        uv2_hom = torch.matmul(intrinsics2, XYZ2)
        uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)

        u2 = uv2[0, :] - bbox2[1] - .5
        v2 = uv2[1, :] - bbox2[0] - .5
        uv2 = torch.cat([u2.view(1, -1),  v2.view(1, -1)], dim=0)

        pos2 = uv_to_pos(uv2).long()

        bound_mask = (pos2[:,0] >=0) & (pos2[:,0] < depth2.shape[0]) & (pos2[:,1] >=0) & (pos2[:,1] < depth2.shape[1])
        inds = inds[bound_mask]
        pos1 = pos1[bound_mask]
        XYZ2 = XYZ2[:,bound_mask]
        pos2 = pos2[bound_mask]
        new_inds = torch.arange(0, pos2.size(0), device=device)

        annotated_depth = depth2[pos2[:,0].long(), pos2[:,1].long()]

        valid_depth_mask = (annotated_depth != -1)
        annotated_depth = annotated_depth[valid_depth_mask]
        new_inds = new_inds[valid_depth_mask]

        estimated_depth = XYZ2[2, new_inds]
        inlier_mask = torch.abs(estimated_depth - annotated_depth) < threshold

        # pos1 == pos2 == inds
        inds = inds[new_inds][inlier_mask]
        pos1 = pos1[new_inds][inlier_mask]
        pos2 = pos2[new_inds][inlier_mask]

        if inds.size(0) == 0:
            raise EmptyTensorError

        return pos1, pos2

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

    def chunk_loss(self, coords3d, depth, intrinsics, world2camera, world2grid, feature2d, feature3d):
        ind2d, ind3d = self.voxel_pixel(coords3d.cuda(), 
                                        depth.cuda(), 
                                        intrinsics.cuda(), 
                                        world2camera.cuda(), 
                                        world2grid.cuda())


        if self.config.pretrain.sample_points:
            replace = self.config.pretrain.sample_points > ind2d.shape[0] 
            choice = torch.from_numpy(np.random.choice(ind2d.shape[0], self.config.pretrain.sample_points, replace=replace)).long()
            ind2d = ind2d[choice]
            ind3d = ind3d[choice]

        ind2d = torch.round(downscale_positions(ind2d, 1))

        feature2d = feature2d[:, ind2d[:,1].long(), ind2d[:,0].long()].transpose(0,1).contiguous()
        feature3d = feature3d[ind3d]

        feature2d = self.w2d(feature2d)
        feature3d = self.w3d(feature3d)
        contrastive_loss = self.contrastive_loss(feature2d, feature3d)
        return contrastive_loss

    def voxel_voxel(self, coords1, coords2, grid2world1, world2grid2):
        from common.nn_search import nearest_neighbour
        inds, mask = nearest_neighbour(coords1, coords2, radius=2.0)
        matched_coords1 = coords1[mask]
        matched_coords2 = coords2[inds]
        return matched_coords1, matched_coords2

    def voxel_pixel(self, coords3d, depth, intrinsics, world2camera, world2grid):
        grid2world = torch.inverse(world2grid)
        coords3d_world = Pri3D.coords_multiplication(grid2world, coords3d)
        coords3d_camera = Pri3D.coords_multiplication(world2camera, coords3d_world)

        intrinsics4x4 = torch.zeros((4,4)).cuda()
        intrinsics4x4[:3,:3] = intrinsics[:3,:3]
        intrinsics4x4[3,3] = 1.0
        coords2d = Pri3D.coords_multiplication(intrinsics4x4, coords3d_camera)
        coords2d[:,:2] = coords2d[:,:2] / coords2d[:,2:]

        mask_h = (coords2d[:,1] >= 0) & (coords2d[:,1] < self.config.dataset.size[0])
        mask_w = (coords2d[:,0] >= 0) & (coords2d[:,0] < self.config.dataset.size[1])
        mask_spatial = mask_h & mask_w

        depth = depth[coords2d[mask_spatial,1].long(), coords2d[mask_spatial,0].long()]
        mask_depth = torch.abs(coords2d[mask_spatial,2] - depth) < 0.2
        return coords2d[mask_spatial][mask_depth][:,:2].long(), torch.arange(mask_spatial.shape[0])[mask_spatial][mask_depth]
