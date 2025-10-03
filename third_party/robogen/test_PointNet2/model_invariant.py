# NOTE:
# Trying to implement PointNet++
# Borrowed from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

try:
    from pointnet2_ops import pointnet2_utils
    HAS_POINTNET_OPS=True
except:
    HAS_POINTNET_OPS=False
    print('no pointnet2_ops')

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz_, npoint, keep_gripper_in_fps=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    if keep_gripper_in_fps: ### NOTE: assuming there are 4 gripper points
        xyz = xyz_[:, :-4, :]
        npoint = npoint - 4
    else:
        xyz = xyz_
    
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = farthest * 0 # set to 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    if keep_gripper_in_fps:
        gripper_indices = torch.Tensor([N, N+1, N+2, N+3]).long().to(device)
        gripper_indices = gripper_indices.unsqueeze(0).repeat(B, 1)
        centroids = torch.cat([centroids, gripper_indices], dim=1)
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, keep_gripper_in_fps=False, use_in=False):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.keep_gripper_in_fps = keep_gripper_in_fps
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                if use_in:
                    bns.append(nn.InstanceNorm2d(out_channel))
                else:
                    bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        # new_xyz = index_points(xyz, farthest_point_sample(xyz, S, self.keep_gripper_in_fps))
        if HAS_POINTNET_OPS:
            new_xyz = fps(xyz, S) # [B, npoint, 3]
        else: 
            new_xyz = index_points(xyz, farthest_point_sample(xyz, S, self.keep_gripper_in_fps))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
                # grouped_points =  F.relu((conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, use_in=False):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            if use_in:
                self.mlp_bns.append(nn.InstanceNorm1d(out_channel))
            else:
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            # new_points = F.relu((conv(new_points)))
        return new_points

class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2, self).__init__()
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=3, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=0, mlp_list=[[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 256) (B, 256, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 64) (B, 512, 64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 16) (B, 1024, 16)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 512, 64)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes


class PointNet2_small2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2_small2, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=0, mlp_list=[[16, 16, 16], [32, 32, 32]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=48, mlp_list=[[64, 64, 64], [64, 96, 64]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128, [[128, 196, 128], [128, 196, 128]])

        self.fp3 = PointNetFeaturePropagation(64+64+128+128, [128, 128])
        self.fp2 = PointNetFeaturePropagation(16+32+128, [64, 64])
        self.fp1 = PointNetFeaturePropagation(64, [64, 64, 64])
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 512) (B, 96, 512)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 128) (B, 256, 128)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 32) (B, 512, 32)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 128)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 512)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes: outputing logtis

class PointNet2_super(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False, use_in=False):
        super(PointNet2_super, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=input_channel - 3, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512], use_in=use_in)
        self.fp5 = PointNetFeaturePropagation(512+512+256+256, [512, 512], use_in=use_in)
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256], use_in=use_in)
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256], use_in=use_in)
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], use_in=use_in)
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], use_in=use_in)
        self.conv1 = nn.Conv1d(128, 128, 1)
        if use_in:
            self.bn1 = nn.InstanceNorm1d(128)
        else:
            self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 128)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        # x = F.relu(self.conv1(l0_points))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes
    

class PointNet2_Binary(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False, use_in=False, use_text_embedding=False):
        super(PointNet2_Binary, self).__init__()
        self.encoded_text_dim = 128  # Output dimension after encoding
        if use_text_embedding:
            self.text_encoder = nn.Linear(
                1024, self.encoded_text_dim
            )  # SIGLIP input dim
            self.film_predictor = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),  # [B, 128] -> [B, 256]
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # [B, 256] -> [B, 2048]
            )
            # Init as gamma=0 and beta=1
            self.film_predictor[-1].weight.data.zero_()
            self.film_predictor[-1].bias.data.copy_(
                torch.cat([torch.ones(1024), torch.zeros(1024)])
            )

        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=input_channel - 3, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        # self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512], use_in=use_in)
        # self.fp5 = PointNetFeaturePropagation(512+512+256+256, [512, 512], use_in=use_in)
        # self.fp4 = PointNetFeaturePropagation(1024, [256, 256], use_in=use_in)
        # self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256], use_in=use_in)
        # self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], use_in=use_in)
        # self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], use_in=use_in)
        # self.conv1 = nn.Conv1d(128, 128, 1)
        # if use_in:
        #     self.bn1 = nn.InstanceNorm1d(128)
        # else:
        #     self.bn1 = nn.BatchNorm1d(128)
        # # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

        self.binary_head = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Output: [gripper_state, ignore_collision]
        )


    def forward(self, xyz, text_embedding):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 128)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)

        if text_embedding is not None:
            encoded_text = self.text_encoder(text_embedding)  # [B, 128]
            film_params = self.film_predictor(encoded_text)  # [B, 1024 * 2]
            gamma, beta = film_params.chunk(2, dim=1)  # [B, 1024] each
            gamma = gamma.unsqueeze(2)  # [B, 1024, 1] for broadcasting
            beta = beta.unsqueeze(2)  # [B, 1024, 1] for broadcasting
            l6_points = gamma * l6_points + beta  # FiLM modulation: [B, 1024, 16]

        # Pass it through an mlp here
        x = torch.max(l6_points, dim=2)[0]  # Global feature vector (B, 1024)
        x = self.binary_head(x)  # (B, 2)

        return x

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        # x = F.relu(self.conv1(l0_points))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes


class PointNet2_Discrete(nn.Module):
    def __init__(self, num_classes, input_channel=3, keep_gripper_in_fps=False, use_in=False, use_text_embedding=False):
        super(PointNet2_Binary, self).__init__()
        self.encoded_text_dim = 128  # Output dimension after encoding
        if use_text_embedding:
            self.text_encoder = nn.Linear(
                1024, self.encoded_text_dim
            )  # SIGLIP input dim
            self.film_predictor = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),  # [B, 128] -> [B, 256]
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # [B, 256] -> [B, 2048]
            )
            # Init as gamma=0 and beta=1
            self.film_predictor[-1].weight.data.zero_()
            self.film_predictor[-1].bias.data.copy_(
                torch.cat([torch.ones(1024), torch.zeros(1024)])
            )

        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=input_channel - 3, mlp_list=[[16, 16, 32], [32, 32, 64]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=96, mlp_list=[[64, 64, 128], [64, 96, 128]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps=keep_gripper_in_fps, use_in=use_in)
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512], use_in=use_in)
        self.fp5 = PointNetFeaturePropagation(512+512+256+256, [512, 512], use_in=use_in)
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256], use_in=use_in)
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256], use_in=use_in)
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128], use_in=use_in)
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128], use_in=use_in)
        self.conv1 = nn.Conv1d(128, 128, 1)
        if use_in:
            self.bn1 = nn.InstanceNorm1d(128)
        else:
            self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        final_output_dim = 2
        self.conv2 = nn.Conv1d(128, final_output_dim, 1)

        self.roll = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Output: [gripper_state, ignore_collision]
        )

        self.pitch = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Output: [gripper_state, ignore_collision]
        )

        self.yaw = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Output: [gripper_state, ignore_collision]
        )


    def forward(self, xyz, text_embedding):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 128)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)

        if text_embedding is not None:
            encoded_text = self.text_encoder(text_embedding)  # [B, 128]
            film_params = self.film_predictor(encoded_text)  # [B, 1024 * 2]
            gamma, beta = film_params.chunk(2, dim=1)  # [B, 1024] each
            gamma = gamma.unsqueeze(2)  # [B, 1024, 1] for broadcasting
            beta = beta.unsqueeze(2)  # [B, 1024, 1] for broadcasting
            l6_points = gamma * l6_points + beta  # FiLM modulation: [B, 1024, 16]

        # Pass it through an mlp here
        x = torch.max(l6_points, dim=2)[0]  # Global feature vector (B, 1024)
        
        roll_pred = self.roll(x)  # (B, 2)
        pitch_pred = self.pitch(x)
        yaw_pred = self.yaw(x)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points) # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        # x = F.relu(self.conv1(l0_points))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, roll_pred, pitch_pred, yaw_pred # x shape: B, N, num_classes

class PointNet2GripperBinary(nn.Module):
    def __init__(self, num_classes=2, input_channel=3, use_text_embedding=False):
        super().__init__()
        self.use_text_embedding = use_text_embedding
        self.encoded_text_dim = 128

        # Minimal PointNet++ setup
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=4,  # Same as total number of input points
            radius_list=[0.05],
            nsample_list=[4],
            in_channel=input_channel - 3,
            mlp_list=[[32, 64]],
            use_in=False
        )

        if use_text_embedding:
            self.text_encoder = nn.Linear(1024, self.encoded_text_dim)
            self.film_predictor = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64 * 2)
            )
            self.film_predictor[-1].weight.data.zero_()
            self.film_predictor[-1].bias.data.copy_(
                torch.cat([torch.ones(64), torch.zeros(64)])
            )

        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, xyz, text_embedding=None):
        # xyz: (B, C, N) where N = 4
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz[:, 3:, :] if xyz.shape[1] > 3 else None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # Output: (B, 3, 4), (B, 64, 4)

        if self.use_text_embedding and text_embedding is not None:
            encoded_text = self.text_encoder(text_embedding)  # (B, 128)
            film_params = self.film_predictor(encoded_text)   # (B, 128)
            gamma, beta = film_params.chunk(2, dim=1)
            gamma = gamma.unsqueeze(2)
            beta = beta.unsqueeze(2)
            l1_points = gamma * l1_points + beta

        x = torch.max(l1_points, dim=2)[0]  # Global max pooling: (B, 64)
        x = self.mlp(x)  # (B, num_classes)
        return x

class GripperPointClassifier(nn.Module):
    def __init__(self, num_classes=72, input_dim=3):
        super().__init__()
        # Shared MLP
        self.point_mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU()
        )
        # Separate heads for roll, pitch, yaw
        self.fc_roll = nn.Linear(128, num_classes)
        # self.fc_pitch = nn.Linear(128, num_classes)
        # self.fc_yaw = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, N, 3)
        x = self.point_mlp(x)              # (B, 128, N)
        x = torch.max(x, dim=2)[0]         # (B, 128)
        # roll_logits = self.fc_roll(x)      # (B, num_classes)
        # pitch_logits = self.fc_pitch(x)    # (B, num_classes)
        yaw_logits = self.fc_yaw(x)        # (B, num_classes)
        # return roll_logits, pitch_logits, yaw_logits
        return yaw_logits

class PointNet2_superplus(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2_superplus, self).__init__()
        self.sa0 = PointNetSetAbstractionMsg(npoint=2048, radius_list=[0.0125, 0.025], nsample_list=[16, 32], in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128]])
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.025, 0.05], nsample_list=[16, 32], in_channel=64+128, mlp_list=[[64, 64, 128], [128, 196, 256]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=128+256, mlp_list=[[128, 196, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 256+256, [[256, 384, 512], [256, 384, 512]])
        self.sa4 = PointNetSetAbstractionMsg(128, [0.2, 0.4], [16, 32], 512+512, [[256, 384, 512], [256, 384, 512]])
        self.sa5 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]])
        self.sa6 = PointNetSetAbstractionMsg(16, [0.8, 1.6], [16, 32], 512+512, [[512, 512, 512], [512, 512, 512]])
        self.fp6 = PointNetFeaturePropagation(512+512+512+512, [512, 512, 512])
        self.fp5 = PointNetFeaturePropagation(512+512+512, [512, 512, 512])
        self.fp4 = PointNetFeaturePropagation(512+512+512, [512, 384, 256])
        self.fp3 = PointNetFeaturePropagation(256+256+256, [256, 256, 256])
        self.fp2 = PointNetFeaturePropagation(256+256+128, [256, 128, 128])
        self.fp1 = PointNetFeaturePropagation(128+128+64, [128, 128, 128])
        self.fp0 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l01_xyz, l01_points = self.sa0(l0_xyz, None) # (B, 3, 1024) (B, 96, 1024)
        l1_xyz, l1_points = self.sa1(l01_xyz, l01_points) # (B, 3, 1024) (B, 96, 1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # (B, 3, 16) (B, 1024, 16)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points) # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points) # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # (B, 128, 1024)
        l01_points = self.fp1(l01_xyz, l1_xyz, l01_points, l1_points) # (B, 128, num_point)
        l0_points = self.fp0(l0_xyz, l01_xyz, None, l01_points) # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x # x shape: B, N, num_classes


class PointNet2_text(nn.Module):
    """
    Modified version of PointNet2_super to work with this codebase
    """

    def __init__(self, num_classes, input_channel, keep_gripper_in_fps=False, use_text_embedding=False):
        super(PointNet2_text, self).__init__()
        self.encoded_text_dim = 128  # Output dimension after encoding
        if use_text_embedding:
            self.text_encoder = nn.Linear(
                1024, self.encoded_text_dim
            )  # SIGLIP input dim
            self.film_predictor = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),  # [B, 128] -> [B, 256]
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # [B, 256] -> [B, 2048]
            )
            # Init as gamma=0 and beta=1
            self.film_predictor[-1].weight.data.zero_()
            self.film_predictor[-1].bias.data.copy_(
                torch.cat([torch.ones(1024), torch.zeros(1024)])
            )

        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=96,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            256,
            [0.1, 0.2],
            [16, 32],
            128 + 128,
            [[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa4 = PointNetSetAbstractionMsg(
            128,
            [0.2, 0.4],
            [16, 32],
            256 + 256,
            [[256, 256, 512], [256, 384, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa5 = PointNetSetAbstractionMsg(
            64,
            [0.4, 0.8],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa6 = PointNetSetAbstractionMsg(
            16,
            [0.8, 1.6],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, text_embedding=None):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 256) (B, 512, 256)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)  # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)  # (B, 3, 16) (B, 1024, 16)

        # Apply FiLM conditioning at bottleneck
        encoded_text = self.text_encoder(text_embedding)  # [B, 128]
        film_params = self.film_predictor(encoded_text)  # [B, 1024 * 2]
        gamma, beta = film_params.chunk(2, dim=1)  # [B, 1024] each
        gamma = gamma.unsqueeze(2)  # [B, 1024, 1] for broadcasting
        beta = beta.unsqueeze(2)  # [B, 1024, 1] for broadcasting
        l6_points = gamma * l6_points + beta  # FiLM modulation: [B, 1024, 16]

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x  # x shape: B, N, num_classes

class PointNet2_textV2(nn.Module):
    """
    Modified version of PointNet2_super to work with this codebase
    """

    def __init__(self, num_classes, input_channel, keep_gripper_in_fps=False, use_text_embedding=False):
        super(PointNet2_textV2, self).__init__()
        self.encoded_text_dim = 128  # Output dimension after encoding
        if use_text_embedding:
            self.text_encoder = nn.Linear(
                1024, self.encoded_text_dim
            )  # SIGLIP input dim
            self.film_predictor_mid = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512 * 2),   # for sa3 features
            )
            self.film_predictor_bottleneck = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),  # [B, 128] -> [B, 256]
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # [B, 256] -> [B, 2048]
            )
            # Init FiLM (gamma ~1, beta ~0)
            for film in [self.film_predictor_mid, self.film_predictor_bottleneck]:
                film[-1].weight.data.zero_()
                out_dim = film[-1].bias.shape[0] // 2
                film[-1].bias.data.copy_(torch.cat([torch.ones(out_dim), torch.zeros(out_dim)]))

        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.025, 0.05],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[16, 16, 32], [32, 32, 64]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.05, 0.1],
            nsample_list=[16, 32],
            in_channel=96,
            mlp_list=[[64, 64, 128], [64, 96, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            256,
            [0.1, 0.2],
            [16, 32],
            128 + 128,
            [[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa4 = PointNetSetAbstractionMsg(
            128,
            [0.2, 0.4],
            [16, 32],
            256 + 256,
            [[256, 256, 512], [256, 384, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa5 = PointNetSetAbstractionMsg(
            64,
            [0.4, 0.8],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa6 = PointNetSetAbstractionMsg(
            16,
            [0.8, 1.6],
            [16, 32],
            512 + 512,
            [[512, 512, 512], [512, 512, 512]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, text_embedding=None):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B, 3, 512) (B, 256, 512)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 3, 256) (B, 512, 256)

        # ---- FiLM at mid-level ----
        encoded_text = self.text_encoder(text_embedding)  # [B,128]
        film_params_mid = self.film_predictor_mid(encoded_text)  # [B, 512*2]
        gamma_mid, beta_mid = film_params_mid.chunk(2, dim=1)
        l3_points = gamma_mid.unsqueeze(2) * l3_points + beta_mid.unsqueeze(2)

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B, 3, 128) (B, 1024, 16)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)  # (B, 3, 64) (B , 1024, 64)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)  # (B, 3, 16) (B, 1024, 16)

        # ---- FiLM at bottleneck ----
        film_params_bot = self.film_predictor_bottleneck(encoded_text)  # [B,1024*2]
        gamma_bot, beta_bot = film_params_bot.chunk(2, dim=1)
        l6_points = gamma_bot.unsqueeze(2) * l6_points + beta_bot.unsqueeze(2)

        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x  # x shape: B, N, num_classes

class PointNet2_text_masked(nn.Module):
    """
    Masked-object–tuned PointNet++ with optional FiLM text conditioning.

    Key changes vs your previous version:
      • SA1/SA2 keep higher npoints (1024→512) for detail retention
      • SA1/SA2 radii shrunk roughly ×0.5 for smaller object scale
      • Keep deeper radii moderately large for global context
      • FiLM kept at mid-level (l3). Bottleneck FiLM is optional via flag.
    """

    def __init__(
        self,
        num_classes,
        input_channel,
        keep_gripper_in_fps=False,
        use_text_embedding=True,
        use_bottleneck_film=False,  # default off for masked objects
        encoded_text_dim=128,
    ):
        super().__init__()
        self.use_text_embedding = use_text_embedding
        self.use_bottleneck_film = use_bottleneck_film
        self.encoded_text_dim = encoded_text_dim

        # -----------------------------
        # Text enc + FiLM
        # -----------------------------
        if self.use_text_embedding:
            # Expecting a 1024-d SIGLIP-like embedding; adjust if needed
            self.text_encoder = nn.Linear(1024, self.encoded_text_dim)

            # FiLM at mid (l3) — modulates 512-ch features
            self.film_predictor_mid = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512 * 2),  # gamma|beta for l3 (512 channels)
            )

            if self.use_bottleneck_film:
                # FiLM at bottleneck (l6) — modulates 1024-ch features
                self.film_predictor_bottleneck = nn.Sequential(
                    nn.Linear(self.encoded_text_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 1024 * 2),
                )

            # Init FiLM (gamma≈1, beta≈0) for stability
            for film in [m for m in [getattr(self, 'film_predictor_mid', None), getattr(self, 'film_predictor_bottleneck', None)] if m is not None]:
                with torch.no_grad():
                    film[-1].weight.zero_()
                    out_dim = film[-1].bias.shape[0] // 2
                    film[-1].bias.copy_(torch.cat([torch.ones(out_dim), torch.zeros(out_dim)]))

        # -----------------------------
        # Set Abstraction (MSG) — tuned radii + npoints
        # -----------------------------
        # SA1: from full masked object cloud (~1.5k–3k) to 1024
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=1024,                                 # retain more detail
            radius_list=[0.01, 0.03],                    # 0.025, 0.05 → smaller neighborhoods
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[16, 16, 32], [32, 32, 64]],       # output: 32+64=96
            keep_gripper_in_fps=keep_gripper_in_fps,
        )

        # SA2: 1024 → 512
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=512,                                  # retain more than masked-tuned 256
            radius_list=[0.03, 0.07],                    # 0.05, 0.1 → smaller
            nsample_list=[16, 32],
            in_channel=96,                               # from SA1
            mlp_list=[[64, 64, 128], [64, 96, 128]],     # output: 128+128=256
            keep_gripper_in_fps=keep_gripper_in_fps,
        )

        # SA3: keep similar channels; modestly smaller radii
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=256,                                  # keep consistent hierarchy
            radius_list=[0.07, 0.14],                    # 0.1, 0.2 → modestly smaller
            nsample_list=[16, 32],
            in_channel=256,                              # 128 + 128
            mlp_list=[[128, 196, 256], [128, 196, 256]], # output: 256+256=512
            keep_gripper_in_fps=keep_gripper_in_fps,
        )

        # SA4: global-ish start; slightly smaller radii than original
        self.sa4 = PointNetSetAbstractionMsg(
            npoint=128,                                  # keep same as original
            radius_list=[0.16, 0.32],                    # 0.2, 0.4 → slightly smaller
            nsample_list=[16, 32],
            in_channel=512,                              # 256 + 256
            mlp_list=[[256, 256, 512], [256, 384, 512]], # output: 512+512=1024
            keep_gripper_in_fps=keep_gripper_in_fps,
        )

        # SA5: keep as-is (sufficiently global already)
        self.sa5 = PointNetSetAbstractionMsg(
            npoint=64,
            radius_list=[0.32, 0.64],                    # 0.4, 0.8 → slightly smaller
            nsample_list=[16, 32],
            in_channel=1024,                             # 512 + 512
            mlp_list=[[512, 512, 512], [512, 512, 512]], # output: 512+512=1024
            keep_gripper_in_fps=keep_gripper_in_fps,
        )

        # SA6: final bottleneck
        self.sa6 = PointNetSetAbstractionMsg(
            npoint=16,
            radius_list=[0.64, 1.28],                    # 0.8, 1.6 → slightly smaller
            nsample_list=[16, 32],
            in_channel=1024,                             # 512 + 512
            mlp_list=[[512, 512, 512], [512, 512, 512]], # output: 512+512=1024
            keep_gripper_in_fps=keep_gripper_in_fps,
        )

        # -----------------------------
        # Feature Propagation (channels unchanged)
        # -----------------------------
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, text_embedding=None):
        # xyz: [B, C, N]; first 3 are XYZ
        l0_xyz = xyz[:, :3, :]

        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])   # Bx3x1024, Bx96x1024
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)           # Bx3x512, Bx256x512
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)           # Bx3x256, Bx512x256

        # ---- FiLM @ mid (l3) ----
        if self.use_text_embedding:
            assert text_embedding is not None, "text_embedding must be provided when use_text_embedding=True"
            encoded_text = self.text_encoder(text_embedding)      # [B, encoded_text_dim]
            film_params_mid = self.film_predictor_mid(encoded_text)
            gamma_mid, beta_mid = film_params_mid.chunk(2, dim=1) # each [B,512]
            l3_points = gamma_mid.unsqueeze(2) * l3_points + beta_mid.unsqueeze(2)

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)           # Bx3x128,  Bx1024x128
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)           # Bx3x64,  Bx1024x64
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)           # Bx3x16,  Bx1024x16

        # ---- Optional FiLM @ bottleneck (l6) ----
        if self.use_text_embedding and self.use_bottleneck_film:
            film_params_bot = self.film_predictor_bottleneck(encoded_text)
            gamma_bot, beta_bot = film_params_bot.chunk(2, dim=1) # each [B,1024]
            l6_points = gamma_bot.unsqueeze(2) * l6_points + beta_bot.unsqueeze(2)

        # Feature Propagation back to original resolution
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # Bx512x64
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # Bx512x128
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # Bx256x256
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # Bx256x512
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # Bx128x1024
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)       # Bx128xN

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)                                           # BxC_outxN
        x = x.permute(0, 2, 1)                                      # BxN x C_out
        return x


class PointNet2_text_10k(nn.Module):
    def __init__(self, num_classes, input_channel, keep_gripper_in_fps=False, use_text_embedding=False):
        super(PointNet2_text_10k, self).__init__()
        self.encoded_text_dim = 128  
        if use_text_embedding:
            self.text_encoder = nn.Linear(1024, self.encoded_text_dim)  
            self.film_predictor_mid = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512 * 2),   # for sa3 features
            )
            self.film_predictor_bottleneck = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # for sa6 features
            )
            # Init FiLM (gamma ~1, beta ~0)
            for film in [self.film_predictor_mid, self.film_predictor_bottleneck]:
                film[-1].weight.data.zero_()
                out_dim = film[-1].bias.shape[0] // 2
                film[-1].bias.data.copy_(torch.cat([torch.ones(out_dim), torch.zeros(out_dim)]))

        # Adjusted SA layers
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=2048,
            radius_list=[0.017, 0.033],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[32, 64], [64, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.033, 0.067],
            nsample_list=[16, 32],
            in_channel=192,
            mlp_list=[[64, 128], [64, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.067, 0.133],
            nsample_list=[32, 64],
            in_channel=128 + 128,
            mlp_list=[[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        # (keep sa4–sa6 as in original, just shrink radii)
        self.sa4 = PointNetSetAbstractionMsg(
            256, [0.133, 0.267], [32, 64], 512, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps
        )
        self.sa5 = PointNetSetAbstractionMsg(
            128, [0.267, 0.533], [32, 64], 1024, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps
        )
        self.sa6 = PointNetSetAbstractionMsg(
            32, [0.533, 1.067], [32, 64], 1024, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(64 + 128 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        
    def forward(self, xyz, text_embedding=None):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # normalize RGB if present
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)


        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # ---- FiLM at mid-level ----
        encoded_text = self.text_encoder(text_embedding)  # [B,128]
        film_params_mid = self.film_predictor_mid(encoded_text)  # [B, 512*2]
        gamma_mid, beta_mid = film_params_mid.chunk(2, dim=1)
        l3_points = gamma_mid.unsqueeze(2) * l3_points + beta_mid.unsqueeze(2)

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)

        # ---- FiLM at bottleneck ----
        film_params_bot = self.film_predictor_bottleneck(encoded_text)  # [B,1024*2]
        gamma_bot, beta_bot = film_params_bot.chunk(2, dim=1)
        l6_points = gamma_bot.unsqueeze(2) * l6_points + beta_bot.unsqueeze(2)

        # (feature propagation same as before)
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x  # x shape: B, N, num_classes
    


class PointNet2_10k_discretized(nn.Module):
    def __init__(self, num_classes, input_channel, keep_gripper_in_fps=False, use_text_embedding=False):
        super().__init__()
        self.encoded_text_dim = 128  
        if use_text_embedding:
            self.text_encoder = nn.Linear(1024, self.encoded_text_dim)  
            self.film_predictor_mid = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512 * 2),   # for sa3 features
            )
            self.film_predictor_bottleneck = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # for sa6 features
            )
            # Init FiLM (gamma ~1, beta ~0)
            for film in [self.film_predictor_mid, self.film_predictor_bottleneck]:
                film[-1].weight.data.zero_()
                out_dim = film[-1].bias.shape[0] // 2
                film[-1].bias.data.copy_(torch.cat([torch.ones(out_dim), torch.zeros(out_dim)]))

        # Adjusted SA layers
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=2048,
            radius_list=[0.017, 0.033],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[32, 64], [64, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.033, 0.067],
            nsample_list=[16, 32],
            in_channel=192,
            mlp_list=[[64, 128], [64, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.067, 0.133],
            nsample_list=[32, 64],
            in_channel=128 + 128,
            mlp_list=[[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        # (keep sa4–sa6 as in original, just shrink radii)
        self.sa4 = PointNetSetAbstractionMsg(
            256, [0.133, 0.267], [32, 64], 512, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps
        )
        self.sa5 = PointNetSetAbstractionMsg(
            128, [0.267, 0.533], [32, 64], 1024, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps
        )
        self.sa6 = PointNetSetAbstractionMsg(
            32, [0.533, 1.067], [32, 64], 1024, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(64 + 128 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        output_dim_num = 4
        self.conv2 = nn.Conv1d(128, output_dim_num, 1)


        self.roll = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Output: [gripper_state, ignore_collision]
        )

        self.pitch = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Output: [gripper_state, ignore_collision]
        )

        self.yaw = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # Output: [gripper_state, ignore_collision]
        )
        
    def forward(self, xyz, text_embedding=None):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # normalize RGB if present
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)


        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # ---- FiLM at mid-level ----
        encoded_text = self.text_encoder(text_embedding)  # [B,128]
        film_params_mid = self.film_predictor_mid(encoded_text)  # [B, 512*2]
        gamma_mid, beta_mid = film_params_mid.chunk(2, dim=1)
        l3_points = gamma_mid.unsqueeze(2) * l3_points + beta_mid.unsqueeze(2)

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)

        # ---- FiLM at bottleneck ----
        film_params_bot = self.film_predictor_bottleneck(encoded_text)  # [B,1024*2]
        gamma_bot, beta_bot = film_params_bot.chunk(2, dim=1)
        l6_points = gamma_bot.unsqueeze(2) * l6_points + beta_bot.unsqueeze(2)

        # Pass it through an mlp here
        global_feat = torch.max(l6_points, dim=2)[0]  # Global feature vector (B, 1024)
        
        roll_pred = self.roll(global_feat)  # (B, 72)
        pitch_pred = self.pitch(global_feat)  # (B, 72)
        yaw_pred = self.yaw(global_feat)

        # (feature propagation same as before)
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, roll_pred, pitch_pred, yaw_pred  # x shape: B, N, num_classes

# class OrientationHead(nn.Module):
#     def __init__(self, in_dim, num_bins):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(in_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
#         self.roll = nn.Linear(128, num_bins)
#         self.pitch = nn.Linear(128, num_bins//2)
#         self.yaw = nn.Linear(128, num_bins)

#     def forward(self, x):
#         h = self.fc(x)
        
#         return self.roll(h), self.pitch(h), self.yaw(h)

# class GripperOrientNet(nn.Module):
#     def __init__(self, num_classes, input_channel, keep_gripper_in_fps, use_text_embedding=False):
#         super().__init__()
#         # ----- backbone (same as PointNet2_text_10k) -----
#         self.backbone = PointNet2_text_10k(num_classes, input_channel, keep_gripper_in_fps=keep_gripper_in_fps, use_text_embedding=use_text_embedding)
#         num_bins = 72

#         # orientation head on global feature
#         self.orientation_head = OrientationHead(in_dim=67, num_bins=num_bins)

#     def forward(self, xyz, text_embedding=None, return_orientation=False, weight=None, feat=None):
#         if return_orientation:
#             return self.orientation_head(xyz)
        
#         # ----- extract features -----
#         breakpoint()
#         x = self.backbone(xyz, text_embedding)  # displacement + weight + features
#         return x


class GripperOrientNet(nn.Module):
    def __init__(self, num_classes, input_channel, keep_gripper_in_fps=False, use_text_embedding=False):
        super().__init__()
        self.encoded_text_dim = 128  
        if use_text_embedding:
            self.text_encoder = nn.Linear(1024, self.encoded_text_dim)  
            self.film_predictor_mid = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512 * 2),   # for sa3 features
            )
            self.film_predictor_bottleneck = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # for sa6 features
            )
            # Init FiLM (gamma ~1, beta ~0)
            for film in [self.film_predictor_mid, self.film_predictor_bottleneck]:
                film[-1].weight.data.zero_()
                out_dim = film[-1].bias.shape[0] // 2
                film[-1].bias.data.copy_(torch.cat([torch.ones(out_dim), torch.zeros(out_dim)]))

        self.in_dim = 67
        self.num_bins = 72

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.roll = nn.Linear(128, self.num_bins)
        self.pitch = nn.Linear(128, self.num_bins//2)
        self.yaw = nn.Linear(128, self.num_bins)

        # Adjusted SA layers
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=2048,
            radius_list=[0.017, 0.033],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[32, 64], [64, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.033, 0.067],
            nsample_list=[16, 32],
            in_channel=192,
            mlp_list=[[64, 128], [64, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.067, 0.133],
            nsample_list=[32, 64],
            in_channel=128 + 128,
            mlp_list=[[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        # (keep sa4–sa6 as in original, just shrink radii)
        self.sa4 = PointNetSetAbstractionMsg(
            256, [0.133, 0.267], [32, 64], 512, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps
        )
        self.sa5 = PointNetSetAbstractionMsg(
            128, [0.267, 0.533], [32, 64], 1024, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps
        )
        self.sa6 = PointNetSetAbstractionMsg(
            32, [0.533, 1.067], [32, 64], 1024, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(64 + 128 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        
    def forward(self, xyz=None, text_embedding=None, feats=None,  return_orientation=False):

        if return_orientation:
            feats = feats.clone().detach()
            h = self.fc(feats)
            return self.roll(h), self.pitch(h), self.yaw(h)
        
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # normalize RGB if present
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)


        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # ---- FiLM at mid-level ----
        encoded_text = self.text_encoder(text_embedding)  # [B,128]
        film_params_mid = self.film_predictor_mid(encoded_text)  # [B, 512*2]
        gamma_mid, beta_mid = film_params_mid.chunk(2, dim=1)
        l3_points = gamma_mid.unsqueeze(2) * l3_points + beta_mid.unsqueeze(2)

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)

        # ---- FiLM at bottleneck ----
        film_params_bot = self.film_predictor_bottleneck(encoded_text)  # [B,1024*2]
        gamma_bot, beta_bot = film_params_bot.chunk(2, dim=1)
        l6_points = gamma_bot.unsqueeze(2) * l6_points + beta_bot.unsqueeze(2)

        # (feature propagation same as before)
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        # l0_points = l0_points.clone()
        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        weight = torch.softmax(x[:, :, 3], dim=1)
        per_point_feats = x[:, :, 4:]
        pos_prediction = x[:, :, :3] + l0_xyz.permute(0, 2, 1)
        pos_prediction = pos_prediction * weight.unsqueeze(-1)
        pos_prediction = pos_prediction.sum(dim=1)

        max_pooled = torch.max(per_point_feats, dim=1).values
        weighted_avg = torch.sum(per_point_feats * weight.unsqueeze(-1), dim=1)
        orient_in = torch.cat(
            [pos_prediction, max_pooled, weighted_avg], dim=1
        )

        h = self.fc(orient_in)
        roll = self.roll(h)
        pitch = self.pitch(h)
        yaw = self.yaw(h)

        return x[..., :3], pos_prediction, roll, pitch, yaw  # x shape: B, N, num_classes

class GripperOrientNet_4points(nn.Module):
    def __init__(self, num_classes, input_channel, keep_gripper_in_fps=False, use_text_embedding=False):
        super().__init__()
        self.encoded_text_dim = 128  
        if use_text_embedding:
            self.text_encoder = nn.Linear(1024, self.encoded_text_dim)  
            self.film_predictor_mid = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512 * 2),   # for sa3 features
            )
            self.film_predictor_bottleneck = nn.Sequential(
                nn.Linear(self.encoded_text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1024 * 2),  # for sa6 features
            )
            # Init FiLM (gamma ~1, beta ~0)
            for film in [self.film_predictor_mid, self.film_predictor_bottleneck]:
                film[-1].weight.data.zero_()
                out_dim = film[-1].bias.shape[0] // 2
                film[-1].bias.data.copy_(torch.cat([torch.ones(out_dim), torch.zeros(out_dim)]))

        self.in_dim = 76 # 12 + 32 + 32
        self.num_bins = 72

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.roll = nn.Linear(128, self.num_bins)
        self.pitch = nn.Linear(128, self.num_bins//2)
        self.yaw = nn.Linear(128, self.num_bins)

        # Adjusted SA layers
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=2048,
            radius_list=[0.017, 0.033],
            nsample_list=[16, 32],
            in_channel=input_channel - 3,
            mlp_list=[[32, 64], [64, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=1024,
            radius_list=[0.033, 0.067],
            nsample_list=[16, 32],
            in_channel=192,
            mlp_list=[[64, 128], [64, 128]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.067, 0.133],
            nsample_list=[32, 64],
            in_channel=128 + 128,
            mlp_list=[[128, 196, 256], [128, 196, 256]],
            keep_gripper_in_fps=keep_gripper_in_fps,
        )
        # (keep sa4–sa6 as in original, just shrink radii)
        self.sa4 = PointNetSetAbstractionMsg(
            256, [0.133, 0.267], [32, 64], 512, [[256, 256, 512], [256, 384, 512]], keep_gripper_in_fps
        )
        self.sa5 = PointNetSetAbstractionMsg(
            128, [0.267, 0.533], [32, 64], 1024, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps
        )
        self.sa6 = PointNetSetAbstractionMsg(
            32, [0.533, 1.067], [32, 64], 1024, [[512, 512, 512], [512, 512, 512]], keep_gripper_in_fps
        )
        self.fp6 = PointNetFeaturePropagation(512 + 512 + 512 + 512, [512, 512])
        self.fp5 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [512, 512])
        self.fp4 = PointNetFeaturePropagation(1024, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(64 + 128 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.weight_num_classes = 1
        self.conv1_weight = nn.Conv1d(128, 128, 1)
        self.bn1_weight = nn.BatchNorm1d(128)
        self.conv2_weight = nn.Conv1d(128, self.weight_num_classes, 1)

        self.displacement_num_classes = 12
        self.conv1_displacement = nn.Conv1d(128, 128, 1)
        self.bn1_displacement = nn.BatchNorm1d(128)
        self.conv2_displacement = nn.Conv1d(128, self.displacement_num_classes, 1)

        self.orient_features_num_classes = 32
        self.conv1_orient_features = nn.Conv1d(128, 128, 1)
        self.bn1_orient_features = nn.BatchNorm1d(128)
        self.conv2_orient_features = nn.Conv1d(128, self.orient_features_num_classes, 1)
        
    def forward(self, xyz=None, text_embedding=None, feats=None,  return_orientation=False):

        if return_orientation:
            feats = feats.clone().detach()
            h = self.fc(feats)
            return self.roll(h), self.pitch(h), self.yaw(h)
        
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        # normalize RGB if present
        if xyz.shape[1] > 3:
            l1_xyz, l1_points = self.sa1(l0_xyz, xyz[:, 3:, :])
        else:
            l1_xyz, l1_points = self.sa1(l0_xyz, None)  # (B, 3, 1024) (B, 96, 1024)


        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # ---- FiLM at mid-level ----
        encoded_text = self.text_encoder(text_embedding)  # [B,128]
        film_params_mid = self.film_predictor_mid(encoded_text)  # [B, 512*2]
        gamma_mid, beta_mid = film_params_mid.chunk(2, dim=1)
        l3_points = gamma_mid.unsqueeze(2) * l3_points + beta_mid.unsqueeze(2)

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)

        # ---- FiLM at bottleneck ----
        film_params_bot = self.film_predictor_bottleneck(encoded_text)  # [B,1024*2]
        gamma_bot, beta_bot = film_params_bot.chunk(2, dim=1)
        l6_points = gamma_bot.unsqueeze(2) * l6_points + beta_bot.unsqueeze(2)

        # (feature propagation same as before)
        l5_points = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)  # (B, 512, 64)
        l4_points = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points)  # (B, 512, 128)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B, 256, 256)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B, 256, 512)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # (B, 128, 1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # (B, 128, num_point)

        # # l0_points = l0_points.clone()
        # x = F.relu(self.bn1(self.conv1(l0_points)))
        # x = self.conv2(x)
        # # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        
        # three separate conv layers for weight, displacement, orientation features
        weight = F.relu(self.bn1_weight(self.conv1_weight(l0_points)))
        weight = self.conv2_weight(weight)
        weight = weight.permute(0, 2, 1)    
        weight = F.softmax(weight, dim=1)

        displacement = F.relu(self.bn1_displacement(self.conv1_displacement(l0_points)))
        displacement = self.conv2_displacement(displacement)
        displacement = displacement.permute(0, 2, 1)

        per_point_feats = F.relu(self.bn1_orient_features(self.conv1_orient_features(l0_points)))
        per_point_feats = self.conv2_orient_features(per_point_feats)
        per_point_feats = per_point_feats.permute(0, 2, 1)

        pcd_prediction = displacement.view(displacement.shape[0], displacement.shape[1], 4, 3) + l0_xyz.permute(0, 2, 1).unsqueeze(2)
        pcd_prediction = pcd_prediction * weight.unsqueeze(-1)
        pcd_prediction = pcd_prediction.sum(dim=1)

        max_pooled = torch.max(per_point_feats, dim=1).values
        weighted_avg = torch.sum(per_point_feats * weight, dim=1)
        orient_in = torch.cat(
            [pcd_prediction.view(pcd_prediction.shape[0], -1), max_pooled, weighted_avg], dim=1
        )
        h = self.fc(orient_in)
        roll = self.roll(h)
        pitch = self.pitch(h)
        yaw = self.yaw(h)

        return displacement, pcd_prediction, roll, pitch, yaw  # x shape: B, N, num_classes
    
if __name__ == '__main__':

    from tqdm import tqdm
    model = PointNet2(num_classes=10).cuda()
    model.eval()
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.deterministic = True
    inpput = torch.rand(1, 3, 2000).cuda()
    out = model(inpput)
    max_diff = -1
    for _ in range(1):
        inpput_translated = inpput + 50
        out_translated = model(inpput_translated)
        diff = torch.norm(out-out_translated)
        max_diff = max(max_diff, diff)
        print("difference: ", diff)