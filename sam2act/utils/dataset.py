
# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
import os
import torch
import pickle
import logging
import numpy as np
from typing import List
import open3d as o3d

import clip
import peract_colab.arm.utils as utils
from sam2act.libs.peract.helpers import utils as peract_helper_utils

from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer_temporal import UniformReplayBuffer_temporal
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

import sam2act.utils.peract_utils as peract_utils
import sam2act.mvt.utils as mvt_utils
import sam2act.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils
from sam2act.mvt.mvt_sam2_single import MVT_SAM2_Single

from sam2act.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS
from sam2act.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from sam2act.libs.peract.helpers.utils import extract_obs
from third_party.robogen.robogen_utils import rotation_transfer_matrix_to_6D_batch, rotation_transfer_matrix_to_6D, \
                          get_4_points_from_gripper_pos_orient

from eval import load_agent
from sam2act.libs.PyRep.pyrep.objects.vision_sensor import VisionSensor

def create_replay(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    replay_size=3e5,
):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement(
                "lang_goal_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
    ]

    replay_buffer = (
        UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    )
    return replay_buffer


def create_replay_temporal(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    num_maskmem,
    replay_size=3e5,
):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement(
                "lang_goal_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
        ReplayElement("initial_frame", (), int),
    ]

    replay_buffer = (
        UniformReplayBuffer_temporal(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
            num_maskmem=num_maskmem,
        )
    )
    return replay_buffer



# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    articubot_dataset=False
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    if articubot_dataset:
        return (
            trans_indicies,
            rot_and_grip_indicies,
            ignore_collisions,
            np.concatenate([obs_tp1.gripper_pose, np.array([grip]), np.array([ignore_collisions])]),
            attention_coordinates,
        )
    else:
        return (
            trans_indicies,
            rot_and_grip_indicies,
            ignore_collisions,
            np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
            attention_coordinates,
        )



# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb
# add individual data points to a replay
def _create_articubot_dataset(
    task, obs, episode_num, sample_frame, key_frame_obs, frame_before_keyframe, action, lang_feats, val
):
    folder_name = 'episode_' + str(episode_num)
    print(episode_num, sample_frame)
    
    
    front_pcd = obs.front_point_cloud.reshape(-1, 3)
    wrist_pcd = obs.wrist_point_cloud.reshape(-1, 3)
    left_shoulder_pcd = obs.left_shoulder_point_cloud.reshape(-1, 3)
    right_shoulder_pcd = obs.right_shoulder_point_cloud.reshape(-1, 3)

    front_rgb = obs.front_rgb.reshape(-1, 3) / 255.0
    wrist_rgb = obs.wrist_rgb.reshape(-1, 3) / 255.0
    left_shoulder_rgb = obs.left_shoulder_rgb.reshape(-1, 3) / 255.0
    right_shoulder_rgb = obs.right_shoulder_rgb.reshape(-1, 3) / 255.0

    all_pcd = np.concatenate([front_pcd, wrist_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    all_rgb = np.concatenate([front_rgb, wrist_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)

    x_range = (-2.06492364, 2.26651619)
    y_range = (-0.96348435, 1.00034714)
    z_range = (0.3, 1.72072086)

    mask = (
    (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
    (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
    (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    )

    # def create_reference_planes_with_colors(x_range, y_range, z_range, num_points_per_axis=100):
    #     # Create linspaces
    #     x = np.linspace(*x_range, num_points_per_axis)
    #     y = np.linspace(*y_range, num_points_per_axis)
    #     z = np.linspace(*z_range, num_points_per_axis)

    #     # Create meshgrids for each plane

    #     # XY Plane (z = z_min)
    #     xx_xy, yy_xy = np.meshgrid(x, y)
    #     zz_xy = np.full_like(xx_xy, z_range[0])
    #     xy_plane = np.stack([xx_xy, yy_xy, zz_xy], axis=-1).reshape(-1, 3)
    #     xy_color = np.tile(np.array([[1.0, 0.0, 0.0]]), (xy_plane.shape[0], 1))  # Red

    #     # YZ Plane (x = x_min)
    #     yy_yz, zz_yz = np.meshgrid(y, z)
    #     xx_yz = np.full_like(yy_yz, x_range[0])
    #     yz_plane = np.stack([xx_yz, yy_yz, zz_yz], axis=-1).reshape(-1, 3)
    #     yz_color = np.tile(np.array([[0.0, 1.0, 0.0]]), (yz_plane.shape[0], 1))  # Green

    #     # ZX Plane (y = y_min)
    #     zz_zx, xx_zx = np.meshgrid(z, x)
    #     yy_zx = np.full_like(zz_zx, y_range[0])
    #     zx_plane = np.stack([xx_zx, yy_zx, zz_zx], axis=-1).reshape(-1, 3)
    #     zx_color = np.tile(np.array([[0.0, 0.0, 1.0]]), (zx_plane.shape[0], 1))  # Blue

    #     # Concatenate all planes and their colors
    #     all_planes = np.concatenate([xy_plane, yz_plane, zx_plane], axis=0)
    #     all_colors = np.concatenate([xy_color, yz_color, zx_color], axis=0)

    #     return all_planes, all_colors

    # # Example usage:
    # # Define bounding ranges based on your pointcloud limits
    # x_range = ( -2.06492364, 2.26651619)
    # y_range = (-0.96348435, 1.00034714)
    # z_range = (0.3, 1.72072086)

    # reference_planes, reference_colors = create_reference_planes_with_colors(x_range, y_range, z_range, num_points_per_axis=50)

    np_points= all_pcd[mask]
    np_rgb = all_rgb[mask]

    # np_points = np.concatenate([np_points, reference_planes], axis=0)
    # np_rgb = np.concatenate([np_rgb, reference_colors], axis=0)

    # rand_indx = np.random.choice(all_pcd.shape[0], 30000)
    # np_points = all_pcd[rand_indx]
    # np_rgb = all_rgb[rand_indx]    

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

    sampled_pcd = obj_pcd.farthest_point_down_sample(4500)

    sampled_points = np.asarray(sampled_pcd.points)
    sampled_rgb = np.asarray(sampled_pcd.colors)
    point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)

    data = {'point_cloud': np.expand_dims(point_cloud, axis=0), 
            'action': action, 'gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(obs.gripper_pose[:3], obs.gripper_pose[3:7], obs.gripper_joint_positions[1]), axis=0),
            'goal_gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(key_frame_obs.gripper_pose[:3], key_frame_obs.gripper_pose[3:7], frame_before_keyframe.gripper_joint_positions[1]), axis=0),
            'state': obs.get_low_dim_data(),
            'lang_feats': lang_feats,}
    
    if val:
        directory = os.path.join('data_articubot', task + '_val', folder_name)
    else:
        directory = os.path.join('data_articubot', task + '_temp', folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    final_path = os.path.join(directory, str(sample_frame) + '.pkl')
    with open(final_path, 'wb') as f:
        print('Saving data to: ', final_path)
        pickle.dump(data, f)

# Create articubot dataset for each frame
def _create_featurized_dataset(
    task, obs, episode_num, sample_frame, key_frame_obs, action, agent, obs_dict, lang_feats, val
):
    # Construct point cloud
    folder_name = 'episode_' + str(episode_num)
    print(episode_num, sample_frame)
    
    # front_pcd = obs.front_point_cloud.reshape(-1, 3)
    # wrist_pcd = obs.wrist_point_cloud.reshape(-1, 3)
    # left_shoulder_pcd = obs.left_shoulder_point_cloud.reshape(-1, 3)
    # right_shoulder_pcd = obs.right_shoulder_point_cloud.reshape(-1, 3)

    # front_rgb = obs.front_rgb.reshape(-1, 3) / 255.0
    # wrist_rgb = obs.wrist_rgb.reshape(-1, 3) / 255.0
    # left_shoulder_rgb = obs.left_shoulder_rgb.reshape(-1, 3) / 255.0
    # right_shoulder_rgb = obs.right_shoulder_rgb.reshape(-1, 3) / 255.0

    # all_pcd = np.concatenate([front_pcd, wrist_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    # all_rgb = np.concatenate([front_rgb, wrist_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)

    # Getting SAM2 features
    proprio = arm_utils.stack_on_channel(obs_dict['low_dim_state'])

    new_obs, pcd = peract_utils._preprocess_inputs(obs_dict, agent.cameras)

    pc, img_feat = rvt_utils.get_pc_img_feat(
        new_obs,
        pcd,
    )

    pc, img_feat = rvt_utils.move_pc_in_bound(
        pc, img_feat, agent.scene_bounds, no_op=not agent.move_pc_in_bound
    )

    img = agent._network.render(
                pc=pc,
                img_feat=img_feat,
                img_aug=0,
                mvt1_or_mvt2=True,
                dyn_cam_info=None,
            )
    out = agent._network.mvt1(
            img=img,
            proprio=proprio,
            lang_emb=None,
            wpt_local=None,
            rot_x_y=None,
            # hm_gt=hm_gt,
    ) # 3, 32, 64, 64

    upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

    out = upsample[0](out) # 3, 32, 128, 128

    front_point_cloud, front_feature = backproject_sam2_features_to_3d(out[0], obs_dict['front_depth'].reshape(-1, *obs_dict['front_depth'].shape[4:]), obs_dict['front_camera_intrinsics'].reshape(-1, *obs_dict['front_camera_intrinsics'].shape[3:]), obs_dict['front_camera_extrinsics'].reshape(-1, *obs_dict['front_camera_extrinsics'].shape[3:])) # N, 32
    left_point_cloud, left_feature = backproject_sam2_features_to_3d(out[1], obs_dict['left_shoulder_depth'].reshape(-1, *obs_dict['left_shoulder_depth'].shape[4:]), obs_dict['left_shoulder_camera_intrinsics'].reshape(-1, *obs_dict['left_shoulder_camera_intrinsics'].shape[3:]), obs_dict['left_shoulder_camera_extrinsics'].reshape(-1, *obs_dict['left_shoulder_camera_extrinsics'].shape[3:])) # N, 32
    right_point_cloud, right_feature = backproject_sam2_features_to_3d(out[2], obs_dict['right_shoulder_depth'].reshape(-1, *obs_dict['right_shoulder_depth'].shape[4:]), obs_dict['right_shoulder_camera_intrinsics'].reshape(-1, *obs_dict['right_shoulder_camera_intrinsics'].shape[3:]), obs_dict['right_shoulder_camera_extrinsics'].reshape(-1, *obs_dict['right_shoulder_camera_extrinsics'].shape[3:])) # N, 32
    # front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs_dict['front_depth'].detach().cpu().numpy(),
    #                                                                          obs_dict['left_shoulder_camera_extrinsics'].detach().cpu().numpy(),
    #                                                                          obs_dict['left_shoulder_camera_intrinsics'].detach().cpu().numpy())
    all_pcd = torch.concat([front_point_cloud, left_point_cloud, right_point_cloud], axis=0) #.detach().cpu().numpy()
    all_features = torch.concat([front_feature, left_feature, right_feature], axis=0) #.detach().cpu().numpy()

    x_range = (-2.06492364, 2.26651619)
    y_range = (-0.96348435, 1.00034714)
    z_range = (0.3, 1.72072086)

    mask = (
    (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
    (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
    (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    )

    np_points = all_pcd[mask]
    np_rgb = all_features[mask]


    # def furthest_point_sampling(points, num_samples):
    #     """
    #     Args:
    #         points (np.ndarray): Input point cloud, shape (N, 3)
    #         num_samples (int): Number of points to sample

    #     Returns:
    #         sampled_indices (np.ndarray): Indices of the sampled points, shape (num_samples,)
    #     """
    #     N = points.shape[0]
    #     sampled_indices = np.zeros(num_samples, dtype=np.int64)
    #     distances = np.full(N, np.inf)

    #     # Randomly select the first point
    #     farthest_index = np.random.randint(0, N)
    #     sampled_indices[0] = farthest_index

    #     for i in range(1, num_samples):
    #         # Compute distances from the current farthest point to all other points
    #         current_point = points[farthest_index]
    #         dist = np.linalg.norm(points - current_point, axis=1)

    #         # Update the minimum distances to the sampled points
    #         distances = np.minimum(distances, dist)

    #         # Select the point with the maximum minimum distance
    #         farthest_index = np.argmax(distances)
    #         sampled_indices[i] = farthest_index

    #     return sampled_indices
    
    def furthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Args:
            points (torch.Tensor): Input point cloud, shape (N, 3)
            num_samples (int): Number of points to sample

        Returns:
            sampled_indices (torch.Tensor): Indices of the sampled points, shape (num_samples,)
        """
        device = points.device
        N = points.shape[0]
        sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
        distances = torch.full((N,), float('inf'), device=device)

        # Randomly select the first point
        farthest_index = torch.randint(0, N, (1,), device=device).item()
        sampled_indices[0] = farthest_index

        for i in range(1, num_samples):
            current_point = points[farthest_index].unsqueeze(0)  # Shape (1, 3)
            dist = torch.norm(points - current_point, dim=1)     # Shape (N,)

            # Update the minimum distances
            distances = torch.minimum(distances, dist)

            # Select the next farthest point
            farthest_index = torch.argmax(distances).item()
            sampled_indices[i] = farthest_index

        return sampled_indices
    
    # Randomly sample 30,000 points from the point cloud
    # rand_idxes = np.random.choice(np_points.shape[0], 30000, replace=False)
    # np_points = np_points[rand_idxes]
    # np_rgb = np_rgb[rand_idxes]

    fps = furthest_point_sampling(np_points, 4500)
    np_points = np_points[fps]
    np_rgb = np_rgb[fps]

    # obj_pcd = o3d.geometry.PointCloud()
    # obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    # obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

    # sampled_pcd = obj_pcd.farthest_point_down_sample(4500)

    # sampled_points = np.asarray(sampled_pcd.points)
    # sampled_rgb = np.asarray(sampled_pcd.colors)
    point_cloud = np.concatenate([np_points.detach().cpu().numpy(), np_rgb.detach().cpu().numpy()], axis=1)

    data = {'point_cloud': np.expand_dims(point_cloud, axis=0), 
            'action': action, 'gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(obs.gripper_pose[:3], obs.gripper_pose[3:7], obs.gripper_joint_positions[1]), axis=0),
            'goal_gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(key_frame_obs.gripper_pose[:3], key_frame_obs.gripper_pose[3:7], key_frame_obs.gripper_joint_positions[1]), axis=0),
            'state': obs.get_low_dim_data(),
            'lang_feats': lang_feats,}
    
    if val:
        directory = os.path.join('data_articubot', task + '_featurized_val', folder_name)
    else:
        directory = os.path.join('data_articubot', task + '_featurized', folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    final_path = os.path.join(directory, str(sample_frame) + '.pkl')
    with open(final_path, 'wb') as f:
        print('Saving data to: ', final_path)
        pickle.dump(data, f)

# def reproject_features_to_3d(features, depth, intrinsics):
#     """
#     Reprojects 2D features with a depth map into 3D space.

#     Args:
#         features (torch.Tensor): shape (C, H, W)
#         depth (torch.Tensor): shape (H, W)
#         intrinsics (torch.Tensor): shape (3, 3)

#     Returns:
#         xyz_points (torch.Tensor): shape (H*W, 3)
#         features_3d (torch.Tensor): shape (H*W, C)
#     """
#     assert features.shape[1:] == depth.shape, "Feature and depth resolution mismatch"
    
#     C, H, W = features.shape
#     device = features.device

#     # Mask out invalid depth (e.g., zeros)
#     valid_mask = depth > 0
#     num_valid = valid_mask.sum()

#     # Create pixel grid
#     y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
#     x = x[valid_mask]
#     y = y[valid_mask]
#     z = depth[valid_mask]

#     ones = torch.ones_like(x)
#     pixel_coords = torch.stack([x, y, ones], dim=0).float()  # (3, N)

#     # Inverse intrinsics
#     K_inv = torch.inverse(intrinsics.to(device))

#     # Unproject
#     cam_coords = K_inv @ pixel_coords  # (3, N)
#     cam_coords = cam_coords * z  # scale rays by depth
#     xyz_points = cam_coords.T  # (N, 3)

#     # Get features at valid pixels
#     point_features = features.permute(1, 2, 0)[valid_mask]  # (N, C)

#     return xyz_points, point_features


def backproject_sam2_features_to_3d(features, depths, intrinsics, extrinsics):
    """
    Args:
        features: [3, 32, H, W] tensor of SAM2 features (float32, GPU or CPU)
        depths: [3, H, W] tensor of depth maps (float32, same device)
        intrinsics: [3, 3, 3] tensor of camera intrinsics (float32)
        extrinsics: [3, 4, 4] tensor of camera-to-world transforms (float32)

    Returns:
        points_3d: [N, 3] tensor of 3D world coordinates
        features_3d: [N, 32] tensor of feature vectors
    """
    device = features.device
    feat_dim, H, W = features.shape

    all_points = []
    all_feats = []

    # Create meshgrid once
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')  # shape: [H, W]
    ones = torch.ones_like(uu)

    pixel_coords = torch.stack([uu, vv, ones], dim=0).reshape(3, -1).float()  # [3, H*W]

    K = intrinsics # [3, 3]
    K_inv = torch.inverse(K)
    T = extrinsics # [4, 4]

    depth = depths.reshape(-1)  # [H*W]

    # 3D points in camera coordinates
    cam_coords = K_inv @ pixel_coords * depth.unsqueeze(0)  # [3, H*W]

    # Homogenize
    cam_coords_h = torch.cat([cam_coords, torch.ones(1, cam_coords.shape[1], device=device)], dim=0)  # [4, H*W]

    # Transform to world frame
    world_coords = (T @ cam_coords_h)[:3]  # [3, H*W]

    # Features
    feat = features.reshape(feat_dim, -1)  # [32, H*W]

    # Mask invalid depths
    valid_mask = depth > 0
    world_coords = world_coords[:, valid_mask].T  # [N, 3]
    feat = feat[:, valid_mask].T  # [N, 32]

    all_points.append(world_coords)
    all_feats.append(feat)

    # Combine all views
    points_3d = torch.cat(all_points, dim=0)  # [N_total, 3]
    features_3d = torch.cat(all_feats, dim=0)  # [N_total, 32]

    return points_3d, features_3d

# For rolling out
def _get_articubot_dataset(obs, add_rgb_zeros=False, add_rgb_ones=False, add_one_hot=False, one_hot_dim=3):
    front_pcd = obs['front_point_cloud'].detach().cpu().numpy()
    front_pcd = front_pcd[0, 0].transpose([1,2,0]).reshape(-1, 3)
    wrist_pcd = obs['wrist_point_cloud'].detach().cpu().numpy()
    wrist_pcd = wrist_pcd[0, 0].transpose([1,2,0]).reshape(-1, 3)
    left_shoulder_pcd = obs['left_shoulder_point_cloud'].detach().cpu().numpy()
    left_shoulder_pcd = left_shoulder_pcd[0, 0].transpose([1,2,0]).reshape(-1, 3)    
    right_shoulder_pcd = obs['right_shoulder_point_cloud'].detach().cpu().numpy()
    right_shoulder_pcd = right_shoulder_pcd[0, 0].transpose([1,2,0]).reshape(-1, 3)

    front_rgb = obs['front_rgb'].detach().cpu().numpy()
    front_rgb = front_rgb[0, 0].transpose([1,2,0]).reshape(-1, 3) / 255.0
    wrist_rgb = obs['wrist_rgb'].detach().cpu().numpy()
    wrist_rgb = wrist_rgb[0, 0].transpose([1,2,0]).reshape(-1, 3) / 255.0
    left_shoulder_rgb = obs['left_shoulder_rgb'].detach().cpu().numpy()
    left_shoulder_rgb = left_shoulder_rgb[0, 0].transpose([1,2,0]).reshape(-1, 3) / 255.0
    right_shoulder_rgb = obs['right_shoulder_rgb'].detach().cpu().numpy()
    right_shoulder_rgb = right_shoulder_rgb[0, 0].transpose([1,2,0]).reshape(-1, 3) / 255.0

    all_pcd = np.concatenate([front_pcd, wrist_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    all_rgb = np.concatenate([front_rgb, wrist_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)

    x_range = (-2.06492364, 2.26651619)
    y_range = (-0.96348435, 1.00034714)
    z_range = (0.3, 1.72072086)

    mask = (
    (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
    (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
    (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    )

    np_points= all_pcd[mask]
    np_rgb = all_rgb[mask]

    # rand_indx = np.random.choice(all_pcd.shape[0], 30000)
    # np_points = all_pcd[rand_indx]
    # np_rgb = all_rgb[rand_indx]    

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

    sampled_pcd = obj_pcd.farthest_point_down_sample(4500)

    sampled_points = np.asarray(sampled_pcd.points)
    sampled_rgb = np.asarray(sampled_pcd.colors)

    if add_rgb_zeros or add_rgb_ones:
        point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)
    else: 
        point_cloud = sampled_points

    gripper_pose = obs['gripper_pose'][0][0].detach().cpu().numpy()
    joint_pos = obs['gripper_joint_positions'][0][0].detach().cpu().numpy()

    gripper_pcd = np.expand_dims(get_4_points_from_gripper_pos_orient(gripper_pose[:3], gripper_pose[3:7], joint_pos[1]), axis=0)
    gripper_pcd = torch.from_numpy(gripper_pcd)

    if add_rgb_zeros:
        gripper_pcd = torch.cat([gripper_pcd, torch.zeros(gripper_pcd.shape)], dim=2)

    elif add_rgb_ones:
        gripper_pcd = torch.cat([gripper_pcd, torch.ones(gripper_pcd.shape)], dim=2)

    point_cloud = torch.from_numpy(np.expand_dims(point_cloud, axis=0))

    if add_one_hot:
        pointcloud_one_hot = torch.zeros(point_cloud.shape[0], point_cloud.shape[1], one_hot_dim)
        pointcloud_one_hot[:, :, 0] = 1
        point_cloud = torch.cat([point_cloud, pointcloud_one_hot], dim=2)
        gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], one_hot_dim)
        gripper_pcd_one_hot[:, :, 1] = 1
        gripper_pcd = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
    
    point_cloud = point_cloud.unsqueeze(0)
    gripper_pcd = gripper_pcd.unsqueeze(0)
    
    obs_dict = {'point_cloud': point_cloud,
                'gripper_pcd': gripper_pcd,}
    
    return obs_dict


def visualize(points, predictions):
    point_geometry = o3d.geometry.PointCloud()
    print(points.shape)
    print(predictions.shape)
    point_geometry.points = o3d.utility.Vector3dVector(points[:, :, :, :3].reshape(-1, 3))
    point_geometry.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (4500,1)))

    
    # gripper_geometry = o3d.geometry.PointCloud()
    # gripper_geometry.points = o3d.utility.Vector3dVector(points[1024:1162])
    # gripper_geometry.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (138, 1)))

    four_point_geometry = o3d.geometry.PointCloud()
    four_point_geometry.points = o3d.utility.Vector3dVector(predictions[0, :, :, :].reshape(-1, 3).detach().cpu().numpy())
    four_point_geometry.paint_uniform_color(np.array([0, 1, 0]))
    # four_point_geometry.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 1, 0]]), (4, 1)))

    # gripper_geometry = o3d.geometry.PointCloud()
    # gripper_geometry.points = o3d.utility.Vector3dVector(gripper_pcd[0, :,:3].reshape(-1, 3))
    # gripper_geometry.paint_uniform_color(np.array([0, 0, 1]))

    # goal_gripper_geometry = o3d.geometry.PointCloud()
    # goal_gripper_geometry.points = o3d.utility.Vector3dVector(goal_gripper_pcd[0, :,:3].reshape(-1, 3))
    # goal_gripper_geometry.paint_uniform_color(np.array([0, 0, 0]))

    # o3d.visualization.draw_geometries([point_geometry, four_point_geometry, goal_pcd, mesh_frame])
    o3d.visualization.draw_geometries([point_geometry, four_point_geometry])

# add individual data points to a replay
def _add_keypoints_to_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    device="cpu",
):
    prev_action = None
    obs = inital_obs
    for k in range(
        next_keypoint_idx, len(episode_keypoints)   #  loop in all the following keypoint
    ):  # confused here, it seems that there are many same samples in the replay
        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]    # keypoint frame
        obs_tm1 = demo[max(0, keypoint - 1)]   # frame before keypoint
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(         #  get keypoint action 
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        terminal = k == len(episode_keypoints) - 1    # if is the last keypoint, terminal
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(      #  obs is the i_th frame
            obs,
            CAMERAS,
            t=k - next_keypoint_idx,     # t for calculate time, represent t_th keypoint
            prev_action=prev_action,
            episode_length=25,
        )
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,  # rot + grip: 3+1
            "gripper_pose": obs_tp1.gripper_pose,   # 3+4
            "lang_goal": np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        obs = obs_tp1
        sample_frame = keypoint

    # final step    # FIXME It is no need to do this step?
    obs_dict_tp1 = extract_obs(
        obs_tp1,
        CAMERAS,
        t=k + 1 - next_keypoint_idx,
        prev_action=prev_action,
        episode_length=25,
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop("wrist_world_to_cam", None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)

def fill_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model=None,
    device="cpu",
):
    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...:", task)
        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)

            # extract keypoints
            episode_keypoints = keypoint_discovery(demo)  # list of keypoint   [id0, id1, id2]
            next_keypoint_idx = 0
            for i in range(len(demo) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                    continue

                obs = demo[i]
                desc = descs[0]
                # if our starting point is past one of the keypoints, then remove it
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):
                    break
                _add_keypoints_to_replay(
                    replay,
                    task,
                    task_replay_storage_folder,
                    d_idx,
                    i,
                    obs,
                    demo,
                    episode_keypoints,
                    cameras,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    next_keypoint_idx=next_keypoint_idx,
                    description=desc,
                    clip_model=clip_model,
                    device=device,
                )

        # save TERMINAL info in replay_info.npy
        task_idx = replay._task_index[task]
        with open(
            os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        ) as fp:
            np.save(
                fp,
                replay._store["terminal"][
                    replay._task_replay_start_index[
                        task_idx
                    ] : replay._task_replay_start_index[task_idx]
                    + replay._task_add_count[task_idx].value
                ],
            )

        print("Replay filled with demos.")

def fill_articubot(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model=None,
    sam2feats=False,
    device="cpu",
    val=False,
    args=None,
):

    disk_exist = False
    # if replay._disk_saving:
    #     if os.path.exists(task_replay_storage_folder):
    #         print(
    #             "[Info] Replay dataset already exists in the disk: {}".format(
    #                 task_replay_storage_folder
    #             ),
    #             flush=True,
    #         )
    #         disk_exist = True
    #     else:
    #         logging.info("\t saving to disk: %s", task_replay_storage_folder)
    #         os.makedirs(task_replay_storage_folder, exist_ok=True)

    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...:", task)
        model_path = 'runs/sam2act_rlbench/model_89.pth'

        agent = load_agent(
                model_path=model_path,
                exp_cfg_path=args.exp_cfg_path,
                mvt_cfg_path=args.mvt_cfg_path,
                eval_log_dir=None,
                device=args.device,
                use_input_place_with_mean=False,
                articubot=True,
        )

        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)

            # extract keypoints
            episode_keypoints = keypoint_discovery(demo)  # list of keypoint   [id0, id1, id2]
            next_keypoint_idx = 0
            frame_before_keypoint = demo[0]
            for i in range(len(demo)):
                # if not demo_augmentation and i > 0:
                #     break
                # if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                #     continue
                print(episode_keypoints[next_keypoint_idx])
                obs = demo[i]
                key_frame_obs = demo[episode_keypoints[next_keypoint_idx]]

                if i == episode_keypoints[next_keypoint_idx] and next_keypoint_idx < len(episode_keypoints):
                    next_keypoint_idx = next_keypoint_idx + 1
                    frame_before_keypoint = demo[episode_keypoints[next_keypoint_idx-1]]

                if i >= episode_keypoints[next_keypoint_idx-1]:
                    frame_before_keypoint = demo[i]
                    

                

                

                obs_tp1 = demo[i]    # keypoint frame
                obs_tm1 = demo[max(0, i - 1)]   # frame before keypoint
                (
                    trans_indicies,
                    rot_grip_indicies,
                    ignore_collisions,
                    action,
                    attention_coordinates,
                ) = _get_action(         #  get keypoint action 
                    obs_tp1,
                    obs_tm1,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    articubot_dataset=True,
                )

                tokens = clip.tokenize([descs[0]]).numpy()
                token_tensor = torch.from_numpy(tokens).to(device)
                with torch.no_grad():
                    lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)

                # lang_embs = lang_embs[0].float().detach().cpu().numpy()

                lang_feats = lang_feats[0].float().detach().cpu().numpy()

                if sam2feats:
                    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
                    obs_config = peract_helper_utils.create_obs_config(CAMERAS, camera_resolution, method_name="", use_mask_from_replay=False)

                    obs_dict = extract_obs(      #  obs is the i_th frame
                        obs,
                        CAMERAS,
                        t= next_keypoint_idx,     # t for calculate time, represent t_th keypoint
                        prev_action=None,
                        episode_length=25,
                    )

                    def reshape_dict_arrays_to_tensor(input_dict):
                        """
                        Converts every NumPy array in a dictionary to a PyTorch tensor
                        with shape (1, 1, n), where n is the flattened size of the array.

                        Args:
                            input_dict (dict): Dictionary with NumPy array values.

                        Returns:
                            dict: Dictionary with reshaped torch.Tensor values.
                        """
                        output_dict = {}
                        for key, value in input_dict.items():
                            if isinstance(value, np.ndarray):
                                tensor = torch.tensor(value, dtype=torch.float32, device='cuda:0').unsqueeze(0).unsqueeze(0)
                                output_dict[key] = tensor
                            else:
                                raise TypeError(f"Value for key '{key}' is not a NumPy array.")
                        return output_dict
                    
                    obs_dict = reshape_dict_arrays_to_tensor(obs_dict)
                    _create_featurized_dataset(task, obs, d_idx, i, key_frame_obs, action, agent, obs_dict, lang_feats, val=val)
                else:
                    _create_articubot_dataset(task, obs, d_idx, i, key_frame_obs, frame_before_keypoint, action, lang_feats, val=val)

                # desc = descs[0]
                # if our starting point is past one of the keypoints, then remove it
                # while (
                #     next_keypoint_idx < len(episode_keypoints)
                #     and i >= episode_keypoints[next_keypoint_idx]
                # ):
                #     next_keypoint_idx += 1
                # if next_keypoint_idx == len(episode_keypoints):
                #     break
                # _create_articubot_dataset(
                #     replay,
                #     task,
                #     task_replay_storage_folder,
                #     d_idx,
                #     i,
                #     obs,
                #     demo,
                #     episode_keypoints,
                #     cameras,
                #     rlbench_scene_bounds,
                #     voxel_sizes,
                #     rotation_resolution,
                #     crop_augmentation,
                #     next_keypoint_idx=next_keypoint_idx,
                #     description=desc,
                #     clip_model=clip_model,
                #     device=device,
                # )

        # save TERMINAL info in replay_info.npy
        # task_idx = replay._task_index[task]
        # with open(
        #     os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        # ) as fp:
        #     np.save(
        #         fp,
        #         replay._store["terminal"][
        #             replay._task_replay_start_index[
        #                 task_idx
        #             ] : replay._task_replay_start_index[task_idx]
        #             + replay._task_add_count[task_idx].value
        #         ],
        #     )

        # print("Replay filled with demos.")



# add individual data points to a replay
def _add_keypoints_to_replay_temporal(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    device="cpu",
):
    prev_action = None
    obs = inital_obs
    initial_frame = sample_frame
    for k in range(
        next_keypoint_idx, len(episode_keypoints)   #  loop in all the following keypoint
    ):  # confused here, it seems that there are many same samples in the replay
        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]    # keypoint frame
        obs_tm1 = demo[max(0, keypoint - 1)]   # frame before keypoint
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(         #  get keypoint action 
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        terminal = k == len(episode_keypoints) - 1    # if is the last keypoint, terminal
        reward = float(terminal) * 1.0 if terminal else 0

        obs_dict = extract_obs(      #  obs is the i_th frame
            obs,
            CAMERAS,
            t=k - next_keypoint_idx,     # t for calculate time, represent t_th keypoint
            prev_action=prev_action,
            episode_length=25,
        )
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
            "initial_frame": initial_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,  # rot + grip: 3+1
            "gripper_pose": obs_tp1.gripper_pose,   # 3+4
            "lang_goal": np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        obs = obs_tp1
        sample_frame = keypoint

    # final step    # FIXME It is no need to do this step?
    obs_dict_tp1 = extract_obs(
        obs_tp1,
        CAMERAS,
        t=k + 1 - next_keypoint_idx,
        prev_action=prev_action,
        episode_length=25,
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop("wrist_world_to_cam", None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)


def fill_replay_temporal(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    rank,
    clip_model=None,
    device="cpu",
):

    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            if rank == 0:
                print(
                    "[Info] Replay dataset already exists in the disk: {}".format(
                        task_replay_storage_folder
                    ),
                    flush=True,
                )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)
    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...:", task)
        for d_idx in range(start_idx, start_idx + num_demos):
            print("Filling demo %d" % d_idx)
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)

            # extract keypoints
            episode_keypoints = keypoint_discovery(demo)  # list of keypoint   [id0, id1, id2]
            next_keypoint_idx = 0
            for i in range(len(demo) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                    continue

                obs = demo[i]
                desc = descs[0]
                # if our starting point is past one of the keypoints, then remove it
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):
                    break
                _add_keypoints_to_replay_temporal(
                    replay,
                    task,
                    task_replay_storage_folder,
                    d_idx,
                    i,
                    obs,
                    demo,
                    episode_keypoints,
                    cameras,
                    rlbench_scene_bounds,
                    voxel_sizes,
                    rotation_resolution,
                    crop_augmentation,
                    next_keypoint_idx=next_keypoint_idx,
                    description=desc,
                    clip_model=clip_model,
                    device=device,
                )

        # save TERMINAL info in replay_info.npy
        # task_idx = replay._task_index[task]
        # with open(
        #     os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        # ) as fp:
        #     np.save(
        #         fp,
        #         replay._store["terminal"][
        #             replay._task_replay_start_index[
        #                 task_idx
        #             ] : replay._task_replay_start_index[task_idx]
        #             + replay._task_add_count[task_idx].value
        #         ],
        #     )

        # print("Replay filled with demos.")

