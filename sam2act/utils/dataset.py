
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
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

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

def resample_to_fixed(x, rgb, feats, target_points=5500):
    """
    x: numpy array of shape (N, C) for one point cloud
       (N points, C features e.g. xyzrgb)
    Returns: numpy array of shape (target_points, C)
    """
    N, C = x.shape
    
    if N == target_points:
        return x
    elif N > target_points:  # downsample
        idx = np.random.choice(N, target_points, replace=False)
    else:  # upsample with replacement
        idx = np.random.choice(N, target_points, replace=True)
    
    if feats is not None:
        return x[idx], rgb[idx], feats[idx]

    return x[idx], rgb[idx]

# ORIGINAL
def _create_articubot_dataset(
    task, obs, episode_num, sample_frame, key_frame_obs, action, lang_feats, val
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


    all_pcd = np.concatenate([front_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    all_rgb = np.concatenate([front_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)

    x_range = (-2.06492364, 2.26651619)
    y_range = (-0.96348435, 1.00034714)
    z_range = (0.3, 1.72072086)

    # Table filtered out
    # x_range = (-0.5048, 2.26651619)
    # y_range = (-0.96348435, 1.00034714)
    # z_range = (0.7501, 1.72072086)

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

    if False:
        if np_points.shape[0] < 6500:
            print("Too few points: ", np_points.shape[0])
            np_points, np_rgb = resample_to_fixed(np_points, np_rgb, target_points=6500)
            

        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(np_points)
        obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)


        sampled_pcd = obj_pcd.farthest_point_down_sample(6500)
        sampled_points = np.asarray(sampled_pcd.points)
        sampled_rgb = np.asarray(sampled_pcd.colors)

        point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)



    # np_points = np.concatenate([np_points, reference_planes], axis=0)
    # np_rgb = np.concatenate([np_rgb, reference_colors], axis=0)

    # rand_indx = np.random.choice(all_pcd.shape[0], 30000)
    # np_points = all_pcd[rand_indx]
    # np_rgb = all_rgb[rand_indx]    

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)


    sampled_pcd = obj_pcd.farthest_point_down_sample(10000)
    sampled_points = np.asarray(sampled_pcd.points)
    sampled_rgb = np.asarray(sampled_pcd.colors)

    point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)
    # point_cloud = np.concatenate([np_points, np_rgb], axis=1)


    # point_cloud = np.concatenate([np_points, np_rgb], axis=1)
    # point_cloud = resample_to_fixed(point_cloud, target_points=5500)

    data = {'point_cloud': np.expand_dims(point_cloud, axis=0), 
            'action': action, 'gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(obs.gripper_pose[:3], obs.gripper_pose[3:7], obs.gripper_joint_positions[1]), axis=0),
            'goal_gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(key_frame_obs.gripper_pose[:3], key_frame_obs.gripper_pose[3:7], key_frame_obs.gripper_joint_positions[1]), axis=0),
            'state': obs.get_low_dim_data(),
            'lang_feats': lang_feats,}
    
    if val:
        directory = os.path.join('data_articubot', task + '_modified_keypoints_val', folder_name)
    else:
        directory = os.path.join('data_articubot', task + '_10k_modified_keypoints+overhead', folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    final_path = os.path.join(directory, str(sample_frame) + '.pkl')
    with open(final_path, 'wb') as f:
        print('Saving data to: ', final_path)
        pickle.dump(data, f)

#   Orientation discretized 
def _create_articubot_dataset_orientation_discretized(
    task, obs, episode_num, sample_frame, key_frame_obs, action, lang_feats, val
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

    # all_pcd = np.concatenate([front_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    # all_rgb = np.concatenate([front_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)

    x_range = (-2.06492364, 2.26651619)
    y_range = (-0.96348435, 1.00034714)
    z_range = (0.3, 1.72072086)

    # Table filtered out
    # x_range = (-0.5048, 2.26651619)
    # y_range = (-0.96348435, 1.00034714)
    # z_range = (0.7501, 1.72072086)

    mask = (
    (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
    (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
    (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    )

    np_points= all_pcd[mask]
    np_rgb = all_rgb[mask]

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)


    sampled_pcd = obj_pcd.farthest_point_down_sample(10000)
    sampled_points = np.asarray(sampled_pcd.points)
    sampled_rgb = np.asarray(sampled_pcd.colors)

    point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)

    # point_cloud = np.concatenate([np_points, np_rgb], axis=1)
    # point_cloud = resample_to_fixed(point_cloud, target_points=5500)

    # data = {'point_cloud': np.expand_dims(point_cloud, axis=0), 
    #         'action': action, 'gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(obs.gripper_pose[:3], obs.gripper_pose[3:7], obs.gripper_joint_positions[1]), axis=0),
    #         'goal_gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(key_frame_obs.gripper_pose[:3], key_frame_obs.gripper_pose[3:7], key_frame_obs.gripper_joint_positions[1]), axis=0),
    #         'state': obs.get_low_dim_data(),
    #         'lang_feats': lang_feats,}

    gripper_rot = R.from_matrix(obs.gripper_matrix[:3 ,:3]).as_euler("xyz", degrees=True)
    goal_gripper_rot = R.from_matrix(key_frame_obs.gripper_matrix[:3, :3]).as_euler("xyz", degrees=True)

    data = {
        'point_cloud': np.expand_dims(point_cloud, axis=0),
        'action': action,
        'gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(obs.gripper_pose[:3], obs.gripper_pose[3:7], obs.gripper_joint_positions[1]), axis=0),
        'gripper_pos': np.expand_dims(obs.gripper_pose[:3], axis=0),
        'gripper_rot': np.expand_dims(gripper_rot, axis=0),
        'goal_gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(key_frame_obs.gripper_pose[:3], key_frame_obs.gripper_pose[3:7], key_frame_obs.gripper_joint_positions[1]), axis=0),
        'goal_gripper_pos': np.expand_dims(key_frame_obs.gripper_pose[:3], axis=0),
        'goal_gripper_rot': np.expand_dims(goal_gripper_rot, axis=0),
        'lang_feats': lang_feats,
    }

    if val:
        directory = os.path.join('data_articubot', task + '_orientation_discretized_full_val', folder_name)
    else:
        directory = os.path.join('data_articubot', task + 'orientation_discretized', folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    final_path = os.path.join(directory, str(sample_frame) + '.pkl')
    with open(final_path, 'wb') as f:
        print('Saving data to: ', final_path)
        pickle.dump(data, f)

def _create_articubot_dataset_zoomed(
    task, obs, episode_num, sample_frame, key_frame_obs, action, lang_feats, val,
    zoom_size=0.8, target_points=10000
):
    folder_name = 'episode_' + str(episode_num)
    print(episode_num, sample_frame)

    # Get base point clouds and colors
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

    # Get goal gripper points
    goal_gripper_points = get_4_points_from_gripper_pos_orient(
        key_frame_obs.gripper_pose[:3], 
        key_frame_obs.gripper_pose[3:7], 
        key_frame_obs.gripper_joint_positions[1]
    )
    goal_center = np.mean(goal_gripper_points, axis=0)

    # Define zoom-in bounding box (cube around goal gripper)
    x_range = (goal_center[0] - zoom_size/2, goal_center[0] + zoom_size/2)
    y_range = (goal_center[1] - zoom_size/2, goal_center[1] + zoom_size/2)
    z_range = (goal_center[2] - zoom_size/2, goal_center[2] + zoom_size/2)

    mask = (
        (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
        (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
        (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    )

    np_points = all_pcd[mask]
    np_rgb = all_rgb[mask]

    # Handle too few / too many points
    if np_points.shape[0] < target_points:
        print(f"[Zoomed] Too few points: {np_points.shape[0]}")
        breakpoint()
        return  # Skip saving this sample if too few points

        print(f"[Zoomed] Too few points: {np_points.shape[0]}, resampling...")
        # np_points, np_rgb = resample_to_fixed(np_points, np_rgb, target_points=target_points)


    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

    sampled_pcd = obj_pcd.farthest_point_down_sample(target_points)
    sampled_points = np.asarray(sampled_pcd.points)
    sampled_rgb = np.asarray(sampled_pcd.colors)

    # wrist_obj = o3d.geometry.PointCloud()
    # wrist_obj.points = o3d.utility.Vector3dVector(wrist_pcd)
    # wrist_obj.colors = o3d.utility.Vector3dVector(wrist_rgb)
    # wrist_sampled_pcd = wrist_obj.farthest_point_down_sample(7000)

    # wrist_sampled_points = np.asarray(wrist_sampled_pcd.points)
    # wrist_sampled_rgb = np.asarray(wrist_sampled_pcd.colors)

    # sampled_points = np.concatenate([sampled_points, wrist_sampled_points], axis=0)
    # sampled_rgb = np.concatenate([sampled_rgb, wrist_sampled_rgb], axis=0)

    point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)
    # point_cloud = np.concatenate([obj_pcd.points, obj_pcd.colors], axis=1)


    # Pack data
    data = {
        'point_cloud': np.expand_dims(point_cloud, axis=0), 
        'action': action, 
        'gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(
            obs.gripper_pose[:3], obs.gripper_pose[3:7], obs.gripper_joint_positions[1]), axis=0),
        'goal_gripper_pcd': np.expand_dims(goal_gripper_points, axis=0),
        'state': obs.get_low_dim_data(),
        'lang_feats': lang_feats,
    }
    
    # Save path
    if val:
        directory = os.path.join('data_articubot', task + '_zoomed_in_goal_val', folder_name)
    else:
        directory = os.path.join('data_articubot', task + '_zoomed_in_goal', folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    final_path = os.path.join(directory, str(sample_frame) + '.pkl')
    with open(final_path, 'wb') as f:
        print('Saving ZOOMED data to: ', final_path)
        pickle.dump(data, f)

# MASKED
def _create_articubot_dataset_object_only(
    task, obs, episode_num, sample_frame, key_frame_obs, action, lang_feats, val
):
    folder_name = 'episode_' + str(episode_num)
    print(episode_num, sample_frame)

    # --- Helper to filter pcd & rgb with mask ---
    def filter_pcd_with_mask(pcd, rgb, mask):
        pcd = pcd.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3) / 255.0
        mask = mask.reshape(-1, 3)

        # keep = (mask!=10) & (mask!=31) & (mask!=34) & (mask!=35) & (mask!=39) & (mask!=40) & (mask!=41) & (mask!=42) & (mask!=43) & (mask!=44) & (mask!=45) & (mask!=46) & (mask!=48) & (mask!=52) & (mask!=55)
        exclude_vals = [10, 31, 34, 35, 39, 40, 41, 42,
                        43, 44, 45, 46, 48, 52, 55]

        keep = (~np.isin(mask[..., 0], exclude_vals)) | (mask[..., 1] > 0)
        return pcd[keep], rgb[keep]
    
    # Filter each view with its corresponding mask
    front_pcd, front_rgb = filter_pcd_with_mask(obs.front_point_cloud, obs.front_rgb, obs.front_mask)
    wrist_pcd, wrist_rgb = filter_pcd_with_mask(obs.wrist_point_cloud, obs.wrist_rgb, obs.wrist_mask)
    left_shoulder_pcd, left_shoulder_rgb = filter_pcd_with_mask(obs.left_shoulder_point_cloud, obs.left_shoulder_rgb, obs.left_shoulder_mask)
    right_shoulder_pcd, right_shoulder_rgb = filter_pcd_with_mask(obs.right_shoulder_point_cloud, obs.right_shoulder_rgb, obs.right_shoulder_mask)

    # Concatenate selected pointclouds
    np_points= np.concatenate([front_pcd, wrist_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    np_rgb = np.concatenate([front_rgb, wrist_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)

    # # Bounding box filter
    # x_range = (-2.06492364, 2.26651619)
    # y_range = (-0.96348435, 1.00034714)
    # z_range = (0.3, 1.72072086)

    # mask = (
    #     (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
    #     (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
    #     (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    # )

    # np_points = all_pcd[mask]
    # np_rgb = all_rgb[mask]

    if np_points.shape[0] < 2000:
        print("Too few points: ", np_points.shape[0])
        np_points, np_rgb = resample_to_fixed(np_points, np_rgb, target_points=2000)
        point_cloud = np.concatenate([np_points, np_rgb], axis=1)
    
    else:
        # Downsample scene point cloud
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(np_points)
        obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

        sampled_pcd = obj_pcd.farthest_point_down_sample(2000)

        # Combine scene + wrist
        sampled_points = sampled_pcd.points
        sampled_rgb = sampled_pcd.colors
        point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)

    # Package dataset
    data = {
        'point_cloud': np.expand_dims(point_cloud, axis=0),
        'action': action,
        'gripper_pcd': np.expand_dims(
            get_4_points_from_gripper_pos_orient(
                obs.gripper_pose[:3], obs.gripper_pose[3:7], obs.gripper_joint_positions[1]
            ), axis=0),
        'goal_gripper_pcd': np.expand_dims(
            get_4_points_from_gripper_pos_orient(
                key_frame_obs.gripper_pose[:3], key_frame_obs.gripper_pose[3:7], key_frame_obs.gripper_joint_positions[1]
            ), axis=0),
        'state': obs.get_low_dim_data(),
        'lang_feats': lang_feats,
    }

    # Save
    if val:
        directory = os.path.join('data_articubot', task + '_masked_val', folder_name)
    else:
        directory = os.path.join('data_articubot', task + '_masked', folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    final_path = os.path.join(directory, str(sample_frame) + '.pkl')
    with open(final_path, 'wb') as f:
        print('Saving data to: ', final_path)
        pickle.dump(data, f)

def _create_articubot_dataset_object_sampled_more(
    task, obs, episode_num, sample_frame, key_frame_obs, action, lang_feats, val
):
    folder_name = 'episode_' + str(episode_num)
    print(episode_num, sample_frame)

    # --- Helper to filter pcd & rgb with mask ---
    def filter_pcd_with_mask(pcd, rgb, mask):
        pcd = pcd.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3) / 255.0
        mask = mask.reshape(-1, 3)

        exclude_vals = [10, 31, 34, 35, 39, 40, 41, 42,
                        43, 44, 45, 46, 48, 52, 55]

        keep = (~np.isin(mask[..., 0], exclude_vals)) | (mask[..., 1] > 0)
        return pcd[keep], rgb[keep], pcd, rgb
    
    # Filter each view with its corresponding mask
    front_pcd, front_rgb, front_scene_pcd, front_scene_rgb = filter_pcd_with_mask(obs.front_point_cloud, obs.front_rgb, obs.front_mask)
    wrist_pcd, wrist_rgb, wrist_scene_pcd, wrist_scene_rgb, = filter_pcd_with_mask(obs.wrist_point_cloud, obs.wrist_rgb, obs.wrist_mask)
    left_shoulder_pcd, left_shoulder_rgb, left_shoulder_scene_pcd, left_shoulder_scene_rgb = filter_pcd_with_mask(obs.left_shoulder_point_cloud, obs.left_shoulder_rgb, obs.left_shoulder_mask)
    right_shoulder_pcd, right_shoulder_rgb, right_shoulder_scene_pcd, right_shoulder_scene_rgb = filter_pcd_with_mask(obs.right_shoulder_point_cloud, obs.right_shoulder_rgb, obs.right_shoulder_mask)

    # Concatenate object-masked pointclouds
    obj_points = np.concatenate([front_pcd, wrist_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    obj_rgb = np.concatenate([front_rgb, wrist_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)

    # Concatenate full scene pointclouds (no filtering)
    scene_points = np.concatenate([
        front_scene_pcd,
        wrist_scene_pcd,
        left_shoulder_scene_pcd,
        right_shoulder_scene_pcd
    ], axis=0)

    scene_rgb = np.concatenate([
        front_scene_rgb,
        wrist_scene_rgb,
        left_shoulder_scene_rgb,
        right_shoulder_scene_rgb
    ], axis=0)

    x_range = (-2.06492364, 2.26651619)
    y_range = (-0.96348435, 1.00034714)
    z_range = (0.3, 1.72072086)

    mask = (
    (scene_points[:, 0] >= x_range[0]) & (scene_points[:, 0] <= x_range[1]) &
    (scene_points[:, 1] >= y_range[0]) & (scene_points[:, 1] <= y_range[1]) &
    (scene_points[:, 2] >= z_range[0]) & (scene_points[:, 2] <= z_range[1])
    )

    scene_points = scene_points[mask]
    scene_rgb = scene_rgb[mask]

    # --- Step 1: Sample 2000 object points ---
    if obj_points.shape[0] > 2000:
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        obj_pcd.colors = o3d.utility.Vector3dVector(obj_rgb)
        obj_pcd = obj_pcd.farthest_point_down_sample(3000)
        obj_points = np.asarray(obj_pcd.points)
        obj_rgb = np.asarray(obj_pcd.colors)

    # --- Step 2: Sample 8000 scene points ---
    if scene_points.shape[0] < 8000:
        print("Too few scene points: ", scene_points.shape[0])
        scene_points, scene_rgb = resample_to_fixed(scene_points, scene_rgb, None, target_points=8000)
    else:
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
        scene_pcd.colors = o3d.utility.Vector3dVector(scene_rgb)
        scene_pcd = scene_pcd.farthest_point_down_sample(10000 - obj_points.shape[0])
        scene_points = np.asarray(scene_pcd.points)
        scene_rgb = np.asarray(scene_pcd.colors)

    # --- Step 3: Concatenate (10k total) ---
    all_points = np.concatenate([obj_points, scene_points], axis=0)
    all_rgb = np.concatenate([obj_rgb, scene_rgb], axis=0)
    point_cloud = np.concatenate([all_points, all_rgb], axis=1)

    # Package dataset
    data = {
        'point_cloud': np.expand_dims(point_cloud, axis=0),
        'action': action,
        'gripper_pcd': np.expand_dims(
            get_4_points_from_gripper_pos_orient(
                obs.gripper_pose[:3], obs.gripper_pose[3:7], obs.gripper_joint_positions[1]
            ), axis=0),
        'goal_gripper_pcd': np.expand_dims(
            get_4_points_from_gripper_pos_orient(
                key_frame_obs.gripper_pose[:3], key_frame_obs.gripper_pose[3:7], key_frame_obs.gripper_joint_positions[1]
            ), axis=0),
        'state': obs.get_low_dim_data(),
        'lang_feats': lang_feats,
    }

    # Save
    if val:
        directory = os.path.join('data_articubot', task + '_10k_object_biased_val', folder_name)
    else:
        directory = os.path.join('data_articubot', task + '_10k_object_biased', folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    final_path = os.path.join(directory, str(sample_frame) + '.pkl')
    with open(final_path, 'wb') as f:
        print('Saving data to: ', final_path)
        pickle.dump(data, f)

    

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

    front_rgb = torch.from_numpy(obs.front_rgb.transpose([2,0,1]) / 255.0).unsqueeze(0).float().to(agent._device)
    wrist_rgb = torch.from_numpy(obs.wrist_rgb.transpose([2,0,1])/ 255.0).unsqueeze(0).float().to(agent._device)
    left_shoulder_rgb = torch.from_numpy(obs.left_shoulder_rgb.transpose([2,0,1]) / 255.0).unsqueeze(0).float().to(agent._device)
    right_shoulder_rgb = torch.from_numpy(obs.right_shoulder_rgb.transpose([2,0,1]) / 255.0).unsqueeze(0).float().to(agent._device)

    front_rgb = F.interpolate(front_rgb, size=(256, 256), mode='bilinear', align_corners=True)
    wrist_rgb = F.interpolate(wrist_rgb, size=(256, 256), mode='bilinear', align_corners=True)
    left_shoulder_rgb = F.interpolate(left_shoulder_rgb, size=(256, 256), mode='bilinear', align_corners=True)
    right_shoulder_rgb = F.interpolate(right_shoulder_rgb, size=(256, 256), mode='bilinear', align_corners=True)

    # all_pcd = np.concatenate([front_pcd, wrist_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    # all_rgb = np.concatenate([front_rgb, wrist_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)

    # Getting SAM2 features
    # proprio = arm_utils.stack_on_channel(obs_dict['low_dim_state'])
    # sam2feats = {}

    # new_obs, pcd = peract_utils._preprocess_inputs(obs_dict, [camera])
    # pc, img_feat = rvt_utils.get_pc_img_feat(
    #     new_obs,
    #     pcd,
    # )

    # pc, img_feat = rvt_utils.move_pc_in_bound(
    #     pc, img_feat, agent.scene_bounds, no_op=not agent.move_pc_in_bound
    # )

    # img = agent._network.render(
    #             pc=pc,
    #             img_feat=img_feat,
    #             img_aug=0,
    #             mvt1_or_mvt2=True,
    #             dyn_cam_info=None,
    #         )

    front_sam2_feats, _ = agent._network.mvt1.sam2_image_encoder_forward(agent._network.sam2, front_rgb)
    wrist_sam2_feats, _ = agent._network.mvt1.sam2_image_encoder_forward(agent._network.sam2, wrist_rgb)
    left_shoulder_sam2_feats, _ = agent._network.mvt1.sam2_image_encoder_forward(agent._network.sam2, left_shoulder_rgb)
    right_shoulder_sam2_feats, _ = agent._network.mvt1.sam2_image_encoder_forward(agent._network.sam2, right_shoulder_rgb)
    
    B = 1  # looks like your batch dim is 1
    tokens, _, C = front_sam2_feats[0].shape   # (4096, 1, 32)

    H = W = int(tokens ** 0.5)  # 64
    front_sam2_feats = front_sam2_feats[0].permute(1, 2, 0).reshape(B, C, H, W)
    wrist_sam2_feats = wrist_sam2_feats[0].permute(1, 2, 0).reshape(B, C, H, W)
    left_shoulder_sam2_feats = left_shoulder_sam2_feats[0].permute(1, 2, 0).reshape(B, C, H, W)
    right_shoulder_sam2_feats = right_shoulder_sam2_feats[0].permute(1, 2, 0).reshape(B, C, H, W)

    def filter_pcd_with_mask(pcd, rgb, feats, mask):
        pcd = pcd.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3) / 255.0
        feats = feats.reshape(-1, feats.shape[-1])
        mask = mask.reshape(-1, 3)

        # keep = (mask!=10) & (mask!=31) & (mask!=34) & (mask!=35) & (mask!=39) & (mask!=40) & (mask!=41) & (mask!=42) & (mask!=43) & (mask!=44) & (mask!=45) & (mask!=46) & (mask!=48) & (mask!=52) & (mask!=55)
        exclude_vals = [10, 31, 34, 35, 39, 40, 41, 42,
                        43, 44, 45, 46, 48, 52, 55]

        keep = (~np.isin(mask[..., 0], exclude_vals)) | (mask[..., 1] > 0)
        return pcd[keep], rgb[keep], feats[keep].detach().cpu().numpy()

    def filter_pcd(pcd, rgb, feats, mask):
        pcd = pcd.reshape(-1, 3)
        rgb = rgb.reshape(-1, 3) / 255.0
        feats = feats.reshape(-1, feats.shape[-1])
        mask = mask.reshape(-1, 3)

        # keep = (mask!=10) & (mask!=31) & (mask!=34) & (mask!=35) & (mask!=39) & (mask!=40) & (mask!=41) & (mask!=42) & (mask!=43) & (mask!=44) & (mask!=45) & (mask!=46) & (mask!=48) & (mask!=52) & (mask!=55)
        return pcd, rgb, feats.detach().cpu().numpy()
    

    # out = agent._network.mvt1(
    #         img=img,
    #         proprio=proprio,
    #         lang_emb=None,
    #         wpt_local=None,
    #         rot_x_y=None,
    #         articubot=True,
    #         # hm_gt=hm_gt,
    # ) # 3, 32, 64, 64
    upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    front_sam2_feats = upsample(front_sam2_feats) # 1, 32, 128, 128
    left_shoulder_sam2_feats = upsample(left_shoulder_sam2_feats) # 1, 32, 128, 128
    right_shoulder_sam2_feats = upsample(right_shoulder_sam2_feats) # 1, 32, 128, 128
    wrist_sam2_feats = upsample(wrist_sam2_feats) # 1, 32, 128, 128

    front_sam2_feats = front_sam2_feats.squeeze().permute(1, 2, 0) # 128, 128, 32
    left_shoulder_sam2_feats = left_shoulder_sam2_feats.squeeze().permute(1, 2, 0) # 128, 128, 32
    right_shoulder_sam2_feats = right_shoulder_sam2_feats.squeeze().permute(1, 2, 0) # 128, 128, 32
    wrist_sam2_feats = wrist_sam2_feats.squeeze().permute(1, 2, 0) # 128, 128, 32

    front_pcd, front_rgb, front_sam2_feats = filter_pcd(obs.front_point_cloud, obs.front_rgb, front_sam2_feats, obs.front_mask)
    wrist_pcd, wrist_rgb, wrist_sam2_feats = filter_pcd(obs.wrist_point_cloud, obs.wrist_rgb, wrist_sam2_feats, obs.wrist_mask)
    left_shoulder_pcd, left_shoulder_rgb, left_shoulder_sam2_feats = filter_pcd(obs.left_shoulder_point_cloud, obs.left_shoulder_rgb, left_shoulder_sam2_feats, obs.left_shoulder_mask)
    right_shoulder_pcd, right_shoulder_rgb, right_shoulder_sam2_feats = filter_pcd(obs.right_shoulder_point_cloud, obs.right_shoulder_rgb, right_shoulder_sam2_feats, obs.right_shoulder_mask)


    all_pcd = np.concatenate([front_pcd, wrist_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    all_rgb = np.concatenate([front_rgb, wrist_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)
    all_sam2_feats = np.concatenate([front_sam2_feats.reshape(-1, front_sam2_feats.shape[-1]),
                                        wrist_sam2_feats.reshape(-1, wrist_sam2_feats.shape[-1]),
                                        left_shoulder_sam2_feats.reshape(-1, left_shoulder_sam2_feats.shape[-1]),
                                        right_shoulder_sam2_feats.reshape(-1, right_shoulder_sam2_feats.shape[-1])], axis=0)
    # x_range = (-0.5048, 2.26651619)
    # y_range = (-0.96348435, 1.00034714)
    # z_range = (0.7501, 1.72072086)

    # mask = (
    # (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
    # (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
    # (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    # )
    # reference_planes, reference_colors = create_reference_planes_with_colors(x_range, y_range, z_range, num_points_per_axis=50)

    # np_points= all_pcd[mask]
    # np_rgb = all_rgb[mask]

    # np_points = np.concatenate([np_points, reference_planes], axis=0)
    # np_rgb = np.concatenate([np_rgb, reference_colors], axis=0)

    # rand_indx = np.random.choice(all_pcd.shape[0], 30000)
    # np_points = all_pcd[rand_indx]
    # np_rgb = all_rgb[rand_indx]    

    # obj_pcd = o3d.geometry.PointCloud()
    # obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    # obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

    # sampled_pcd = obj_pcd.voxel_down_sample(0.02)
    # sampled_pcd = obj_pcd.furthest_down_sample(10000)
    # sampled_points = np.asarray(sampled_pcd.points)
    # sampled_rgb = np.asarray(sampled_pcd.colors)
    # point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)


    if all_pcd.shape[0] < 2000:
        print("Too few points: ", all_pcd.shape[0])
        all_pcd, all_rgb, all_sam2_feats = resample_to_fixed(all_pcd, all_rgb, all_sam2_feats, target_points=2000)
    
    else:
        fps = furthest_point_sampling(torch.from_numpy(all_pcd), 10000)
        all_pcd = all_pcd[fps]
        all_rgb = all_rgb[fps]
        all_sam2_feats = all_sam2_feats[fps]

    # obj_pcd = o3d.geometry.PointCloud()
    # obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    # obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

    # sampled_pcd = obj_pcd.farthest_point_down_sample(4500)

    # sampled_points = np.asarray(sampled_pcd.points)
    # sampled_rgb = np.asarray(sampled_pcd.colors)
    # point_cloud = np.concatenate([np_points.detach().cpu().numpy(), np_rgb.detach().cpu().numpy()], axis=1)

    data = {'point_cloud': np.expand_dims(all_pcd, axis=0),
            'rgb': np.expand_dims(all_rgb, axis=0),
            'features': np.expand_dims(all_sam2_feats, axis=0),
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
        depths: [H, W] tensor of depth maps (float32, same device)
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
    K_inv = torch.inverse(K).float()
    T = extrinsics.float() # [4, 4]

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
def _get_articubot_dataset(obs, add_rgb_zeros=False, add_rgb_ones=False, add_one_hot=False, one_hot_dim=3, collision=True):
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

    
    # save mask
    if collision:
        x_range = (-2.06492364, 2.26651619)
        y_range = (-0.96348435, 1.00034714)
        z_range = (0.3, 1.72072086)
    else:
    # Table filtered out
        x_range = (-0.5048, 2.26651619)
        y_range = (-0.96348435, 1.00034714)
        z_range = (0.7501, 1.72072086)

    mask = (
    (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
    (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
    (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    )

    np_points= all_pcd[mask]
    np_rgb = all_rgb[mask]

    if collision:
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(np_points)
        obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

        sampled_pcd = obj_pcd.farthest_point_down_sample(10000)
        # sampled_pcd = obj_pcd.voxel_down_sample(0.02)
        sampled_points = np.asarray(sampled_pcd.points)
        sampled_rgb = np.asarray(sampled_pcd.colors)
    else:
        if np_points.shape[0] < 6500:
            print("Too few points: ", np_points.shape[0])
            np_points, np_rgb = resample_to_fixed(np_points, np_rgb, target_points=6500)
            

        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(np_points)
        obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)


        sampled_pcd = obj_pcd.farthest_point_down_sample(6500)
        sampled_points = np.asarray(sampled_pcd.points)
        sampled_rgb = np.asarray(sampled_pcd.colors)

    # rand_indx = np.random.choice(all_pcd.shape[0], 30000)
    # np_points = all_pcd[rand_indx]
    # np_rgb = all_rgb[rand_indx]    



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


# For rolling out
def _get_articubot_dataset_orientation_discretized(obs, add_rgb_zeros=False, add_rgb_ones=False, add_one_hot=False, one_hot_dim=3, collision=True):
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
    
    # save mask
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

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

    sampled_pcd = obj_pcd.farthest_point_down_sample(10000)
    # sampled_pcd = obj_pcd.voxel_down_sample(0.02)
    sampled_points = np.asarray(sampled_pcd.points)
    sampled_rgb = np.asarray(sampled_pcd.colors)



    if add_rgb_zeros or add_rgb_ones:
        point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)
    else: 
        point_cloud = sampled_points

    gripper_pose = obs['gripper_pose'][0][0][:3].detach().cpu().numpy()
    gripper_pcd = np.expand_dims(gripper_pose, axis=(0,1))
    
    # gripper_pcd = np.expand_dims(get_4_points_from_gripper_pos_orient(gripper_pose[:3], gripper_pose[3:7], joint_pos[1]), axis=0)
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
                'gripper_pos': gripper_pcd,}
    
    return obs_dict

# Scene masked out
def _get_articubot_dataset_masked(
    obs, add_rgb_zeros=False, add_rgb_ones=False, add_one_hot=False, one_hot_dim=3, collision=False, num_points=10000
):
    # --- Helper to filter a point cloud by mask values ---
    def filter_pcd_with_mask(pcd, rgb, mask):
        pcd = pcd.detach().cpu().numpy()[0, 0].transpose([1, 2, 0]).reshape(-1, 3)
        rgb = rgb.detach().cpu().numpy()[0, 0].transpose([1, 2, 0]).reshape(-1, 3) / 255.0
        mask = mask.detach().cpu().numpy()[0, 0].transpose([1, 2, 0]).reshape(-1, 3)

        # Exclude these mask values
        exclude_vals = [10, 31, 34, 35, 39, 40, 41, 42,
                        43, 44, 45, 46, 48, 52, 55]
        
        keep = (~np.isin(mask[..., 0], exclude_vals)) | (mask[..., 1] > 0)

        return pcd[keep], rgb[keep]

    # Apply filtering for each camera
    front_pcd, front_rgb = filter_pcd_with_mask(obs['front_point_cloud'], obs['front_rgb'], obs['front_mask'])
    wrist_pcd, wrist_rgb = filter_pcd_with_mask(obs['wrist_point_cloud'], obs['wrist_rgb'], obs['wrist_mask'])
    left_shoulder_pcd, left_shoulder_rgb = filter_pcd_with_mask(obs['left_shoulder_point_cloud'], obs['left_shoulder_rgb'], obs['left_shoulder_mask'])
    right_shoulder_pcd, right_shoulder_rgb = filter_pcd_with_mask(obs['right_shoulder_point_cloud'], obs['right_shoulder_rgb'], obs['right_shoulder_mask'])

    # Concatenate
    all_pcd = np.concatenate([front_pcd, wrist_pcd, left_shoulder_pcd, right_shoulder_pcd], axis=0)
    all_rgb = np.concatenate([front_rgb, wrist_rgb, left_shoulder_rgb, right_shoulder_rgb], axis=0)

    # Bounding box
    # if collision:
    #     x_range = (-2.06492364, 2.26651619)
    #     y_range = (-0.96348435, 1.00034714)
    #     z_range = (0.3, 1.72072086)
    # else:
    #     x_range = (-0.5048, 2.26651619)
    #     y_range = (-0.96348435, 1.00034714)
    #     z_range = (0.7501, 1.72072086)

    # mask = (
    #     (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
    #     (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
    #     (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    # )
    # np_points = all_pcd[mask]
    # np_rgb = all_rgb[mask]

    if all_pcd.shape[0] < num_points:
        print("Too few points: ", all_pcd.shape[0])
        all_pcd, all_rgb = resample_to_fixed(all_pcd, all_rgb, None, target_points=num_points)

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(all_pcd)
    obj_pcd.colors = o3d.utility.Vector3dVector(all_rgb)
    sampled_pcd = obj_pcd.farthest_point_down_sample(num_points)
    sampled_points = np.asarray(sampled_pcd.points)
    sampled_rgb = np.asarray(sampled_pcd.colors)

    # Point cloud assembly
    if add_rgb_zeros or add_rgb_ones:
        point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)
    else:
        point_cloud = sampled_points

    # Gripper points
    gripper_pose = obs['gripper_pose'][0][0].detach().cpu().numpy()
    joint_pos = obs['gripper_joint_positions'][0][0].detach().cpu().numpy()
    gripper_pcd = np.expand_dims(
        get_4_points_from_gripper_pos_orient(gripper_pose[:3], gripper_pose[3:7], joint_pos[1]), axis=0
    )
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

    obs_dict = {
        'point_cloud': point_cloud,
        'gripper_pcd': gripper_pcd,
    }
    return obs_dict

# Biased sampling of the objects
def _get_articubot_dataset_10k_object_sampled_more(
    obs, add_rgb_zeros=False, add_rgb_ones=False,
    add_one_hot=False, one_hot_dim=3,
):
    def filter_pcd_with_mask(pcd, rgb, mask):
        pcd = pcd.detach().cpu().numpy()[0, 0].transpose([1, 2, 0]).reshape(-1, 3)
        rgb = rgb.detach().cpu().numpy()[0, 0].transpose([1, 2, 0]).reshape(-1, 3) / 255.0
        mask = mask.detach().cpu().numpy()[0, 0].transpose([1, 2, 0]).reshape(-1, 3)

        exclude_vals = [10, 31, 34, 35, 39, 40, 41, 42,
                        43, 44, 45, 46, 48, 52, 55]
        keep_obj = (~np.isin(mask[..., 0], exclude_vals)) | (mask[..., 1] > 0)

        return pcd[keep_obj], rgb[keep_obj], pcd, rgb

    # Filter each camera into object and scene
    obj_pcds, obj_rgbs, scene_pcds, scene_rgbs = [], [], [], []
    for cam in ["front", "wrist", "left_shoulder", "right_shoulder"]:
        pcd, rgb, scene_pcd, scene_rgb = filter_pcd_with_mask(
            obs[f"{cam}_point_cloud"], obs[f"{cam}_rgb"], obs[f"{cam}_mask"]
        )
        obj_pcds.append(pcd); obj_rgbs.append(rgb)
        scene_pcds.append(scene_pcd); scene_rgbs.append(scene_rgb)

    obj_pcd = np.concatenate(obj_pcds, axis=0)
    obj_rgb = np.concatenate(obj_rgbs, axis=0)
    scene_pcd = np.concatenate(scene_pcds, axis=0)
    scene_rgb = np.concatenate(scene_rgbs, axis=0)

    x_range = (-2.06492364, 2.26651619)
    y_range = (-0.96348435, 1.00034714)
    z_range = (0.3, 1.72072086)

    mask = (
    (scene_pcd[:, 0] >= x_range[0]) & (scene_pcd[:, 0] <= x_range[1]) &
    (scene_pcd[:, 1] >= y_range[0]) & (scene_pcd[:, 1] <= y_range[1]) &
    (scene_pcd[:, 2] >= z_range[0]) & (scene_pcd[:, 2] <= z_range[1])
    )

    scene_pcd = scene_pcd[mask]
    scene_rgb = scene_rgb[mask]

    # --- Step 1: Sample 2000 object points ---
    if obj_pcd.shape[0] > 2000:
        obj_geom = o3d.geometry.PointCloud()
        obj_geom.points = o3d.utility.Vector3dVector(obj_pcd)
        obj_geom.colors = o3d.utility.Vector3dVector(obj_rgb)
        obj_geom = obj_geom.farthest_point_down_sample(2000)
        obj_pcd = np.asarray(obj_geom.points)
        obj_rgb = np.asarray(obj_geom.colors)

    # --- Step 2: Sample 8000 scene points ---
    if scene_pcd.shape[0] < 8000:
        print("Too few scene points: ", scene_pcd.shape[0])
        scene_pcd, scene_rgb = resample_to_fixed(scene_pcd, scene_rgb, None, target_points=8000)
    else:
        scene_geom = o3d.geometry.PointCloud()
        scene_geom.points = o3d.utility.Vector3dVector(scene_pcd)
        scene_geom.colors = o3d.utility.Vector3dVector(scene_rgb)
        scene_geom = scene_geom.farthest_point_down_sample(10000 - obj_pcd.shape[0])
        scene_pcd = np.asarray(scene_geom.points)
        scene_rgb = np.asarray(scene_geom.colors)

    # Concatenate final cloud
    all_pcd = np.concatenate([obj_pcd, scene_pcd], axis=0)
    all_rgb = np.concatenate([obj_rgb, scene_rgb], axis=0)

    # Assemble point cloud
    if add_rgb_zeros or add_rgb_ones:
        point_cloud = np.concatenate([all_pcd, all_rgb], axis=1)
    else:
        point_cloud = all_pcd

    point_cloud = torch.from_numpy(np.expand_dims(point_cloud, axis=0))

    # Gripper points
    gripper_pose = obs['gripper_pose'][0][0].detach().cpu().numpy()
    joint_pos = obs['gripper_joint_positions'][0][0].detach().cpu().numpy()
    gripper_pcd = np.expand_dims(
        get_4_points_from_gripper_pos_orient(gripper_pose[:3], gripper_pose[3:7], joint_pos[1]), axis=0
    )
    gripper_pcd = torch.from_numpy(gripper_pcd)

    if add_rgb_zeros:
        gripper_pcd = torch.cat([gripper_pcd, torch.zeros(gripper_pcd.shape)], dim=2)
    elif add_rgb_ones:
        gripper_pcd = torch.cat([gripper_pcd, torch.ones(gripper_pcd.shape)], dim=2)

    if add_one_hot:
        pointcloud_one_hot = torch.zeros(point_cloud.shape[0], point_cloud.shape[1], one_hot_dim)
        pointcloud_one_hot[:, :, 0] = 1
        point_cloud = torch.cat([point_cloud, pointcloud_one_hot], dim=2)

        gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], one_hot_dim)
        gripper_pcd_one_hot[:, :, 1] = 1
        gripper_pcd = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)

    point_cloud = point_cloud.unsqueeze(0)
    gripper_pcd = gripper_pcd.unsqueeze(0)

    return {"point_cloud": point_cloud, "gripper_pcd": gripper_pcd}

def _get_featurized_dataset(point_cloud, obs):
    device = point_cloud.device
    gripper_pose = obs['gripper_pose'][0][0].detach().cpu().numpy()
    joint_pos = obs['gripper_joint_positions'][0][0].detach().cpu().numpy()

    gripper_pcd = np.expand_dims(get_4_points_from_gripper_pos_orient(gripper_pose[:3], gripper_pose[3:7], joint_pos[1]), axis=0)
    gripper_pcd = torch.from_numpy(gripper_pcd).to(device)
    gripper_pcd = torch.cat([gripper_pcd, torch.zeros((gripper_pcd.shape[0], gripper_pcd.shape[1], 32)).to(device)], dim=2)

    pointcloud_one_hot = torch.zeros(point_cloud.shape[0], point_cloud.shape[1], 2).to(device)
    pointcloud_one_hot[:, :, 0] = 1
    point_cloud = torch.cat([point_cloud, pointcloud_one_hot], dim=2)
    gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2).to(device)
    gripper_pcd_one_hot[:, :, 1] = 1
    gripper_pcd = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
    
    point_cloud = point_cloud.unsqueeze(0)
    gripper_pcd = gripper_pcd.unsqueeze(0)
    
    obs_dict = {'point_cloud': point_cloud,
                'gripper_pcd': gripper_pcd,}
    
    return obs_dict


def _get_articubot_dataset_zoomed(
    obs, predicted_goal, zoom_size=0.6, target_points=10000, add_rgb_ones=True, add_rgb_zeros=False, add_one_hot=True, one_hot_dim=3
):

    def filter_pcd_with_mask(pcd, rgb, mask):
        pcd = pcd.detach().cpu().numpy()[0, 0].transpose([1, 2, 0]).reshape(-1, 3)
        rgb = rgb.detach().cpu().numpy()[0, 0].transpose([1, 2, 0]).reshape(-1, 3) / 255.0
        mask = mask.detach().cpu().numpy()[0, 0].transpose([1, 2, 0]).reshape(-1, 3)

        return pcd, rgb

    # Filter each camera into object and scene
    scene_pcds, scene_rgbs = [], []
    for cam in ["front", "left_shoulder", "right_shoulder"]:
        scene_pcd, scene_rgb = filter_pcd_with_mask(
            obs[f"{cam}_point_cloud"], obs[f"{cam}_rgb"], obs[f"{cam}_mask"]
        )
        scene_pcds.append(scene_pcd); scene_rgbs.append(scene_rgb)

    wrist_pcd = obs['wrist_point_cloud'].detach().cpu().numpy()[0,0].transpose([1,2,0]).reshape(-1, 3)
    wrist_rgb = obs['wrist_rgb'].detach().cpu().numpy()[0,0].transpose([1,2,0]).reshape(-1, 3) / 255.0


    # Get goal gripper points
    goal_gripper_points = predicted_goal.squeeze().detach().cpu().numpy()  # (4, 3)
    goal_center = np.mean(goal_gripper_points, axis=0)

    # Define zoom-in bounding box (cube around goal gripper)
    x_range = (goal_center[0] - zoom_size/2, goal_center[0] + zoom_size/2)
    y_range = (goal_center[1] - zoom_size/2, goal_center[1] + zoom_size/2)
    z_range = (goal_center[2] - zoom_size/2, goal_center[2] + zoom_size/2)

    all_pcd = np.concatenate(scene_pcds, axis=0)
    all_rgb = np.concatenate(scene_rgbs, axis=0)

    mask = (
        (all_pcd[:, 0] >= x_range[0]) & (all_pcd[:, 0] <= x_range[1]) &
        (all_pcd[:, 1] >= y_range[0]) & (all_pcd[:, 1] <= y_range[1]) &
        (all_pcd[:, 2] >= z_range[0]) & (all_pcd[:, 2] <= z_range[1])
    )

    all_pcd = all_pcd[mask]
    all_rgb = all_rgb[mask]

    # Handle too few / too many points
    # if np_points.shape[0] < target_points:

    #     print(f"[Zoomed] Too few points: {np_points.shape[0]}, resampling...")
    #     # np_points, np_rgb = resample_to_fixed(np_points, np_rgb, target_points=target_points)

    scene_obj = o3d.geometry.PointCloud()
    scene_obj.points = o3d.utility.Vector3dVector(all_pcd)
    scene_obj.colors = o3d.utility.Vector3dVector(all_rgb)
    scene_obj = scene_obj.farthest_point_down_sample(3000)

    wrist_obj = o3d.geometry.PointCloud()
    wrist_obj.points = o3d.utility.Vector3dVector(wrist_pcd)
    wrist_obj.colors = o3d.utility.Vector3dVector(wrist_rgb)
    wrist_obj = wrist_obj.farthest_point_down_sample(7000)

    wrist_sampled_points = np.asarray(wrist_obj.points)
    wrist_sampled_rgb = np.asarray(wrist_obj.colors)

    sampled_points = np.asarray(scene_obj.points)
    sampled_rgb = np.asarray(scene_obj.colors)

    sampled_points = np.concatenate([sampled_points, wrist_sampled_points], axis=0)
    sampled_rgb = np.concatenate([sampled_rgb, wrist_sampled_rgb], axis=0)

    point_cloud = torch.from_numpy(np.expand_dims(np.concatenate([sampled_points, sampled_rgb], axis=1), axis=0))

    gripper_pose = obs['gripper_pose'][0][0].detach().cpu().numpy()
    joint_pos = obs['gripper_joint_positions'][0][0].detach().cpu().numpy()
    gripper_pcd = np.expand_dims(
        get_4_points_from_gripper_pos_orient(gripper_pose[:3], gripper_pose[3:7], joint_pos[1]), axis=0
    )
    gripper_pcd = torch.from_numpy(gripper_pcd)


    if add_rgb_zeros:
        gripper_pcd = torch.cat([gripper_pcd, torch.zeros(gripper_pcd.shape)], dim=2)
    elif add_rgb_ones:
        gripper_pcd = torch.cat([gripper_pcd, torch.ones(gripper_pcd.shape)], dim=2)

    if add_one_hot:
        pointcloud_one_hot = torch.zeros(point_cloud.shape[0], point_cloud.shape[1], one_hot_dim)
        pointcloud_one_hot[:, :, 0] = 1
        point_cloud = torch.cat([point_cloud, pointcloud_one_hot], dim=2)

        gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], one_hot_dim)
        gripper_pcd_one_hot[:, :, 1] = 1
        gripper_pcd = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)

    point_cloud = point_cloud.unsqueeze(0)
    gripper_pcd = gripper_pcd.unsqueeze(0)

    return {"point_cloud": point_cloud, "gripper_pcd": gripper_pcd}

    
def visualize(points):
    point_geometry = o3d.geometry.PointCloud()
    # print(points.shape)
    # print(predictions.shape)
    point_geometry.points = o3d.utility.Vector3dVector(points[:, :, :, :3].reshape(-1, 3))
    # point_geometry.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (4500,1)))

    
    # gripper_geometry = o3d.geometry.PointCloud()
    # gripper_geometry.points = o3d.utility.Vector3dVector(points[1024:1162])
    # gripper_geometry.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (138, 1)))

    # four_point_geometry = o3d.geometry.PointCloud()
    # four_point_geometry.points = o3d.utility.Vector3dVector(predictions[0, :, :, :].reshape(-1, 3).detach().cpu().numpy())
    # four_point_geometry.paint_uniform_color(np.array([0, 1, 0]))
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
            timestep = 0
            # for i in episode_keypoints:
            for i in range(len(demo)):
                # if not demo_augmentation and i > 0:
                #     break
                # if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                #     continue
                obs = demo[i]

                if i == episode_keypoints[next_keypoint_idx] and next_keypoint_idx < len(episode_keypoints)-1:
                    next_keypoint_idx = next_keypoint_idx + 1

                keypoint = episode_keypoints[next_keypoint_idx]

                print(keypoint)

                key_frame_obs = demo[keypoint]

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
                    # _create_articubot_dataset(task, obs, d_idx, i, key_frame_obs, action, lang_feats, val=val)
                    _create_articubot_dataset_orientation_discretized(task, obs, d_idx, i, key_frame_obs, action, lang_feats, val=val)

                    # _create_articubot_dataset_zoomed(task, obs, d_idx, i, key_frame_obs, action, lang_feats, val=val)
                    # _create_articubot_dataset_object_sampled_more(task, obs, d_idx, i, key_frame_obs, action, lang_feats, val=val)

                # timestep += 1
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

