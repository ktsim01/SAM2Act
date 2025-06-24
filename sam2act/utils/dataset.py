
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

from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer_temporal import UniformReplayBuffer_temporal
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from sam2act.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS
from sam2act.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from sam2act.libs.peract.helpers.utils import extract_obs
from third_party.robogen.robogen_utils import rotation_transfer_matrix_to_6D_batch, rotation_transfer_matrix_to_6D, \
                          get_4_points_from_gripper_pos_orient


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
    obs, episode_num, sample_frame, key_frame_obs, action
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

    rand_indx = np.random.choice(all_pcd.shape[0], 30000)
    np_points = all_pcd[rand_indx]
    np_rgb = all_rgb[rand_indx]    

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(np_rgb)

    sampled_pcd = obj_pcd.farthest_point_down_sample(4500)

    sampled_points = np.asarray(sampled_pcd.points)
    sampled_rgb = np.asarray(sampled_pcd.colors)
    point_cloud = np.concatenate([sampled_points, sampled_rgb], axis=1)

    data = {'point_cloud': np.expand_dims(point_cloud, axis=0), 
            'action': action, 'gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(obs.gripper_pose[:3], obs.gripper_pose[3:7], obs.gripper_joint_positions[1]), axis=0),
            'goal_gripper_pcd': np.expand_dims(get_4_points_from_gripper_pos_orient(key_frame_obs.gripper_pose[:3], key_frame_obs.gripper_pose[3:7], key_frame_obs.gripper_joint_positions[1]), axis=0),
            'state': obs.get_low_dim_data()}
    
    if not os.path.exists('put_item_in_drawer_articubot/' + folder_name):
        os.makedirs('put_item_in_drawer_articubot/' + folder_name)
    
    with open('put_item_in_drawer_articubot/' + folder_name + '/' + str(sample_frame) + '.pkl', 'wb') as f:
        print('Saving data to: ', folder_name + '/' + str(sample_frame) + '.pkl')
        pickle.dump(data, f)

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
                # if not demo_augmentation and i > 0:
                #     break
                # if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                #     continue
                print(episode_keypoints[next_keypoint_idx])
                obs = demo[i]
                key_frame_obs = demo[episode_keypoints[next_keypoint_idx]]
                if i == episode_keypoints[next_keypoint_idx] and next_keypoint_idx < len(episode_keypoints):
                    next_keypoint_idx = next_keypoint_idx + 1

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
                )
                            
                _create_articubot_dataset(obs, d_idx, i, key_frame_obs, action)
                desc = descs[0]
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

