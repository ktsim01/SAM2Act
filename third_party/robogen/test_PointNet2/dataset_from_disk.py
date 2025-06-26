import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import zarr
import os
from termcolor import cprint
import numpy as np
from tqdm import tqdm
import pickle
import random

class PointNetDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, all_obj_paths, beg_ratio=0, end_ratio=0.9, eval_episode=None, only_first_stage=False, is_pickle=False, use_all_data=False, 
                 conditioning_on_demo=False, n_obs_steps=1):
        self.all_obj_paths = all_obj_paths
        self.beg_ratio = beg_ratio
        self.end_ratio = end_ratio
        self.is_pickle = is_pickle
        self.use_all_data = use_all_data
        self.conditioning_on_demo = conditioning_on_demo
        self.n_obs_steps = n_obs_steps
        if only_first_stage:
            cprint('======= ONLY FIRST STAGE =======', 'red')

        if eval_episode is not None:
            cprint('======= EVAL MODE =======', 'red')
            cprint(f'Only evaluating the first observation of {eval_episode} episodes', 'red')
            
        # TODO for conditioning
        # for each trajectory of the object, record the grasping pose and opening pose. Store as a dictionary maybe, key is object_traj-id

        self.all_zarr_paths = []
        self.episode_idx_to_obj_id = {}
        self.obj_id_to_all_episodes_indices = {}
        episode_idx = 0
        cprint(f'MINO {all_obj_paths}', 'red')
        for obj_path in all_obj_paths:
            all_subfolder = os.listdir(obj_path)
            for s in ['action_dist', 'demo_rgbs', 'all_demo_path.txt', 'meta_info.json', 'example_pointcloud']:
                if s in all_subfolder:
                    all_subfolder.remove(s)
            all_subfolder = sorted(all_subfolder)
            beg = int(beg_ratio * len(all_subfolder))
            end = int(end_ratio * len(all_subfolder))
            if not self.use_all_data:
                end = min(end, 75)
            if eval_episode is not None:
                end = beg + eval_episode
            all_subfolder = all_subfolder[beg:end]
            self.all_zarr_paths += [os.path.join(obj_path, s) for s in all_subfolder]
            this_obj_episode_beg = episode_idx
            for s in all_subfolder:
                self.episode_idx_to_obj_id[episode_idx] = obj_path
                episode_idx += 1
            this_obj_episode_end = episode_idx
            self.obj_id_to_all_episodes_indices[obj_path] = [i for i in range(this_obj_episode_beg, this_obj_episode_end)]            

        cprint('Preparing all zarr paths', 'green')
        self.episode_lengths = []
        self.episode_idx_to_grasp_frame_idx = {}
        self.episode_idx_to_open_frame_idx = {}
        for idx, zarr_path in enumerate(tqdm(self.all_zarr_paths)):
            if is_pickle:
                all_substeps = os.listdir(zarr_path)
                all_substeps = sorted(all_substeps, key=lambda x: int(x.split('.')[0]))
                    
                first_goal = None

                for i, substep in enumerate(all_substeps):
                    if eval_episode is not None and i >=1:
                        self.episode_lengths.append(i)
                        break

                    substep_path = os.path.join(zarr_path, substep)
                    with open(substep_path, 'rb') as f:
                        try:
                            data = pickle.load(f)
                        except:
                            print(substep_path)
                    action = data['action'][:]

                    current_goal = data['goal_gripper_pcd'][:]
                    if first_goal is None:
                        first_goal = current_goal
                    elif only_first_stage and not np.allclose(first_goal, current_goal):
                        self.episode_lengths.append(i)
                        break
                    
                    if not np.allclose(first_goal, current_goal):
                        self.episode_idx_to_grasp_frame_idx[idx] = i
                
                # assume -10 erases all the distorted goal. This is just an approximation. 
                self.episode_idx_to_open_frame_idx[idx] = len(all_substeps) - 1 #- 10

            
            else:
                all_substeps = os.listdir(zarr_path)
                all_substeps = sorted(all_substeps, key=lambda x: int(x))

                first_goal = None

                for i, substep in enumerate(all_substeps):
                    
                    if eval_episode is not None and i >=1:
                        self.episode_lengths.append(i)
                        break

                    substep_path = os.path.join(zarr_path, substep)
                    group = zarr.open(substep_path, 'r')
                    src_store = group.store
                    src_root = zarr.group(src_store)

                    action = src_root['data']['action'][:]

                    current_goal = src_root['data']['goal_gripper_pcd'][:]
                    if first_goal is None:
                        first_goal = current_goal
                    elif only_first_stage and not np.allclose(first_goal, current_goal):
                        self.episode_lengths.append(i)
                        break

            if not only_first_stage and eval_episode is None:
                self.episode_lengths.append(len(all_substeps))
                
        # exit()

        self.episode_lengths = np.array(self.episode_lengths)
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        cprint(f'Finished preparing all zarr paths with total datapoints: {self.accumulated_episode_lengths[-1]}', 'green')

    def __len__(self):
        return self.accumulated_episode_lengths[-1]
    
    def read_pickle_data(self, episode_idx, step_idx):
        step_path = os.path.join(self.all_zarr_paths[episode_idx], str(step_idx) + '.pkl')
        with open(step_path, 'rb') as f:
            data = pickle.load(f)
        pointcloud = data['point_cloud'][:][0].astype(np.float32)
        gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
        goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)
        gripper_open = data['action'][-2].astype(np.float32)
        collision = data['action'][-1].astype(np.float32)
        return pointcloud, gripper_pcd, goal_gripper_pcd, gripper_open, collision

    def __getitem__(self, idx):
        # TODO for conditioning:
        # after we gete episode_idx, figure out which object this episode is from, and randomly sample another episode from this same object.
        # return additionally the grasping and opening pose of this other trajectory for this object. 
        
        episode_idx = np.searchsorted(self.accumulated_episode_lengths, idx, side='right')
        start_idx = idx - self.accumulated_episode_lengths[episode_idx]

        if start_idx < 0:
            start_idx += self.episode_lengths[episode_idx]
            
        if self.is_pickle:
            # step_path = os.path.join(self.all_zarr_paths[episode_idx], str(start_idx) + '.pkl')
            # with open(step_path, 'rb') as f:
            #     data = pickle.load(f)
            # pointcloud = data['point_cloud'][:][0].astype(np.float32)
            # gripper_pcd = data['gripper_pcd'][:][0].astype(np.float32)
            # goal_gripper_pcd = data['goal_gripper_pcd'][:][0].astype(np.float32)
            pointcloud, gripper_pcd, goal_gripper_pcd, gripper_open, collision = self.read_pickle_data(episode_idx, start_idx)
            remaining_steps = self.n_obs_steps - 1
            gripper_pcd_history = []
            for i in range(start_idx - remaining_steps, start_idx):
                _, i_gripper_pcd, _  = self.read_pickle_data(episode_idx, max(0, i))
                gripper_pcd_history.append(i_gripper_pcd)
            if len(gripper_pcd_history) > 0:
                gripper_pcd_history = np.stack(gripper_pcd_history)

            if self.conditioning_on_demo:
                obj_id = self.episode_idx_to_obj_id[episode_idx]
                this_obj_episodes = self.obj_id_to_all_episodes_indices[obj_id]
                random_other_traj_idx = random.choice(this_obj_episodes)
                while random_other_traj_idx == episode_idx:
                    random_other_traj_idx = random.choice(this_obj_episodes)
                grasp_frame_idx, open_frame_idx = self.episode_idx_to_grasp_frame_idx[random_other_traj_idx], self.episode_idx_to_open_frame_idx[random_other_traj_idx]
                # import pdb; pdb.set_trace()
                # this actually reads the first frame
                demo_grasp_pcd, demo_grasp_gripper_pcd, demo_grasp_goal_gripper_pcd = self.read_pickle_data(random_other_traj_idx, 0)

                # and this reads the last -10 frame
                demo_open_pcd, demo_open_gripper_pcd, demo_open_goal_gripper_pcd = self.read_pickle_data(random_other_traj_idx, open_frame_idx)
                    
        else:
            zarr_path = self.all_zarr_paths[episode_idx]
            
            step_path = os.path.join(zarr_path, str(start_idx))
            group = zarr.open(step_path, 'r')
            src_store = group.store
            src_root = zarr.group(src_store)
            pointcloud = src_root['data']['point_cloud'][:][0]
            gripper_pcd = src_root['data']['gripper_pcd'][:][0]
            goal_gripper_pcd = src_root['data']['goal_gripper_pcd'][:][0]

        if not self.conditioning_on_demo:
            if self.n_obs_steps > 1:
                return pointcloud, gripper_pcd, goal_gripper_pcd, gripper_pcd_history
            return pointcloud, gripper_pcd, goal_gripper_pcd, gripper_open, collision
        else:
            if False:
                from matplotlib import pyplot as plt
                fig = plt.figure(figsize=(12, 4))
                ax1 = fig.add_subplot(131, projection='3d')
                ax2 = fig.add_subplot(132, projection='3d')
                ax3 = fig.add_subplot(133, projection='3d')

                ax1.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], color='grey')
                ax1.scatter(gripper_pcd[:, 0], gripper_pcd[:, 1], gripper_pcd[:, 2], s=20, color='blue')
                ax1.scatter(goal_gripper_pcd[:, 0], goal_gripper_pcd[:, 1], goal_gripper_pcd[:, 2], s=20, color='red')

                ax2.scatter(demo_grasp_pcd[:, 0], demo_grasp_pcd[:, 1], demo_grasp_pcd[:, 2], color='grey')
                ax2.scatter(demo_grasp_gripper_pcd[:, 0], demo_grasp_gripper_pcd[:, 1], demo_grasp_gripper_pcd[:, 2], s=20, color='blue')
                ax2.scatter(demo_grasp_goal_gripper_pcd[:, 0], demo_grasp_goal_gripper_pcd[:, 1], demo_grasp_goal_gripper_pcd[:, 2], s=20, color='red')

                ax3.scatter(demo_open_pcd[:, 0], demo_open_pcd[:, 1], demo_open_pcd[:, 2], color='grey')
                ax3.scatter(demo_open_gripper_pcd[:, 0], demo_open_gripper_pcd[:, 1], demo_open_gripper_pcd[:, 2], s=20, color='blue')
                ax3.scatter(demo_open_goal_gripper_pcd[:, 0], demo_open_goal_gripper_pcd[:, 1], demo_open_goal_gripper_pcd[:, 2], s=20, color='red')

                plt.show()
            return {
                "pointcloud": pointcloud,
                "gripper_pcd": gripper_pcd,
                "goal_gripper_pcd": goal_gripper_pcd, 
                "demo_grasp_pcd": demo_grasp_pcd,
                "demo_grasp_gripper_pcd": demo_grasp_gripper_pcd,
                "demo_grasp_goal_gripper_pcd": demo_grasp_goal_gripper_pcd,
                "demo_open_pcd": demo_open_pcd,
                "demo_open_gripper_pcd": demo_open_gripper_pcd,
                "demo_open_goal_gripper_pcd": demo_open_goal_gripper_pcd,
                "gripper_pcd_history": gripper_pcd_history,
            }
    
class PredictTwoGoalsDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, all_obj_paths, beg_ratio=0, end_ratio=0.9, eval_episode=None, only_first_stage=False, is_pickle=False, use_all_data=False):
        self.all_obj_paths = all_obj_paths
        self.beg_ratio = beg_ratio
        self.end_ratio = end_ratio
        self.is_pickle = is_pickle
        self.use_all_data = use_all_data
        
        if only_first_stage:
            cprint('======= ONLY FIRST STAGE =======', 'red')

        if eval_episode is not None:
            cprint('======= EVAL MODE =======', 'red')
            cprint(f'Only evaluating the first observation of {eval_episode} episodes', 'red')

        self.all_zarr_paths = []
        for obj_path in all_obj_paths:
            all_subfolder = os.listdir(obj_path)
            for s in ['action_dist', 'demo_rgbs', 'all_demo_path.txt', 'meta_info.json', 'example_pointcloud']:
                if s in all_subfolder:
                    all_subfolder.remove(s)
            all_subfolder = sorted(all_subfolder)
            beg = int(beg_ratio * len(all_subfolder))
            end = int(end_ratio * len(all_subfolder))
            if not self.use_all_data:
                end = min(end, 75)
            if eval_episode is not None:
                end = beg + eval_episode
            all_subfolder = all_subfolder[beg:end]
            self.all_zarr_paths += [os.path.join(obj_path, s) for s in all_subfolder]

        cprint('Preparing all zarr paths', 'green')
        self.episode_lengths = []

        for idx, zarr_path in enumerate(tqdm(self.all_zarr_paths)):
            if is_pickle:
                all_substeps = os.listdir(zarr_path)
                all_substeps = sorted(all_substeps, key=lambda x: int(x.split('.')[0]))
                    
                first_goal = None

                for i, substep in enumerate(all_substeps):
                    if eval_episode is not None and i >=1:
                        self.episode_lengths.append(i)
                        break

                    substep_path = os.path.join(zarr_path, substep)
                    with open(substep_path, 'rb') as f:
                        data = pickle.load(f)
                    action = data['action'][:]

                    current_goal = data['goal_gripper_pcd'][:]
                    if first_goal is None:
                        first_goal = current_goal
                    elif only_first_stage and not np.allclose(first_goal, current_goal):
                        self.episode_lengths.append(i)
                        break

            
            else:
                all_substeps = os.listdir(zarr_path)
                all_substeps = sorted(all_substeps, key=lambda x: int(x))

                first_goal = None

                for i, substep in enumerate(all_substeps):
                    
                    if eval_episode is not None and i >=1:
                        self.episode_lengths.append(i)
                        break

                    substep_path = os.path.join(zarr_path, substep)
                    group = zarr.open(substep_path, 'r')
                    src_store = group.store
                    src_root = zarr.group(src_store)

                    action = src_root['data']['action'][:]

                    current_goal = src_root['data']['goal_gripper_pcd'][:]
                    if first_goal is None:
                        first_goal = current_goal
                    elif only_first_stage and not np.allclose(first_goal, current_goal):
                        self.episode_lengths.append(i)
                        break

            if not only_first_stage and eval_episode is None:
                self.episode_lengths.append(len(all_substeps))

        self.episode_lengths = np.array(self.episode_lengths)
        self.accumulated_episode_lengths = np.cumsum(self.episode_lengths)
        cprint(f'Finished preparing all zarr paths with total datapoints: {self.accumulated_episode_lengths[-1]}', 'green')

    def __len__(self):
        return len(self.episode_lengths) # num data points is just num trajectories

    def __getitem__(self, idx):
        episode_idx = idx
        start_idx = 0
        end_idx = self.episode_lengths[episode_idx] - 1

        if self.is_pickle:
            step_path = os.path.join(self.all_zarr_paths[episode_idx], str(start_idx) + '.pkl')
            with open(step_path, 'rb') as f:
                data = pickle.load(f)
            pointcloud = data['point_cloud'][:][0]
            gripper_pcd = data['gripper_pcd'][:][0]
            goal_gripper_pcd = data['goal_gripper_pcd'][:][0]

            step_path_end = os.path.join(self.all_zarr_paths[episode_idx], str(end_idx) + '.pkl')
            with open(step_path_end, 'rb') as f:
                data_end = pickle.load(f)
            goal_gripper_pcd_end = data_end['goal_gripper_pcd'][:][0]
        else:
            zarr_path = self.all_zarr_paths[episode_idx]
            
            step_path = os.path.join(zarr_path, str(start_idx))
            group = zarr.open(step_path, 'r')
            src_store = group.store
            src_root = zarr.group(src_store)
            pointcloud = src_root['data']['point_cloud'][:][0]
            gripper_pcd = src_root['data']['gripper_pcd'][:][0]
            goal_gripper_pcd = src_root['data']['goal_gripper_pcd'][:][0]

        return pointcloud, gripper_pcd, np.concatenate([goal_gripper_pcd, goal_gripper_pcd_end], axis=0)
        
def get_dataloader(all_obj_paths=None, batch_size=32, beg_ratio=0, end_ratio=0.9, shuffle=True, eval_episode=None, only_first_stage=False):
    if all_obj_paths is None:
        all_obj_paths = ['0705-obj-41510', '0705-obj-45448', '0705-obj-46462', '0705-obj-46732', '0705-obj-46801', '0705-obj-46874', '0705-obj-46922', '0705-obj-46966', '0705-obj-47570', '0705-obj-47578', '0705-obj-48700', '0705-obj-45526', '0705-obj-45661', '0705-obj-45694', '0705-obj-45780', '0705-obj-45910', '0705-obj-45961', '0705-obj-46408', '0705-obj-46417', '0705-obj-46440', '0705-obj-46490', '0705-obj-46762', '0705-obj-46825', '0705-obj-46893', '0705-obj-47235', '0705-obj-47281', '0705-obj-47315', '0705-obj-47529', '0705-obj-47669', '0705-obj-47944', '0705-obj-48063', '0705-obj-48177', '0705-obj-48356', '0705-obj-48623', '0705-obj-48876', '0705-obj-49025', '0705-obj-49062', '0705-obj-49132', '0705-obj-49133', '0712-obj-40417', '0712-obj-41085', '0712-obj-41452', '0712-obj-45162', '0712-obj-45176', '0712-obj-45194', '0712-obj-45203', '0712-obj-45248', '0712-obj-45271', '0712-obj-45290', '0712-obj-45305']
        all_obj_paths = ['/scratch/chialiang/dp3_demo/' + s for s in all_obj_paths]
    dataset = PointNetDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# from test_PointNet2.all_data import *
# from scripts.datasets.randomize_partition_10_obj import *
# from scripts.datasets.randomize_partition_50_obj import *
# from scripts.datasets.randomize_partition_100_obj import *
# from scripts.datasets.randomize_partition_200_obj import *

def get_dataloader_from_pickle(all_obj_paths=None, batch_size=32, beg_ratio=0, end_ratio=0.9, shuffle=True, eval_episode=None, only_first_stage=False):
    dataset_prefix='/scratch/chialiang/dp3_demo'
    if all_obj_paths is None:
        all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
        f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
        f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
        f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
        f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
        f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
        f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
        f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
        f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
        f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
        f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
        f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
        f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
        f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
        f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
        f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
        f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
        f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
        f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
        f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}',]
    dataset = PointNetDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage, is_pickle=True)    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_dataset_from_pickle(all_obj_paths=None, beg_ratio=0, end_ratio=0.9, eval_episode=None, only_first_stage=False, 
                            use_all_data=False, use_combined_action=False, dataset_prefix=None, num_train_objects=200, 
                            predict_two_goals=False, conditioning_on_demo=False, n_obs_steps=1):
    
    if dataset_prefix is None:
        dataset_prefix='/scratch/chialiang/dp3_demo'
        if use_combined_action:
            dataset_prefix='/scratch/chialiang/dp3_demo_combine_2_new'
    
    if all_obj_paths is None:
        print("num_train_objects: ", num_train_objects)
        print("num_train_objects: ", num_train_objects)
        print("num_train_objects: ", num_train_objects)
        print("num_train_objects: ", num_train_objects)
        print("num_train_objects: ", num_train_objects)
        if num_train_objects == 'debug':
            all_obj_paths = [f'{dataset_prefix}/0628-act3d-obj-47570-gripper-goal-1-displacement-to-object-1-combined-steps-2-filter-zero-close-action-1']
        elif num_train_objects == '10':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(10)]
        elif num_train_objects == '50':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(50)]
        elif num_train_objects == '100':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(100)]
        elif num_train_objects == '200':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(200)]
        elif num_train_objects == '300':
            all_zarr_paths_part_1 = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(246)]
            all_subfolders = sorted(os.listdir(dataset_prefix))
            object_other_categories_no_cam_rand = [x for x in all_subfolders if "1121-other-cat-no-cam-rand" in x]
            all_zarr_paths_part_2 = [f"{dataset_prefix}/{x}" for x in object_other_categories_no_cam_rand]
            all_obj_paths = all_zarr_paths_part_1 + all_zarr_paths_part_2
            
        elif num_train_objects == "camera_random_10_obj_high_level":
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_10_save_data_name_{}".format(i)]) for i in range(20)]
        elif num_train_objects == 'camera_random_50_obj_high_level':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_50_save_data_name_{}".format(i)]) for i in range(87)]
        elif num_train_objects == 'camera_random_100_obj_high_level':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_100_save_data_name_{}".format(i)]) for i in range(175)]
        elif num_train_objects == 'camera_random_200_obj_high_level':
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["camera_random_200_save_data_name_{}".format(i)]) for i in range(350)]
        elif num_train_objects == 'camera_random_500_obj_high_level' or num_train_objects == "500_object_high_level":
            all_obj_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(462)]
            
        elif num_train_objects == '300_old':
            
            all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
            f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
            f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
            f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
            f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
            f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
            f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
            f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
            f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
            f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
            f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
            f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
            f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
            f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
            f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
            f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
            f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
            f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
            f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
            f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
            f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
            f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
            f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
            f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
            f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
            f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
            f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
            f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
            f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}']
        elif num_train_objects == '500':
            all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
            f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
            f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
            f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
            f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
            f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
            f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
            f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
            f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
            f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
            f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
            f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
            f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
            f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
            f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
            f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
            f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
            f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
            f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
            f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
            f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
            f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
            f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
            f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
            f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
            f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
            f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
            f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
            f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}',
            f'{dataset_prefix}/{save_data_name_287}', f'{dataset_prefix}/{save_data_name_288}', f'{dataset_prefix}/{save_data_name_289}', f'{dataset_prefix}/{save_data_name_290}', f'{dataset_prefix}/{save_data_name_291}', f'{dataset_prefix}/{save_data_name_292}', f'{dataset_prefix}/{save_data_name_293}', f'{dataset_prefix}/{save_data_name_294}', f'{dataset_prefix}/{save_data_name_295}', f'{dataset_prefix}/{save_data_name_296}', f'{dataset_prefix}/{save_data_name_297}', f'{dataset_prefix}/{save_data_name_298}', f'{dataset_prefix}/{save_data_name_299}', f'{dataset_prefix}/{save_data_name_300}', f'{dataset_prefix}/{save_data_name_301}', f'{dataset_prefix}/{save_data_name_302}', f'{dataset_prefix}/{save_data_name_303}', f'{dataset_prefix}/{save_data_name_304}', f'{dataset_prefix}/{save_data_name_305}', f'{dataset_prefix}/{save_data_name_306}', f'{dataset_prefix}/{save_data_name_307}', f'{dataset_prefix}/{save_data_name_308}', f'{dataset_prefix}/{save_data_name_309}', f'{dataset_prefix}/{save_data_name_310}', f'{dataset_prefix}/{save_data_name_311}', f'{dataset_prefix}/{save_data_name_312}', f'{dataset_prefix}/{save_data_name_313}', f'{dataset_prefix}/{save_data_name_314}', f'{dataset_prefix}/{save_data_name_315}', f'{dataset_prefix}/{save_data_name_316}', f'{dataset_prefix}/{save_data_name_317}', f'{dataset_prefix}/{save_data_name_318}', f'{dataset_prefix}/{save_data_name_319}', f'{dataset_prefix}/{save_data_name_320}', f'{dataset_prefix}/{save_data_name_321}', f'{dataset_prefix}/{save_data_name_322}', f'{dataset_prefix}/{save_data_name_323}', f'{dataset_prefix}/{save_data_name_324}', f'{dataset_prefix}/{save_data_name_325}', f'{dataset_prefix}/{save_data_name_326}', f'{dataset_prefix}/{save_data_name_327}', f'{dataset_prefix}/{save_data_name_328}', f'{dataset_prefix}/{save_data_name_329}', f'{dataset_prefix}/{save_data_name_330}', f'{dataset_prefix}/{save_data_name_331}', f'{dataset_prefix}/{save_data_name_332}', f'{dataset_prefix}/{save_data_name_333}', f'{dataset_prefix}/{save_data_name_334}', f'{dataset_prefix}/{save_data_name_335}', f'{dataset_prefix}/{save_data_name_336}', f'{dataset_prefix}/{save_data_name_337}', f'{dataset_prefix}/{save_data_name_338}', f'{dataset_prefix}/{save_data_name_339}', f'{dataset_prefix}/{save_data_name_340}', f'{dataset_prefix}/{save_data_name_341}', f'{dataset_prefix}/{save_data_name_342}', f'{dataset_prefix}/{save_data_name_343}', f'{dataset_prefix}/{save_data_name_344}', f'{dataset_prefix}/{save_data_name_345}', f'{dataset_prefix}/{save_data_name_346}', f'{dataset_prefix}/{save_data_name_347}', f'{dataset_prefix}/{save_data_name_348}', f'{dataset_prefix}/{save_data_name_349}', f'{dataset_prefix}/{save_data_name_350}', f'{dataset_prefix}/{save_data_name_351}', f'{dataset_prefix}/{save_data_name_352}', f'{dataset_prefix}/{save_data_name_353}', f'{dataset_prefix}/{save_data_name_354}', f'{dataset_prefix}/{save_data_name_355}', f'{dataset_prefix}/{save_data_name_356}', f'{dataset_prefix}/{save_data_name_357}', f'{dataset_prefix}/{save_data_name_358}', f'{dataset_prefix}/{save_data_name_359}', f'{dataset_prefix}/{save_data_name_360}', f'{dataset_prefix}/{save_data_name_361}', f'{dataset_prefix}/{save_data_name_362}', f'{dataset_prefix}/{save_data_name_363}', f'{dataset_prefix}/{save_data_name_364}', f'{dataset_prefix}/{save_data_name_365}', f'{dataset_prefix}/{save_data_name_366}', f'{dataset_prefix}/{save_data_name_367}', f'{dataset_prefix}/{save_data_name_368}', f'{dataset_prefix}/{save_data_name_369}', f'{dataset_prefix}/{save_data_name_370}', f'{dataset_prefix}/{save_data_name_371}', f'{dataset_prefix}/{save_data_name_372}', f'{dataset_prefix}/{save_data_name_373}', f'{dataset_prefix}/{save_data_name_374}', f'{dataset_prefix}/{save_data_name_375}', f'{dataset_prefix}/{save_data_name_376}', f'{dataset_prefix}/{save_data_name_377}', f'{dataset_prefix}/{save_data_name_378}', f'{dataset_prefix}/{save_data_name_379}', f'{dataset_prefix}/{save_data_name_380}', f'{dataset_prefix}/{save_data_name_381}', f'{dataset_prefix}/{save_data_name_382}', f'{dataset_prefix}/{save_data_name_383}', f'{dataset_prefix}/{save_data_name_384}', f'{dataset_prefix}/{save_data_name_385}', f'{dataset_prefix}/{save_data_name_386}', f'{dataset_prefix}/{save_data_name_387}', f'{dataset_prefix}/{save_data_name_388}', f'{dataset_prefix}/{save_data_name_389}', f'{dataset_prefix}/{save_data_name_390}', f'{dataset_prefix}/{save_data_name_391}', f'{dataset_prefix}/{save_data_name_392}', f'{dataset_prefix}/{save_data_name_393}', f'{dataset_prefix}/{save_data_name_394}', f'{dataset_prefix}/{save_data_name_395}', f'{dataset_prefix}/{save_data_name_396}', f'{dataset_prefix}/{save_data_name_397}', f'{dataset_prefix}/{save_data_name_398}', f'{dataset_prefix}/{save_data_name_399}', f'{dataset_prefix}/{save_data_name_400}', f'{dataset_prefix}/{save_data_name_401}', f'{dataset_prefix}/{save_data_name_402}', f'{dataset_prefix}/{save_data_name_403}', f'{dataset_prefix}/{save_data_name_404}', f'{dataset_prefix}/{save_data_name_405}', f'{dataset_prefix}/{save_data_name_406}', f'{dataset_prefix}/{save_data_name_407}', f'{dataset_prefix}/{save_data_name_408}', f'{dataset_prefix}/{save_data_name_409}', f'{dataset_prefix}/{save_data_name_410}', f'{dataset_prefix}/{save_data_name_411}', f'{dataset_prefix}/{save_data_name_412}', f'{dataset_prefix}/{save_data_name_413}', f'{dataset_prefix}/{save_data_name_414}', f'{dataset_prefix}/{save_data_name_415}', f'{dataset_prefix}/{save_data_name_416}', f'{dataset_prefix}/{save_data_name_417}', f'{dataset_prefix}/{save_data_name_418}', f'{dataset_prefix}/{save_data_name_419}', f'{dataset_prefix}/{save_data_name_420}', f'{dataset_prefix}/{save_data_name_421}', f'{dataset_prefix}/{save_data_name_422}', f'{dataset_prefix}/{save_data_name_423}', f'{dataset_prefix}/{save_data_name_424}', f'{dataset_prefix}/{save_data_name_425}', f'{dataset_prefix}/{save_data_name_426}', f'{dataset_prefix}/{save_data_name_427}', f'{dataset_prefix}/{save_data_name_428}', f'{dataset_prefix}/{save_data_name_429}', f'{dataset_prefix}/{save_data_name_430}', f'{dataset_prefix}/{save_data_name_431}', f'{dataset_prefix}/{save_data_name_432}', f'{dataset_prefix}/{save_data_name_433}', f'{dataset_prefix}/{save_data_name_434}', f'{dataset_prefix}/{save_data_name_435}', f'{dataset_prefix}/{save_data_name_436}', f'{dataset_prefix}/{save_data_name_437}', f'{dataset_prefix}/{save_data_name_438}', f'{dataset_prefix}/{save_data_name_439}', f'{dataset_prefix}/{save_data_name_440}', f'{dataset_prefix}/{save_data_name_441}', f'{dataset_prefix}/{save_data_name_442}', f'{dataset_prefix}/{save_data_name_443}', f'{dataset_prefix}/{save_data_name_444}', f'{dataset_prefix}/{save_data_name_445}', f'{dataset_prefix}/{save_data_name_446}', f'{dataset_prefix}/{save_data_name_447}', f'{dataset_prefix}/{save_data_name_448}', f'{dataset_prefix}/{save_data_name_449}', f'{dataset_prefix}/{save_data_name_450}', f'{dataset_prefix}/{save_data_name_451}', f'{dataset_prefix}/{save_data_name_452}', f'{dataset_prefix}/{save_data_name_453}', f'{dataset_prefix}/{save_data_name_454}', f'{dataset_prefix}/{save_data_name_455}', f'{dataset_prefix}/{save_data_name_456}', f'{dataset_prefix}/{save_data_name_457}', f'{dataset_prefix}/{save_data_name_458}', f'{dataset_prefix}/{save_data_name_459}', f'{dataset_prefix}/{save_data_name_460}', f'{dataset_prefix}/{save_data_name_461}', f'{dataset_prefix}/{save_data_name_462}',
            ]
        elif num_train_objects == '600':
            all_obj_paths = [f'{dataset_prefix}/{save_data_name_0}', f'{dataset_prefix}/{save_data_name_1}', f'{dataset_prefix}/{save_data_name_2}', f'{dataset_prefix}/{save_data_name_3}', f'{dataset_prefix}/{save_data_name_4}', f'{dataset_prefix}/{save_data_name_5}', f'{dataset_prefix}/{save_data_name_6}', f'{dataset_prefix}/{save_data_name_7}', f'{dataset_prefix}/{save_data_name_8}', f'{dataset_prefix}/{save_data_name_9}', 
            f'{dataset_prefix}/{save_data_name_10}', f'{dataset_prefix}/{save_data_name_11}', f'{dataset_prefix}/{save_data_name_12}', f'{dataset_prefix}/{save_data_name_13}', f'{dataset_prefix}/{save_data_name_14}', f'{dataset_prefix}/{save_data_name_15}', f'{dataset_prefix}/{save_data_name_16}', f'{dataset_prefix}/{save_data_name_17}', f'{dataset_prefix}/{save_data_name_18}', f'{dataset_prefix}/{save_data_name_19}', 
            f'{dataset_prefix}/{save_data_name_20}', f'{dataset_prefix}/{save_data_name_21}', f'{dataset_prefix}/{save_data_name_22}', f'{dataset_prefix}/{save_data_name_23}', f'{dataset_prefix}/{save_data_name_24}', f'{dataset_prefix}/{save_data_name_25}', f'{dataset_prefix}/{save_data_name_26}', f'{dataset_prefix}/{save_data_name_27}', f'{dataset_prefix}/{save_data_name_28}', f'{dataset_prefix}/{save_data_name_29}', 
            f'{dataset_prefix}/{save_data_name_30}', f'{dataset_prefix}/{save_data_name_31}', f'{dataset_prefix}/{save_data_name_32}', f'{dataset_prefix}/{save_data_name_33}', f'{dataset_prefix}/{save_data_name_34}', f'{dataset_prefix}/{save_data_name_35}', f'{dataset_prefix}/{save_data_name_36}', f'{dataset_prefix}/{save_data_name_37}', f'{dataset_prefix}/{save_data_name_38}', f'{dataset_prefix}/{save_data_name_39}', 
            f'{dataset_prefix}/{save_data_name_40}', f'{dataset_prefix}/{save_data_name_41}', f'{dataset_prefix}/{save_data_name_42}', f'{dataset_prefix}/{save_data_name_43}', f'{dataset_prefix}/{save_data_name_44}', f'{dataset_prefix}/{save_data_name_45}', f'{dataset_prefix}/{save_data_name_46}', f'{dataset_prefix}/{save_data_name_47}', f'{dataset_prefix}/{save_data_name_48}', f'{dataset_prefix}/{save_data_name_49}',
            f'{dataset_prefix}/{save_data_name_50}', f'{dataset_prefix}/{save_data_name_51}', f'{dataset_prefix}/{save_data_name_52}', f'{dataset_prefix}/{save_data_name_53}', f'{dataset_prefix}/{save_data_name_54}', f'{dataset_prefix}/{save_data_name_55}', f'{dataset_prefix}/{save_data_name_56}', f'{dataset_prefix}/{save_data_name_57}', f'{dataset_prefix}/{save_data_name_58}', f'{dataset_prefix}/{save_data_name_59}',
            f'{dataset_prefix}/{save_data_name_60}', f'{dataset_prefix}/{save_data_name_61}', f'{dataset_prefix}/{save_data_name_62}', f'{dataset_prefix}/{save_data_name_63}', f'{dataset_prefix}/{save_data_name_64}', f'{dataset_prefix}/{save_data_name_65}', f'{dataset_prefix}/{save_data_name_66}', f'{dataset_prefix}/{save_data_name_67}', f'{dataset_prefix}/{save_data_name_68}', f'{dataset_prefix}/{save_data_name_69}',
            f'{dataset_prefix}/{save_data_name_70}', f'{dataset_prefix}/{save_data_name_71}', f'{dataset_prefix}/{save_data_name_72}', f'{dataset_prefix}/{save_data_name_73}', f'{dataset_prefix}/{save_data_name_74}', f'{dataset_prefix}/{save_data_name_75}', f'{dataset_prefix}/{save_data_name_76}', f'{dataset_prefix}/{save_data_name_77}', f'{dataset_prefix}/{save_data_name_78}', f'{dataset_prefix}/{save_data_name_79}',
            f'{dataset_prefix}/{save_data_name_80}', f'{dataset_prefix}/{save_data_name_81}', f'{dataset_prefix}/{save_data_name_82}', f'{dataset_prefix}/{save_data_name_83}', f'{dataset_prefix}/{save_data_name_84}', f'{dataset_prefix}/{save_data_name_85}', f'{dataset_prefix}/{save_data_name_86}', f'{dataset_prefix}/{save_data_name_87}', f'{dataset_prefix}/{save_data_name_88}', f'{dataset_prefix}/{save_data_name_89}',
            f'{dataset_prefix}/{save_data_name_90}', f'{dataset_prefix}/{save_data_name_91}', f'{dataset_prefix}/{save_data_name_92}', f'{dataset_prefix}/{save_data_name_93}', f'{dataset_prefix}/{save_data_name_94}', f'{dataset_prefix}/{save_data_name_95}', f'{dataset_prefix}/{save_data_name_96}', f'{dataset_prefix}/{save_data_name_97}', f'{dataset_prefix}/{save_data_name_98}', f'{dataset_prefix}/{save_data_name_99}',
            f'{dataset_prefix}/{save_data_name_100}', f'{dataset_prefix}/{save_data_name_101}', f'{dataset_prefix}/{save_data_name_102}', f'{dataset_prefix}/{save_data_name_103}', f'{dataset_prefix}/{save_data_name_104}', f'{dataset_prefix}/{save_data_name_105}', f'{dataset_prefix}/{save_data_name_106}', f'{dataset_prefix}/{save_data_name_107}', f'{dataset_prefix}/{save_data_name_108}', f'{dataset_prefix}/{save_data_name_109}',
            f'{dataset_prefix}/{save_data_name_110}', f'{dataset_prefix}/{save_data_name_111}', f'{dataset_prefix}/{save_data_name_112}', f'{dataset_prefix}/{save_data_name_113}', f'{dataset_prefix}/{save_data_name_114}', f'{dataset_prefix}/{save_data_name_115}', f'{dataset_prefix}/{save_data_name_116}', f'{dataset_prefix}/{save_data_name_117}', f'{dataset_prefix}/{save_data_name_118}', f'{dataset_prefix}/{save_data_name_119}',
            f'{dataset_prefix}/{save_data_name_120}', f'{dataset_prefix}/{save_data_name_121}', f'{dataset_prefix}/{save_data_name_122}', f'{dataset_prefix}/{save_data_name_123}', f'{dataset_prefix}/{save_data_name_124}', f'{dataset_prefix}/{save_data_name_125}', f'{dataset_prefix}/{save_data_name_126}', f'{dataset_prefix}/{save_data_name_127}', f'{dataset_prefix}/{save_data_name_128}', f'{dataset_prefix}/{save_data_name_129}',
            f'{dataset_prefix}/{save_data_name_130}', f'{dataset_prefix}/{save_data_name_131}', f'{dataset_prefix}/{save_data_name_132}', f'{dataset_prefix}/{save_data_name_133}', f'{dataset_prefix}/{save_data_name_134}', f'{dataset_prefix}/{save_data_name_135}', f'{dataset_prefix}/{save_data_name_136}', f'{dataset_prefix}/{save_data_name_137}', f'{dataset_prefix}/{save_data_name_138}', f'{dataset_prefix}/{save_data_name_139}',
            f'{dataset_prefix}/{save_data_name_140}', f'{dataset_prefix}/{save_data_name_141}', f'{dataset_prefix}/{save_data_name_142}', f'{dataset_prefix}/{save_data_name_143}', f'{dataset_prefix}/{save_data_name_144}', f'{dataset_prefix}/{save_data_name_145}', f'{dataset_prefix}/{save_data_name_146}', f'{dataset_prefix}/{save_data_name_147}', f'{dataset_prefix}/{save_data_name_148}', f'{dataset_prefix}/{save_data_name_149}',
            f'{dataset_prefix}/{save_data_name_150}', f'{dataset_prefix}/{save_data_name_151}', f'{dataset_prefix}/{save_data_name_152}', f'{dataset_prefix}/{save_data_name_153}', f'{dataset_prefix}/{save_data_name_154}', f'{dataset_prefix}/{save_data_name_155}', f'{dataset_prefix}/{save_data_name_156}', f'{dataset_prefix}/{save_data_name_157}', f'{dataset_prefix}/{save_data_name_158}', f'{dataset_prefix}/{save_data_name_159}',
            f'{dataset_prefix}/{save_data_name_160}', f'{dataset_prefix}/{save_data_name_161}', f'{dataset_prefix}/{save_data_name_162}', f'{dataset_prefix}/{save_data_name_163}', f'{dataset_prefix}/{save_data_name_164}', f'{dataset_prefix}/{save_data_name_165}', f'{dataset_prefix}/{save_data_name_166}', f'{dataset_prefix}/{save_data_name_167}', f'{dataset_prefix}/{save_data_name_168}', f'{dataset_prefix}/{save_data_name_169}',
            f'{dataset_prefix}/{save_data_name_170}', f'{dataset_prefix}/{save_data_name_171}', f'{dataset_prefix}/{save_data_name_172}', f'{dataset_prefix}/{save_data_name_173}', f'{dataset_prefix}/{save_data_name_174}', f'{dataset_prefix}/{save_data_name_175}', f'{dataset_prefix}/{save_data_name_176}', f'{dataset_prefix}/{save_data_name_177}', f'{dataset_prefix}/{save_data_name_178}', f'{dataset_prefix}/{save_data_name_179}',
            f'{dataset_prefix}/{save_data_name_180}', f'{dataset_prefix}/{save_data_name_181}', f'{dataset_prefix}/{save_data_name_182}', f'{dataset_prefix}/{save_data_name_183}', f'{dataset_prefix}/{save_data_name_184}', f'{dataset_prefix}/{save_data_name_185}', f'{dataset_prefix}/{save_data_name_186}', f'{dataset_prefix}/{save_data_name_187}', f'{dataset_prefix}/{save_data_name_188}', f'{dataset_prefix}/{save_data_name_189}',
            f'{dataset_prefix}/{save_data_name_190}', f'{dataset_prefix}/{save_data_name_191}', f'{dataset_prefix}/{save_data_name_192}', f'{dataset_prefix}/{save_data_name_193}', f'{dataset_prefix}/{save_data_name_194}', f'{dataset_prefix}/{save_data_name_195}', f'{dataset_prefix}/{save_data_name_196}', f'{dataset_prefix}/{save_data_name_197}', f'{dataset_prefix}/{save_data_name_198}', f'{dataset_prefix}/{save_data_name_199}',
            f'{dataset_prefix}/{save_data_name_200}', f'{dataset_prefix}/{save_data_name_201}', f'{dataset_prefix}/{save_data_name_202}', f'{dataset_prefix}/{save_data_name_203}', f'{dataset_prefix}/{save_data_name_204}', f'{dataset_prefix}/{save_data_name_205}', f'{dataset_prefix}/{save_data_name_206}', f'{dataset_prefix}/{save_data_name_207}', f'{dataset_prefix}/{save_data_name_208}', f'{dataset_prefix}/{save_data_name_209}',
            f'{dataset_prefix}/{save_data_name_210}', f'{dataset_prefix}/{save_data_name_211}', f'{dataset_prefix}/{save_data_name_212}', f'{dataset_prefix}/{save_data_name_213}', f'{dataset_prefix}/{save_data_name_214}', f'{dataset_prefix}/{save_data_name_215}', f'{dataset_prefix}/{save_data_name_216}', f'{dataset_prefix}/{save_data_name_217}', f'{dataset_prefix}/{save_data_name_218}', f'{dataset_prefix}/{save_data_name_219}',
            f'{dataset_prefix}/{save_data_name_220}', f'{dataset_prefix}/{save_data_name_221}', f'{dataset_prefix}/{save_data_name_222}', f'{dataset_prefix}/{save_data_name_223}', f'{dataset_prefix}/{save_data_name_224}', f'{dataset_prefix}/{save_data_name_225}', f'{dataset_prefix}/{save_data_name_226}', f'{dataset_prefix}/{save_data_name_227}', f'{dataset_prefix}/{save_data_name_228}', f'{dataset_prefix}/{save_data_name_229}',
            f'{dataset_prefix}/{save_data_name_230}', f'{dataset_prefix}/{save_data_name_231}', f'{dataset_prefix}/{save_data_name_232}', f'{dataset_prefix}/{save_data_name_233}', f'{dataset_prefix}/{save_data_name_234}', f'{dataset_prefix}/{save_data_name_235}', f'{dataset_prefix}/{save_data_name_236}', f'{dataset_prefix}/{save_data_name_237}', f'{dataset_prefix}/{save_data_name_238}', f'{dataset_prefix}/{save_data_name_239}',
            f'{dataset_prefix}/{save_data_name_240}', f'{dataset_prefix}/{save_data_name_241}', f'{dataset_prefix}/{save_data_name_242}', f'{dataset_prefix}/{save_data_name_243}', f'{dataset_prefix}/{save_data_name_244}', f'{dataset_prefix}/{save_data_name_245}', f'{dataset_prefix}/{save_data_name_246}', f'{dataset_prefix}/{save_data_name_247}', f'{dataset_prefix}/{save_data_name_248}', f'{dataset_prefix}/{save_data_name_249}',
            f'{dataset_prefix}/{save_data_name_250}', f'{dataset_prefix}/{save_data_name_251}', f'{dataset_prefix}/{save_data_name_252}', f'{dataset_prefix}/{save_data_name_253}', f'{dataset_prefix}/{save_data_name_254}', f'{dataset_prefix}/{save_data_name_255}', f'{dataset_prefix}/{save_data_name_256}', f'{dataset_prefix}/{save_data_name_257}', f'{dataset_prefix}/{save_data_name_258}', f'{dataset_prefix}/{save_data_name_259}',
            f'{dataset_prefix}/{save_data_name_260}', f'{dataset_prefix}/{save_data_name_261}', f'{dataset_prefix}/{save_data_name_262}', f'{dataset_prefix}/{save_data_name_263}', f'{dataset_prefix}/{save_data_name_264}', f'{dataset_prefix}/{save_data_name_265}', f'{dataset_prefix}/{save_data_name_266}', f'{dataset_prefix}/{save_data_name_267}', f'{dataset_prefix}/{save_data_name_268}', f'{dataset_prefix}/{save_data_name_269}',
            f'{dataset_prefix}/{save_data_name_270}', f'{dataset_prefix}/{save_data_name_271}', f'{dataset_prefix}/{save_data_name_272}', f'{dataset_prefix}/{save_data_name_273}', f'{dataset_prefix}/{save_data_name_274}', f'{dataset_prefix}/{save_data_name_275}', f'{dataset_prefix}/{save_data_name_276}', f'{dataset_prefix}/{save_data_name_277}', f'{dataset_prefix}/{save_data_name_278}', f'{dataset_prefix}/{save_data_name_279}',
            f'{dataset_prefix}/{save_data_name_280}', f'{dataset_prefix}/{save_data_name_281}', f'{dataset_prefix}/{save_data_name_282}', f'{dataset_prefix}/{save_data_name_283}', f'{dataset_prefix}/{save_data_name_284}', f'{dataset_prefix}/{save_data_name_285}', f'{dataset_prefix}/{save_data_name_286}',
            f'{dataset_prefix}/{save_data_name_287}', f'{dataset_prefix}/{save_data_name_288}', f'{dataset_prefix}/{save_data_name_289}', f'{dataset_prefix}/{save_data_name_290}', f'{dataset_prefix}/{save_data_name_291}', f'{dataset_prefix}/{save_data_name_292}', f'{dataset_prefix}/{save_data_name_293}', f'{dataset_prefix}/{save_data_name_294}', f'{dataset_prefix}/{save_data_name_295}', f'{dataset_prefix}/{save_data_name_296}', f'{dataset_prefix}/{save_data_name_297}', f'{dataset_prefix}/{save_data_name_298}', f'{dataset_prefix}/{save_data_name_299}', f'{dataset_prefix}/{save_data_name_300}', f'{dataset_prefix}/{save_data_name_301}', f'{dataset_prefix}/{save_data_name_302}', f'{dataset_prefix}/{save_data_name_303}', f'{dataset_prefix}/{save_data_name_304}', f'{dataset_prefix}/{save_data_name_305}', f'{dataset_prefix}/{save_data_name_306}', f'{dataset_prefix}/{save_data_name_307}', f'{dataset_prefix}/{save_data_name_308}', f'{dataset_prefix}/{save_data_name_309}', f'{dataset_prefix}/{save_data_name_310}', f'{dataset_prefix}/{save_data_name_311}', f'{dataset_prefix}/{save_data_name_312}', f'{dataset_prefix}/{save_data_name_313}', f'{dataset_prefix}/{save_data_name_314}', f'{dataset_prefix}/{save_data_name_315}', f'{dataset_prefix}/{save_data_name_316}', f'{dataset_prefix}/{save_data_name_317}', f'{dataset_prefix}/{save_data_name_318}', f'{dataset_prefix}/{save_data_name_319}', f'{dataset_prefix}/{save_data_name_320}', f'{dataset_prefix}/{save_data_name_321}', f'{dataset_prefix}/{save_data_name_322}', f'{dataset_prefix}/{save_data_name_323}', f'{dataset_prefix}/{save_data_name_324}', f'{dataset_prefix}/{save_data_name_325}', f'{dataset_prefix}/{save_data_name_326}', f'{dataset_prefix}/{save_data_name_327}', f'{dataset_prefix}/{save_data_name_328}', f'{dataset_prefix}/{save_data_name_329}', f'{dataset_prefix}/{save_data_name_330}', f'{dataset_prefix}/{save_data_name_331}', f'{dataset_prefix}/{save_data_name_332}', f'{dataset_prefix}/{save_data_name_333}', f'{dataset_prefix}/{save_data_name_334}', f'{dataset_prefix}/{save_data_name_335}', f'{dataset_prefix}/{save_data_name_336}', f'{dataset_prefix}/{save_data_name_337}', f'{dataset_prefix}/{save_data_name_338}', f'{dataset_prefix}/{save_data_name_339}', f'{dataset_prefix}/{save_data_name_340}', f'{dataset_prefix}/{save_data_name_341}', f'{dataset_prefix}/{save_data_name_342}', f'{dataset_prefix}/{save_data_name_343}', f'{dataset_prefix}/{save_data_name_344}', f'{dataset_prefix}/{save_data_name_345}', f'{dataset_prefix}/{save_data_name_346}', f'{dataset_prefix}/{save_data_name_347}', f'{dataset_prefix}/{save_data_name_348}', f'{dataset_prefix}/{save_data_name_349}', f'{dataset_prefix}/{save_data_name_350}', f'{dataset_prefix}/{save_data_name_351}', f'{dataset_prefix}/{save_data_name_352}', f'{dataset_prefix}/{save_data_name_353}', f'{dataset_prefix}/{save_data_name_354}', f'{dataset_prefix}/{save_data_name_355}', f'{dataset_prefix}/{save_data_name_356}', f'{dataset_prefix}/{save_data_name_357}', f'{dataset_prefix}/{save_data_name_358}', f'{dataset_prefix}/{save_data_name_359}', f'{dataset_prefix}/{save_data_name_360}', f'{dataset_prefix}/{save_data_name_361}', f'{dataset_prefix}/{save_data_name_362}', f'{dataset_prefix}/{save_data_name_363}', f'{dataset_prefix}/{save_data_name_364}', f'{dataset_prefix}/{save_data_name_365}', f'{dataset_prefix}/{save_data_name_366}', f'{dataset_prefix}/{save_data_name_367}', f'{dataset_prefix}/{save_data_name_368}', f'{dataset_prefix}/{save_data_name_369}', f'{dataset_prefix}/{save_data_name_370}', f'{dataset_prefix}/{save_data_name_371}', f'{dataset_prefix}/{save_data_name_372}', f'{dataset_prefix}/{save_data_name_373}', f'{dataset_prefix}/{save_data_name_374}', f'{dataset_prefix}/{save_data_name_375}', f'{dataset_prefix}/{save_data_name_376}', f'{dataset_prefix}/{save_data_name_377}', f'{dataset_prefix}/{save_data_name_378}', f'{dataset_prefix}/{save_data_name_379}', f'{dataset_prefix}/{save_data_name_380}', f'{dataset_prefix}/{save_data_name_381}', f'{dataset_prefix}/{save_data_name_382}', f'{dataset_prefix}/{save_data_name_383}', f'{dataset_prefix}/{save_data_name_384}', f'{dataset_prefix}/{save_data_name_385}', f'{dataset_prefix}/{save_data_name_386}', f'{dataset_prefix}/{save_data_name_387}', f'{dataset_prefix}/{save_data_name_388}', f'{dataset_prefix}/{save_data_name_389}', f'{dataset_prefix}/{save_data_name_390}', f'{dataset_prefix}/{save_data_name_391}', f'{dataset_prefix}/{save_data_name_392}', f'{dataset_prefix}/{save_data_name_393}', f'{dataset_prefix}/{save_data_name_394}', f'{dataset_prefix}/{save_data_name_395}', f'{dataset_prefix}/{save_data_name_396}', f'{dataset_prefix}/{save_data_name_397}', f'{dataset_prefix}/{save_data_name_398}', f'{dataset_prefix}/{save_data_name_399}', f'{dataset_prefix}/{save_data_name_400}', f'{dataset_prefix}/{save_data_name_401}', f'{dataset_prefix}/{save_data_name_402}', f'{dataset_prefix}/{save_data_name_403}', f'{dataset_prefix}/{save_data_name_404}', f'{dataset_prefix}/{save_data_name_405}', f'{dataset_prefix}/{save_data_name_406}', f'{dataset_prefix}/{save_data_name_407}', f'{dataset_prefix}/{save_data_name_408}', f'{dataset_prefix}/{save_data_name_409}', f'{dataset_prefix}/{save_data_name_410}', f'{dataset_prefix}/{save_data_name_411}', f'{dataset_prefix}/{save_data_name_412}', f'{dataset_prefix}/{save_data_name_413}', f'{dataset_prefix}/{save_data_name_414}', f'{dataset_prefix}/{save_data_name_415}', f'{dataset_prefix}/{save_data_name_416}', f'{dataset_prefix}/{save_data_name_417}', f'{dataset_prefix}/{save_data_name_418}', f'{dataset_prefix}/{save_data_name_419}', f'{dataset_prefix}/{save_data_name_420}', f'{dataset_prefix}/{save_data_name_421}', f'{dataset_prefix}/{save_data_name_422}', f'{dataset_prefix}/{save_data_name_423}', f'{dataset_prefix}/{save_data_name_424}', f'{dataset_prefix}/{save_data_name_425}', f'{dataset_prefix}/{save_data_name_426}', f'{dataset_prefix}/{save_data_name_427}', f'{dataset_prefix}/{save_data_name_428}', f'{dataset_prefix}/{save_data_name_429}', f'{dataset_prefix}/{save_data_name_430}', f'{dataset_prefix}/{save_data_name_431}', f'{dataset_prefix}/{save_data_name_432}', f'{dataset_prefix}/{save_data_name_433}', f'{dataset_prefix}/{save_data_name_434}', f'{dataset_prefix}/{save_data_name_435}', f'{dataset_prefix}/{save_data_name_436}', f'{dataset_prefix}/{save_data_name_437}', f'{dataset_prefix}/{save_data_name_438}', f'{dataset_prefix}/{save_data_name_439}', f'{dataset_prefix}/{save_data_name_440}', f'{dataset_prefix}/{save_data_name_441}', f'{dataset_prefix}/{save_data_name_442}', f'{dataset_prefix}/{save_data_name_443}', f'{dataset_prefix}/{save_data_name_444}', f'{dataset_prefix}/{save_data_name_445}', f'{dataset_prefix}/{save_data_name_446}', f'{dataset_prefix}/{save_data_name_447}', f'{dataset_prefix}/{save_data_name_448}', f'{dataset_prefix}/{save_data_name_449}', f'{dataset_prefix}/{save_data_name_450}', f'{dataset_prefix}/{save_data_name_451}', f'{dataset_prefix}/{save_data_name_452}', f'{dataset_prefix}/{save_data_name_453}', f'{dataset_prefix}/{save_data_name_454}', f'{dataset_prefix}/{save_data_name_455}', f'{dataset_prefix}/{save_data_name_456}', f'{dataset_prefix}/{save_data_name_457}', f'{dataset_prefix}/{save_data_name_458}', f'{dataset_prefix}/{save_data_name_459}', f'{dataset_prefix}/{save_data_name_460}', f'{dataset_prefix}/{save_data_name_461}', f'{dataset_prefix}/{save_data_name_462}',
            f'{dataset_prefix}/{save_data_name_463}',f'{dataset_prefix}/{save_data_name_464}',f'{dataset_prefix}/{save_data_name_465}',f'{dataset_prefix}/{save_data_name_466}',f'{dataset_prefix}/{save_data_name_467}',f'{dataset_prefix}/{save_data_name_468}',f'{dataset_prefix}/{save_data_name_469}',f'{dataset_prefix}/{save_data_name_470}',f'{dataset_prefix}/{save_data_name_471}',f'{dataset_prefix}/{save_data_name_472}',f'{dataset_prefix}/{save_data_name_473}',f'{dataset_prefix}/{save_data_name_474}',f'{dataset_prefix}/{save_data_name_475}',f'{dataset_prefix}/{save_data_name_476}',f'{dataset_prefix}/{save_data_name_477}',f'{dataset_prefix}/{save_data_name_478}',f'{dataset_prefix}/{save_data_name_479}',f'{dataset_prefix}/{save_data_name_480}',f'{dataset_prefix}/{save_data_name_481}',f'{dataset_prefix}/{save_data_name_482}',f'{dataset_prefix}/{save_data_name_483}',f'{dataset_prefix}/{save_data_name_484}',f'{dataset_prefix}/{save_data_name_485}',f'{dataset_prefix}/{save_data_name_486}',f'{dataset_prefix}/{save_data_name_487}',f'{dataset_prefix}/{save_data_name_488}',f'{dataset_prefix}/{save_data_name_489}',f'{dataset_prefix}/{save_data_name_490}',f'{dataset_prefix}/{save_data_name_491}',f'{dataset_prefix}/{save_data_name_492}',f'{dataset_prefix}/{save_data_name_493}',f'{dataset_prefix}/{save_data_name_494}',f'{dataset_prefix}/{save_data_name_495}',f'{dataset_prefix}/{save_data_name_496}',f'{dataset_prefix}/{save_data_name_497}',f'{dataset_prefix}/{save_data_name_498}',f'{dataset_prefix}/{save_data_name_499}',f'{dataset_prefix}/{save_data_name_500}',f'{dataset_prefix}/{save_data_name_501}',f'{dataset_prefix}/{save_data_name_502}',f'{dataset_prefix}/{save_data_name_503}',f'{dataset_prefix}/{save_data_name_504}',f'{dataset_prefix}/{save_data_name_505}',f'{dataset_prefix}/{save_data_name_506}',f'{dataset_prefix}/{save_data_name_507}',f'{dataset_prefix}/{save_data_name_508}',f'{dataset_prefix}/{save_data_name_509}',f'{dataset_prefix}/{save_data_name_510}',f'{dataset_prefix}/{save_data_name_511}',f'{dataset_prefix}/{save_data_name_512}',f'{dataset_prefix}/{save_data_name_513}',f'{dataset_prefix}/{save_data_name_514}',f'{dataset_prefix}/{save_data_name_515}',f'{dataset_prefix}/{save_data_name_516}',f'{dataset_prefix}/{save_data_name_517}',f'{dataset_prefix}/{save_data_name_518}',f'{dataset_prefix}/{save_data_name_519}',f'{dataset_prefix}/{save_data_name_520}',f'{dataset_prefix}/{save_data_name_521}',f'{dataset_prefix}/{save_data_name_522}',f'{dataset_prefix}/{save_data_name_523}',f'{dataset_prefix}/{save_data_name_524}',f'{dataset_prefix}/{save_data_name_525}',f'{dataset_prefix}/{save_data_name_526}',f'{dataset_prefix}/{save_data_name_527}',f'{dataset_prefix}/{save_data_name_528}',f'{dataset_prefix}/{save_data_name_529}',f'{dataset_prefix}/{save_data_name_530}',f'{dataset_prefix}/{save_data_name_531}',f'{dataset_prefix}/{save_data_name_532}',f'{dataset_prefix}/{save_data_name_533}',f'{dataset_prefix}/{save_data_name_534}',f'{dataset_prefix}/{save_data_name_535}',f'{dataset_prefix}/{save_data_name_536}',f'{dataset_prefix}/{save_data_name_537}',f'{dataset_prefix}/{save_data_name_538}',f'{dataset_prefix}/{save_data_name_539}',f'{dataset_prefix}/{save_data_name_540}',f'{dataset_prefix}/{save_data_name_541}',f'{dataset_prefix}/{save_data_name_542}',f'{dataset_prefix}/{save_data_name_543}',f'{dataset_prefix}/{save_data_name_544}',f'{dataset_prefix}/{save_data_name_545}',f'{dataset_prefix}/{save_data_name_546}',f'{dataset_prefix}/{save_data_name_547}',f'{dataset_prefix}/{save_data_name_548}',f'{dataset_prefix}/{save_data_name_549}',f'{dataset_prefix}/{save_data_name_550}',f'{dataset_prefix}/{save_data_name_551}',f'{dataset_prefix}/{save_data_name_552}',f'{dataset_prefix}/{save_data_name_553}',f'{dataset_prefix}/{save_data_name_554}',f'{dataset_prefix}/{save_data_name_555}',f'{dataset_prefix}/{save_data_name_556}',f'{dataset_prefix}/{save_data_name_557}',f'{dataset_prefix}/{save_data_name_558}',f'{dataset_prefix}/{save_data_name_559}',f'{dataset_prefix}/{save_data_name_560}',f'{dataset_prefix}/{save_data_name_561}',f'{dataset_prefix}/{save_data_name_562}',f'{dataset_prefix}/{save_data_name_563}',f'{dataset_prefix}/{save_data_name_564}',f'{dataset_prefix}/{save_data_name_565}',f'{dataset_prefix}/{save_data_name_566}',f'{dataset_prefix}/{save_data_name_567}',f'{dataset_prefix}/{save_data_name_568}',f'{dataset_prefix}/{save_data_name_569}',
            ]
        elif num_train_objects == 'mixed_old_and_real_world_noisy_1119':
            dataset_prefix_1 = '/scratch/yufeiw2/dp3_demo'
            dataset_prefix_2 = '/scratch/yufeiw2/dp3_demo_real_world_noise_pcd'
            
            old_list = [i * 3 for i in range(150)]
            all_old_obj_paths = ["{}/{}".format(dataset_prefix_1, globals()["save_data_name_{}".format(i)]) for i in old_list]
            
            all_new_obj_paths = os.listdir(dataset_prefix_2)
            all_new_obj_paths = sorted(all_new_obj_paths)
            all_new_obj_paths = [os.path.join(dataset_prefix_2, x) for x in all_new_obj_paths]
            
            all_obj_paths = all_old_obj_paths + all_new_obj_paths
            
        elif num_train_objects == 'real_world_noisy_pcd_clean_distorted_goal_all':
            dataset_prefix = '/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal'
            all_obj_paths = os.listdir(dataset_prefix)
            all_obj_paths = sorted(all_obj_paths)
            all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
        
        elif num_train_objects == '500_plus_all_real_world':
            non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]
            real_world_camera_500_paths = os.listdir("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd")
            real_world_camera_500_paths = sorted(real_world_camera_500_paths)
            real_world_camera_500_paths = [os.path.join("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd", x) for x in real_world_camera_500_paths]
            all_obj_paths = non_real_world_camera_500_paths + real_world_camera_500_paths
            # all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
            print(all_obj_paths)
            
        elif num_train_objects == '500_plus_all_real_world_clean_distorted_goal':
            dataset_prefix = "/scratch/yufeiw2/dp3_demo_clean_distorted_goal"
            non_real_world_camera_500_paths = ["{}/{}".format(dataset_prefix, globals()["save_data_name_{}".format(i)]) for i in range(463)]
            real_world_camera_500_paths = os.listdir("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal")
            real_world_camera_500_paths = sorted(real_world_camera_500_paths)
            real_world_camera_500_paths = [os.path.join("/scratch/yufeiw2/dp3_demo_real_world_noise_pcd_clean_distorted_goal", x) for x in real_world_camera_500_paths]
            all_obj_paths = non_real_world_camera_500_paths + real_world_camera_500_paths
            # all_obj_paths = [os.path.join(dataset_prefix, x) for x in all_obj_paths]
            print(all_obj_paths)
        elif num_train_objects == 'square_D0':
            all_obj_paths = ['/scratch/minon/articubot_abs']
        elif num_train_objects == 'square_D0_new':
            all_obj_paths = ['/scratch/minon/articubot_abs_cleaned_4000']
        elif num_train_objects == 'square_D2_abs':
            all_obj_paths = ['/scratch/minon/square_d2_abs_better']
            # all_obj_paths = ['/data/minon/tax3d-conditioned-mimicgen/data/robomimic/datasets/square_d2/square_d2_abs/']
        elif num_train_objects == 'three_piece_assembly_D2_abs':
            all_obj_paths = ['/scratch/minon/three_piece_assembly_d2_abs']
        elif num_train_objects == 'threading_D2_abs':
            all_obj_paths = ['/home/ktsim/data/threading_d2_articubot']
        elif num_train_objects == 'put_money_in_safe':
            all_obj_paths = ['/home/ktsim/Projects/SAM2Act/sam2act/data/put_money_in_safe_articubot']
        else:
            raise ValueError('num_train_objects not supported')
        
    if not predict_two_goals:
        dataset = PointNetDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage, 
                                          is_pickle=True, use_all_data=use_all_data, conditioning_on_demo=conditioning_on_demo,
                                          n_obs_steps=n_obs_steps)
    else:
        dataset = PredictTwoGoalsDatasetFromDisk(all_obj_paths, beg_ratio, end_ratio, eval_episode, only_first_stage, is_pickle=True, 
                                                 use_all_data=use_all_data)
    return dataset