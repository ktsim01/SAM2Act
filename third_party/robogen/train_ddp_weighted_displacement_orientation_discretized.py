import torch
torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import os
from torch.utils.data import DataLoader
from third_party.robogen.test_PointNet2.dataset_from_disk import get_dataset_from_pickle, get_train_and_val_dataset_from_pickle
import wandb
from termcolor import cprint
import numpy as np

import sys
sys.path.append('..')

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    print("Local rank: ", os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def add_gaussian_noise_torch(points, noise_magnitude=0.01):
    device = points.device
    noise = torch.randn_like(points) * noise_magnitude
    return points + noise

def angle_to_bin(angle_deg, num_bins=72, range_min=-180.0, range_max=180.0):
    """
    Map an angle in degrees to a bin index [0, num_bins-1].
    Assumes angle is in [range_min, range_max).
    Works with scalars or numpy arrays.
    """
    # Normalize to [0, 1)
    normed = (angle_deg - range_min) / (range_max - range_min)
    bin_idx = (normed * num_bins).long()

    # Clamp just in case (avoid index num_bins)
    return np.clip(bin_idx, 0, num_bins - 1)

def circular_bin_error(pred_idx, gt_idx, num_bins=72):
    diff = torch.abs(pred_idx - gt_idx)
    return torch.min(diff, num_bins - diff)


def apply_random_se3(pcd, max_translation=0.01, max_rotation_deg=10):
    """
    Applies a small random SE(3) transform to a point cloud.
    Args:
        pcd (torch.Tensor): (B, N, 3)
        max_translation (float): max translation in meters
        max_rotation_deg (float): max rotation in degrees
    Returns:
        transformed_pcd (torch.Tensor): (B, N, 3)
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # Random translations
    translations = (torch.rand(B, 3, device=device) - 0.5) * 2 * max_translation  # (B, 3)

    # Random rotations
    angles = (torch.rand(B, 3, device=device) - 0.5) * 2 * (max_rotation_deg / 180.0 * np.pi)  # (B, 3)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Create rotation matrices for X, Y, Z axis
    ones = torch.ones_like(cos[:, 0])
    zeros = torch.zeros_like(cos[:, 0])

    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, cos[:, 0], -sin[:, 0]], dim=1),
        torch.stack([zeros, sin[:, 0], cos[:, 0]], dim=1)
    ], dim=2)  # (B, 3, 3)

    Ry = torch.stack([
        torch.stack([cos[:, 1], zeros, sin[:, 1]], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-sin[:, 1], zeros, cos[:, 1]], dim=1)
    ], dim=2)

    Rz = torch.stack([
        torch.stack([cos[:, 2], -sin[:, 2], zeros], dim=1),
        torch.stack([sin[:, 2], cos[:, 2], zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=2)

    R = Rz @ Ry @ Rx  # Compose rotation matrices (B, 3, 3)

    # Apply SE(3)
    pcd_rotated = torch.bmm(pcd, R.transpose(1, 2))  # (B, N, 3)
    pcd_transformed = pcd_rotated + translations.unsqueeze(1)  # (B, N, 3)

    return pcd_transformed


def apply_random_so2_y(pcd, gripper_pcd, goal_gripper_pcd, max_translation=0.01, max_rotation_deg=20):
    """
    Apply random SO(2) rotation around the Y-axis and small translation.
    
    Args:
        pcd (torch.Tensor): (B, N, 3) scene pointcloud
        max_translation (float): max translation in meters
        max_rotation_deg (float): max rotation angle in degrees
    Returns:
        transformed_pcd (torch.Tensor): (B, N, 3)
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # Random rotation angles θ around Y axis
    theta = (torch.rand(B, device=device) - 0.5) * 2 * (max_rotation_deg / 180.0 * np.pi)  # (B,)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Construct rotation matrix (Y-axis)
    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0] = cos_theta
    R[:, 0, 2] = sin_theta
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -sin_theta
    R[:, 2, 2] = cos_theta

    # Apply rotation
    rotated_pcd = torch.bmm(pcd, R.transpose(1, 2))  # (B, N, 3)
    rotated_gripper_pcd = torch.bmm(gripper_pcd, R.transpose(1, 2))  # (B, 4, 3)
    rotated_goal_gripper_pcd = torch.bmm(goal_gripper_pcd, R.transpose(1, 2))  # (B, 4, 3)

    # Apply small random translation
    translations = (torch.rand(B, 3, device=device) - 0.5) * 2 * max_translation  # (B, 3)
    translated_pcd = rotated_pcd + translations.unsqueeze(1)
    translated_gripper_pcd = rotated_gripper_pcd + translations.unsqueeze(1)  # (B, 4, 3)
    translated_goal_gripper_pcd = rotated_goal_gripper_pcd + translations.unsqueeze(1)  # (B, 4, 3)

    return translated_pcd, translated_gripper_pcd, translated_goal_gripper_pcd

def apply_random_so2_z(pcd, gripper_pcd, goal_gripper_pcd, max_translation=0.01, max_rotation_deg=20):
    """
    Apply random SO(2) rotation around the Z-axis and small translation.

    Args:
        pcd (torch.Tensor): (B, N, 3)
        max_translation (float): max translation in meters
        max_rotation_deg (float): max rotation in degrees

    Returns:
        torch.Tensor: transformed point cloud of shape (B, N, 3)
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # Random rotation angles θ around Z axis
    theta = (torch.rand(B, device=device) - 0.5) * 2 * (max_rotation_deg / 180.0 * np.pi)  # (B,)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Construct rotation matrices Rz for each batch element
    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0] = cos_theta
    R[:, 0, 1] = -sin_theta
    R[:, 1, 0] = sin_theta
    R[:, 1, 1] = cos_theta
    R[:, 2, 2] = 1.0

    # Apply rotation
    rotated = torch.bmm(pcd, R.transpose(1, 2))  # (B, N, 3)
    rotated_gripper = torch.bmm(gripper_pcd, R.transpose(1, 2))  # (B, 4, 3)
    rotated_goal_gripper = torch.bmm(goal_gripper_pcd, R.transpose(1, 2))  # (B, 4, 3)

    # Apply small random translation
    translation = (torch.rand(B, 3, device=device) - 0.5) * 2 * max_translation  # (B, 3)
    translated = rotated + translation.unsqueeze(1)
    translated_gripper = rotated_gripper + translation.unsqueeze(1)  # (B, 4, 3)
    translated_goal_gripper = rotated_goal_gripper + translation.unsqueeze(1)  # (B, 4, 3)

    return translated, translated_gripper, translated_goal_gripper

import glob
import re
def find_latest_checkpoint(directory):
    # Pattern: model_<number>.pth
    checkpoint_files = glob.glob(os.path.join(directory, "model_*.pth"))
    
    if not checkpoint_files:
        return None, 0  # No checkpoints

    # Extract epoch number and pick the highest
    pattern = re.compile(r"model_(\d+)\.pth$")
    max_epoch = 0
    latest_ckpt = None
    for ckpt in checkpoint_files:
        match = pattern.search(ckpt)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_ckpt = ckpt

    return latest_ckpt, max_epoch


import subprocess

def upload_file(local_folder):
    base = "gs://cmu-gpucloud-ktsim/articubot_exps"
    folder_name = os.path.basename(local_folder.rstrip("/"))
    destination = f"{base}/{folder_name}"
    
    try:
        cmd = ["gcloud", "storage", "rsync", "-r", local_folder, destination]
        # print(cmd)
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[Success] Uploaded: {local_folder} -> {destination}")
    except subprocess.CalledProcessError as e:
        print(f"[Failure] Failed to upload {local_folder}: {e.stderr.strip()}")

import os
import re
import tempfile
from google.cloud import storage
import subprocess

def load_checkpoint(model, device, load_model_path):
    if load_model_path is None:
        return model, 0

    # If the path points to GCS
    if load_model_path.startswith("gs://"):
        match = re.search(r"model_(\d+)\.pth$", load_model_path)
        loaded_epoch = int(match.group(1)) if match else 0
        latest_ckpt = load_model_path

        print(f"Downloading model from GCS: {latest_ckpt}")
        temp_path = download_gcs_blob(latest_ckpt)
        model.load_state_dict(torch.load(temp_path, map_location=device))
        print(f"Successfully loaded model from GCS (epoch {loaded_epoch})")

    else:
        # Local file or directory
        if os.path.isdir(load_model_path):
            latest_ckpt, loaded_epoch = find_latest_checkpoint(load_model_path)
        else:
            match = re.search(r"model_(\d+)\.pth$", load_model_path)
            loaded_epoch = int(match.group(1)) if match else 0
            latest_ckpt = load_model_path

        print(f"Loading model from epoch {loaded_epoch}...")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        print("Successfully loaded local model from:", latest_ckpt)

    return model, loaded_epoch


# ---------------- Helper functions ---------------- #

def parse_gcs_path(gcs_path):
    assert gcs_path.startswith("gs://")
    path = gcs_path[5:]
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix

def find_latest_checkpoint_gcs(bucket_name, prefix=""):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    pattern = re.compile(r"model_(\d+)\.pth$")
    max_epoch = 0
    latest_ckpt = None

    for blob in blobs:
        match = pattern.search(blob.name)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_ckpt = f"gs://{bucket_name}/{blob.name}"

    return latest_ckpt, max_epoch

def download_gcs_blob(gcs_path):
    """Downloads a GCS blob to a temporary local file and returns the local path."""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    _, temp_path = tempfile.mkstemp(suffix=".pth")
    blob.download_to_filename(temp_path)
    return temp_path

def train(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(gpu_id)
    input_channel = 3
    if args.use_color:
        input_channel += 3
    if args.add_one_hot_encoding:
        input_channel += 2

    output_dim = 36 # 3 + 1 + 32: position, weight, features

    if args.model_invariant:
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_small2
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_super
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_superplus
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_Binary
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_text
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_textV2
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_text_10k
        from third_party.robogen.test_PointNet2.model_invariant import GripperOrientNet

        if args.model_type == 'pointnet2':
            model = PointNet2_small2(num_classes=output_dim).to(device)
        elif args.model_type == 'pointnet2_large':
            model = PointNet2(num_classes=output_dim).to(device)
        elif args.model_type == 'pointnet2_super':
            model = PointNet2_text(num_classes=output_dim, input_channel=input_channel,  keep_gripper_in_fps=args.keep_gripper_in_fps, use_text_embedding=True).to(device)
        elif args.model_type == 'pointnet2_textV2':
            model = PointNet2_textV2(num_classes=output_dim, input_channel=input_channel,  keep_gripper_in_fps=args.keep_gripper_in_fps, use_text_embedding=True).to(device)
        elif args.model_type == 'pointnet2_text_10k':
            if args.use_text:
                model = PointNet2_text_10k(num_classes=output_dim, input_channel=input_channel, use_text_embedding=True).to(device)
            else:
                model = PointNet2_super(num_classes=output_dim, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel, use_in=args.use_instance_norm).to(device)
        elif args.model_type == 'GripperOrientNet':
            model = GripperOrientNet(num_classes=output_dim, input_channel=input_channel, keep_gripper_in_fps=args.keep_gripper_in_fps, use_text_embedding=True).to(device)
        elif args.model_type == 'pointnet2_binary':
            model = PointNet2_Binary(num_classes=output_dim, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel, use_in=args.use_instance_norm).to(device)
        elif args.model_type == 'attn':
            model = AttnModel(num_classes=output_dim).to(device)
        elif args.model_type == 'pointnet2_superplus':
            model = PointNet2_superplus(num_classes=output_dim).to(device)
        else:
            raise ValueError(f"model_type {args.model_type} not recognized")
    else:
        from third_party.robogen.test_PointNet2.model import PointNet2_small2
        from third_party.robogen.test_PointNet2.model import PointNet2
        from third_party.robogen.test_PointNet2.model import PointNet2_super
        if args.model_type == 'pointnet2':
            model = PointNet2_small2(num_classes=output_dim).to(device)
        elif args.model_type == 'pointnet2_large':
            model = PointNet2(num_classes=output_dim).to(device)
        elif args.model_type == 'pointnet2_super':
            model = PointNet2_super(num_classes=output_dim).to(device)
        elif args.model_type == 'attn':
            model = AttnModel(num_classes=output_dim).to(device)
        else:
            raise ValueError(f"model_type {args.model_type} not recognized")
    
    loaded_epoch = None
    if args.load_model_path is not None:
        model, loaded_epoch = load_checkpoint(model, device, args.load_model_path)

    criterion = torch.nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    # dataloader = get_dataloader(all_obj_paths=args.all_zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True, only_first_stage=args.only_first_stage)
    # dataloader = get_dataloader_from_pickle(all_obj_paths=args.all_zarr_path, batch_size=args.batch_size, beg_ratio=args.beg_ratio, end_ratio=args.end_ratio, shuffle=True, only_first_stage=args.only_first_stage)
    
    output_dir = args.model_type 

    if args.model_invariant:
        output_dir = output_dir + "_model_invariant"
    
    output_dir = output_dir + "_" + str(datetime.date.today())

    if args.use_all_data:
        output_dir = output_dir + "_use_all_data"
    else:
        output_dir = output_dir + "_use_75_episodes"

    if args.use_combined_action:
        output_dir = output_dir + "_use_combined_data"
    
    output_dir = output_dir + "_" + str(args.num_train_objects) + "-obj"
    
    if args.predict_two_goals:
        output_dir = output_dir + "_predict_two_goals"
        
    if args.output_obj_pcd_only:
        output_dir = output_dir + "_output_obj_only"
        
    if args.only_first_stage:
        output_dir = output_dir + "_only_first_stage"
        
    if args.keep_gripper_in_fps:
        output_dir = output_dir + "_keep_gripper_in_fps"
        
    if args.add_one_hot_encoding:
        output_dir = output_dir + "_one_hot"
    
    if not args.using_weight:
        output_dir = output_dir + "_no_weight"

    if args.use_gripper_open:
        output_dir = output_dir + "_use_gripper_open"

    if args.use_collision:
        output_dir = output_dir + "_use_collision"

    if args.use_color:
        output_dir = output_dir + "_use_color"

    if args.use_text:
        output_dir = output_dir + "_use_text"

    if args.gmm:
        output_dir = output_dir + "_gmm"
    
    if args.so2:
        output_dir = output_dir + "_so2"

    output_dir += "_" + args.exp_name
    
    args.exp_path = os.path.join(args.exp_path, output_dir)

    latest_ckpt, latest_epoch = find_latest_checkpoint_gcs("cmu-gpucloud-ktsim", "articubot_exps/" + output_dir)
        
    if latest_ckpt is not None:
        print(f"Found latest checkpoint: {latest_ckpt}, epoch: {latest_epoch}")
        temp_path = download_gcs_blob(latest_ckpt)
        model.load_state_dict(torch.load(temp_path, map_location=device))
        print("Successfully loaded model from: ", latest_ckpt)
    elif loaded_epoch is not None:
        latest_epoch = loaded_epoch

    # optimizer_seg = torch.optim.Adam(
    #     list(model.sa1.parameters()) +
    #     list(model.sa2.parameters()) +
    #     list(model.sa3.parameters()) +
    #     list(model.sa4.parameters()) +
    #     list(model.sa5.parameters()) +
    #     list(model.sa6.parameters()) +
    #     list(model.fp1.parameters()) +
    #     list(model.fp2.parameters()) +
    #     list(model.fp3.parameters()) +
    #     list(model.fp4.parameters()) +
    #     list(model.fp5.parameters()) +
    #     list(model.fp6.parameters()) +
    #     list(model.conv1.parameters()) +
    #     list(model.bn1.parameters()) +
    #     list(model.conv2.parameters()),
    #     lr=args.lr
    # )

    # # Orientation optimizer (independent head)
    # optimizer_orient = torch.optim.Adam(
    #     list(model.fc.parameters()) +
    #     list(model.roll.parameters()) +
    #     list(model.pitch.parameters()) +
    #     list(model.yaw.parameters()),
    #     lr=getattr(args, "lr_orient", args.lr)  # fallback to seg lr if not specified
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    gpu_id = int(os.environ["LOCAL_RANK"])
    model = DDP(model, device_ids=[gpu_id])

    if os.environ['LOCAL_RANK'] == '0':
        if not os.path.exists(args.exp_path):
            os.makedirs(args.exp_path)

        if args.wandb:
            wandb_run = wandb.init(
                    project=f"pointnet-weighted-displacement_{args.num_train_objects}",
                    name=str(output_dir),
                    dir=str(args.exp_path),
                )
            wandb.config.update(
                {
                    "output_dir": args.exp_path,
                    "model_type": args.model_type,
                    "lr": args.lr,
                    "weight_loss_weight": args.weight_loss_weight,
                    "batch_size": args.batch_size
                }
            )
            
            config_dict = args.__dict__
            wandb.config.update(config_dict)

            # save the config file
            with open(os.path.join(args.exp_path, "config.txt"), "w") as f:
                for key, value in config_dict.items():
                    f.write(f"{key}: {value}\n")

    print("trying to load dataset")
    # train_dataset, val_dataset = get_train_and_val_dataset_from_pickle(all_obj_paths=args.all_zarr_path, beg_ratio=args.beg_ratio,
    #                                   end_ratio=args.end_ratio, only_first_stage=args.only_first_stage,
    #                                   use_all_data=args.use_all_data, use_combined_action=args.use_combined_action, 
    #                                   dataset_prefix=args.dataset_prefix, num_train_objects=args.num_train_objects,
    #                                   predict_two_goals=args.predict_two_goals, n_obs_steps=args.n_obs_steps,
    #                                   use_color=args.use_color)
    # train_dataloader = DataLoader(train_dataset, 
    #             shuffle=False,
    #             sampler=DistributedSampler(train_dataset),
    #             batch_size=args.batch_size,
    #             num_workers=4,
    #             pin_memory=True,
    #             )

    # val_dataloader = DataLoader(val_dataset, 
    #             shuffle=False,
    #             sampler=DistributedSampler(val_dataset),
    #             batch_size=args.batch_size,
    #             num_workers=4,
    #             pin_memory=True,
    #             )
    train_dataset = get_dataset_from_pickle(all_obj_paths=args.all_zarr_path, beg_ratio=args.beg_ratio,
                                      end_ratio=args.end_ratio, only_first_stage=args.only_first_stage,
                                      use_all_data=args.use_all_data, use_combined_action=args.use_combined_action, 
                                      dataset_prefix=args.dataset_prefix, num_train_objects=args.num_train_objects,
                                      predict_two_goals=args.predict_two_goals, n_obs_steps=args.n_obs_steps, val=False, orientation_prediction=True)
    train_dataloader = DataLoader(train_dataset, 
                shuffle=False,
                sampler=DistributedSampler(train_dataset),
                batch_size=args.batch_size,
                num_workers=2,
                pin_memory=True,
                )
    
    val_dataset = get_dataset_from_pickle(all_obj_paths=args.all_zarr_path, beg_ratio=args.beg_ratio,
                                      end_ratio=args.end_ratio, only_first_stage=args.only_first_stage,
                                      use_all_data=args.use_all_data, use_combined_action=args.use_combined_action, 
                                      dataset_prefix=args.dataset_prefix, num_train_objects=args.num_train_objects,
                                      predict_two_goals=args.predict_two_goals, n_obs_steps=args.n_obs_steps, val=True, orientation_prediction=True)
    val_dataloader = DataLoader(val_dataset, 
                shuffle=False,
                sampler=DistributedSampler(val_dataset),
                batch_size=args.batch_size,
                num_workers=2,
                pin_memory=True,
                )
    
    global_step = 0
    min_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        if epoch < latest_epoch:
            print(f"Skipping epoch {epoch + 1} as it is less than the latest epoch {latest_epoch}")
            continue

        running_loss = 0.0
        accumulated_displacement_loss = 0.0
        accumulated_weighting_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):
            if args.n_obs_steps > 1:
                pointcloud, gripper_pcd, goal_gripper_pcd, gripper_pcd_history = data
            else:
                pointcloud, gripper_pos, gripper_rot, goal_gripper_pos, goal_gripper_rot, lang_feats = data

            # inputs: B, N, 3
            # gripper_pos: B, 3
            # gripper_rot: B, 3
            # gripper_pcd_history: B, H, 4, 3
            # calculate the displacement from every point to the gripper to get the labels with shape B, N, 4, 3
            gripper_pos = gripper_pos.unsqueeze(1)

            ### SO2 Augmentation
            if args.so2:
                pointcloud_transformed, gripper_pcd_transformed, goal_gripper_pcd_transformed = apply_random_so2_z(pointcloud[..., :3], gripper_pcd, goal_gripper_pcd, max_translation=0.1, max_rotation_deg=45)

                # pointcloud_transformed = add_gaussian_noise_torch(pointcloud_transformed)
                pointcloud[..., :3] = pointcloud_transformed
                gripper_pcd = gripper_pcd_transformed
                goal_gripper_pcd = goal_gripper_pcd_transformed


            # gripper_pcd = add_gaussian_noise_torch(gripper_pcd)

            if args.use_color:
                gripper_pos = torch.cat([gripper_pos, torch.ones(gripper_pos.shape)], dim=2)
            else:
                pointcloud = pointcloud[..., :3]


            if not args.predict_two_goals:
                if args.add_one_hot_encoding:
                    # for pointcloud, we add (1, 0)
                    # for gripper_pcd, we add (0, 1)
                    pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2)
                    pointcloud_one_hot[:, :, 0] = 1
                    pointcloud = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
                    gripper_pos_one_hot = torch.zeros(gripper_pos.shape[0], gripper_pos.shape[1], 2)
                    gripper_pos_one_hot[:, :, 1] = 1
                    gripper_pos = torch.cat([gripper_pos, gripper_pos_one_hot], dim=2)
                    inputs = torch.cat([pointcloud, gripper_pos], dim=1) # B, N+4, 5

                else:
                    inputs = torch.cat([pointcloud, gripper_pcd], dim=1) # B, N+4, 3
                    if args.n_obs_steps > 1:
                        B, H, _, _, = gripper_pcd_history.shape
                        gripper_pcd_history = gripper_pcd_history.reshape(B, -1, 3)
                        inputs = torch.cat([inputs, gripper_pcd_history], dim=1) # B, N+4+4*history, 3
            else:
                inputs = pointcloud

            labels = goal_gripper_pos.unsqueeze(1) - inputs[:, :, :3]
            B, N, _ = labels.shape
            labels = labels.view(B, N, -1) # B, N, 12

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)
            optimizer.zero_grad()
            # optimizer_orient.zero_grad()

            if args.use_text:
                displacement, gripper_pos_prediction, roll, pitch, yaw= model(xyz=inputs, text_embedding=lang_feats) # B, N, 15
            else:
                outputs = model(inputs)

            loss = criterion(displacement, labels)
            accumulated_displacement_loss += loss.item()

            roll_bins  = angle_to_bin(goal_gripper_rot[..., 2], num_bins=72, range_min=-180, range_max=180)
            pitch_bins = angle_to_bin(goal_gripper_rot[..., 1], num_bins=36, range_min=-90, range_max=90)
            yaw_bins   = angle_to_bin(goal_gripper_rot[..., 0], num_bins=72, range_min=-180, range_max=180)

            avg_loss = criterion(gripper_pos_prediction, goal_gripper_pos.to(device))
            loss = loss + avg_loss * args.weight_loss_weight
            accumulated_weighting_loss += (avg_loss * args.weight_loss_weight).item()
            
            orient_loss_roll = ce_loss(roll, roll_bins.long().to(device))
            orient_loss_pitch = ce_loss(pitch, pitch_bins.long().to(device))
            orient_loss_yaw = ce_loss(yaw, yaw_bins.long().to(device))
            orient_loss = (orient_loss_roll + orient_loss_pitch + orient_loss_yaw) / 3

            loss =  loss + orient_loss * args.orientation_loss_weight

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % 10 == 0 and os.environ['LOCAL_RANK'] == '0':
                print(f"Epoch {epoch + 1}, iter {i + 1}, loss: {running_loss / 1000}")
                
                log_info = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "total_loss": running_loss / 1000,
                    "displacement_loss": accumulated_displacement_loss / 1000,
                    "weighting_loss": accumulated_weighting_loss / 1000,
                    "orient_loss_roll": orient_loss_roll.item() / 1000,
                    "orient_loss_pitch": orient_loss_pitch.item() / 1000,
                    "orient_loss_yaw": orient_loss_yaw.item() / 1000,

                }
                if args.wandb:
                    wandb_run.log(log_info, step=global_step)

                running_loss = 0.0
                accumulated_displacement_loss = 0.0
                accumulated_weighting_loss = 0.0
                orient_loss = 0.0


            # log_interval = len(train_dataloader) // 6
            # if (i+1) % log_interval == 0 and os.environ['LOCAL_RANK'] == '0':
            #     print(f"Epoch {epoch + 1}, iter {i + 1}, loss: {running_loss / log_interval}")
                
            #     log_info = {
            #         "epoch": epoch + 1,
            #         "global_step": global_step,
            #         "total_loss": running_loss / log_interval,
            #         "displacement_loss": accumulated_displacement_loss / log_interval,
            #         "weighting_loss": accumulated_weighting_loss / log_interval,
            #     }
            global_step += 1

        
        if (epoch + 1) % 5 == 0:
            accumulated_val_loss = 0.0
            roll_acc = 0.0
            pitch_acc = 0.0
            yaw_acc = 0.0

            model.eval()
            for i, data in enumerate(tqdm(val_dataloader)):
                pointcloud, gripper_pos, gripper_rot, goal_gripper_pos, goal_gripper_rot, lang_feats = data

                if not args.use_color:
                    pointcloud = pointcloud[..., :3]  # Ensure only xyz coordinates are used
                
                gripper_pos = gripper_pos.unsqueeze(1)

                if args.use_color:
                    gripper_pos = torch.cat([gripper_pos, torch.ones(gripper_pos.shape)], dim=2)

                if args.add_one_hot_encoding:
                    pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2)
                    pointcloud_one_hot[:, :, 0] = 1
                    pointcloud = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
                    gripper_pos_one_hot = torch.zeros(gripper_pos.shape[0], gripper_pos.shape[1], 2)
                    gripper_pos_one_hot[:, :, 1] = 1
                    gripper_pos = torch.cat([gripper_pos, gripper_pos_one_hot], dim=2)


                inputs = torch.cat([pointcloud, gripper_pos], dim=1) # B, N+4, 5

                labels = goal_gripper_pos.unsqueeze(1).unsqueeze(1) - inputs[:, :, :3].unsqueeze(2)
                B, N, _, _ = labels.shape
                labels = labels.view(B, N, -1) # B, N, 12

                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.permute(0, 2, 1)
                with torch.no_grad():
                    displacement, gripper_pos_prediction, roll, pitch, yaw = model(inputs, lang_feats) # B, N, 13
                
                roll_pred = torch.argmax(roll, dim=-1)
                pitch_pred = torch.argmax(pitch, dim=-1)
                yaw_pred = torch.argmax(yaw, dim=-1)
                
                roll_gt = angle_to_bin(goal_gripper_rot[..., 2], num_bins=72, range_min=-180, range_max=180).long().to(device)
                pitch_gt = angle_to_bin(goal_gripper_rot[..., 1], num_bins=36, range_min=-90, range_max=90).long().to(device)
                yaw_gt = angle_to_bin(goal_gripper_rot[..., 0], num_bins=72, range_min=-180, range_max=180).long().to(device)

                roll_acc += (circular_bin_error(roll_pred, roll_gt, num_bins=72) <= 1).sum().float()
                pitch_acc += (circular_bin_error(pitch_pred, pitch_gt, num_bins=36) <= 1).sum().float()
                yaw_acc += (circular_bin_error(yaw_pred, yaw_gt, num_bins=72) <= 1).sum().float()

                # roll_acc += (roll_pred == roll_gt).sum().float()
                # pitch_acc += (pitch_pred == pitch_gt).sum().float()
                # yaw_acc += (yaw_pred == yaw_gt).sum().float()

                print("Roll prediction and ground truth", roll_pred, roll_gt)
                print("Pitch prediction and ground truth", pitch_pred, pitch_gt)
                print("Yaw prediction and ground truth", yaw_pred, yaw_gt)

                accumulated_val_loss += criterion(gripper_pos_prediction, goal_gripper_pos.to(device))

            
            torch.distributed.all_reduce(accumulated_val_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(roll_acc, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(pitch_acc, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(yaw_acc, op=torch.distributed.ReduceOp.SUM)

            accumulated_val_loss = accumulated_val_loss.item()
            roll_acc = roll_acc.item()
            pitch_acc = pitch_acc.item()
            yaw_acc = yaw_acc.item()

            if os.environ['LOCAL_RANK'] == '0':
                print(f"Epoch {epoch + 1}, iter {i + 1}, val loss: {accumulated_val_loss / len(val_dataloader.dataset)}")
                print(f"Roll acc: {roll_acc / len(val_dataloader.dataset)}, Pitch acc: {pitch_acc / len(val_dataloader.dataset)}, Yaw acc: {yaw_acc / len(val_dataloader.dataset)}")
                log_info = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "accumulated_val_loss": accumulated_val_loss / len(val_dataloader.dataset),
                    "roll_acc": roll_acc / len(val_dataloader.dataset),
                    "pitch_acc": pitch_acc / len(val_dataloader.dataset),
                    "yaw_acc": yaw_acc / len(val_dataloader.dataset),
                }
                if args.wandb:
                    wandb_run.log(log_info, step=global_step)

                if accumulated_val_loss < min_val_loss:
                    min_val_loss = accumulated_val_loss
                    if os.environ['LOCAL_RANK'] == '0':
                        save_path = f"{args.exp_path}/best_model.pth"
                        torch.save(model.module.state_dict(), save_path)
                        print(f"Saved best model to {save_path}")
                accumulated_val_loss = 0.0
                roll_acc = 0.0
                pitch_acc = 0.0
                yaw_acc = 0.0
            
            model.train()

        if (epoch + 1) % args.save_freq == 0 and os.environ['LOCAL_RANK'] == '0':
            save_path = f"{args.exp_path}/model_{epoch + 1}.pth"
            torch.save(model.module.state_dict(), save_path)
            upload_file(args.exp_path)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_zarr_path', type=str, default=None)
    parser.add_argument('--num_train_objects', default=200)
    parser.add_argument('--dataset_prefix', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--beg_ratio', type=float, default=0)
    parser.add_argument('--end_ratio', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--only_first_stage', action='store_true')
    parser.add_argument('--exp_path', type=str, default="/project_data/held/ziyuw2/Robogen-sim2real/test_PointNet2/exps")
    parser.add_argument('--model_type', type=str, default='pointnet2')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--output_obj_pcd_only', action='store_true')
    parser.add_argument('--weight_loss_weight', type=float, default=10)
    parser.add_argument('--orientation_loss_weight', type=float, default=1)
    parser.add_argument('--use_all_data', action='store_true')
    parser.add_argument('--use_combined_action', action='store_true')
    parser.add_argument('--model_invariant', action='store_true')
    parser.add_argument('--predict_two_goals', action='store_true')
    parser.add_argument('--keep_gripper_in_fps', type=int, default=0)
    parser.add_argument('--add_one_hot_encoding', type=int, default=0)
    parser.add_argument('--using_weight', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--n_obs_steps', type=int, default=1)
    parser.add_argument('--use_instance_norm', action='store_true')
    parser.add_argument('--use_gripper_open', action='store_true')
    parser.add_argument('--use_collision', action='store_true')
    parser.add_argument('--use_color', action='store_true')
    parser.add_argument('--wandb', action='store_true', help="Whether to use wandb for logging")
    parser.add_argument('--use_text', action='store_true', help="Whether to use text input")
    parser.add_argument('--gmm', action='store_true', help="Whether to use GMM loss")
    parser.add_argument('--so2', action='store_true', help="Whether to use SO2 augmentation")
    parser.add_argument('--fixed_variance', type=float, default=0.05)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ddp_setup()
    train(args)
    destroy_process_group()