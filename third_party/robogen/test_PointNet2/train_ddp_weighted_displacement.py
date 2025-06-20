import torch
from tqdm import tqdm
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import os
from torch.utils.data import DataLoader
from third_party.robogen.test_PointNet2.dataset_from_disk import get_dataset_from_pickle
import wandb
from termcolor import cprint

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    print("Local rank: ", os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def train(args):
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = torch.device(gpu_id)

    input_channel = 5 if args.add_one_hot_encoding else 3

    if not args.predict_two_goals: output_dim = 13 
    else: output_dim = 25

    if args.model_invariant:
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_small2
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_super
        from third_party.robogen.test_PointNet2.model_invariant import PointNet2_superplus
        if args.model_type == 'pointnet2':
            model = PointNet2_small2(num_classes=output_dim).to(device)
        elif args.model_type == 'pointnet2_large':
            model = PointNet2(num_classes=output_dim).to(device)
        elif args.model_type == 'pointnet2_super':
            model = PointNet2_super(num_classes=output_dim, keep_gripper_in_fps=args.keep_gripper_in_fps, input_channel=input_channel, use_in=args.use_instance_norm).to(device)
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
    
    if args.load_model_path is not None:
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        print("Successfully load model from: ", args.load_model_path)
    
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

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
        
    output_dir += args.exp_name
    
    args.exp_path = os.path.join(args.exp_path, output_dir)


    gpu_id = int(os.environ["LOCAL_RANK"])
    model = DDP(model, device_ids=[gpu_id])

    if os.environ['LOCAL_RANK'] == '0':
        if not os.path.exists(args.exp_path):
            os.makedirs(args.exp_path)
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
    dataset = get_dataset_from_pickle(all_obj_paths=args.all_zarr_path, beg_ratio=args.beg_ratio,
                                      end_ratio=args.end_ratio, only_first_stage=args.only_first_stage,
                                      use_all_data=args.use_all_data, use_combined_action=args.use_combined_action, 
                                      dataset_prefix=args.dataset_prefix, num_train_objects=args.num_train_objects,
                                      predict_two_goals=args.predict_two_goals, n_obs_steps=args.n_obs_steps)
    dataloader = DataLoader(dataset, 
                shuffle=False,
                sampler=DistributedSampler(dataset),
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True,
                )

    global_step = 0

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        accumulated_displacement_loss = 0.0
        accumulated_weighting_loss = 0.0
        for i, data in enumerate(tqdm(dataloader)):
            if args.n_obs_steps > 1:
                pointcloud, gripper_pcd, goal_gripper_pcd, gripper_pcd_history = data
            else:
                pointcloud, gripper_pcd, goal_gripper_pcd = data

            # inputs: B, N, 3
            # gripper_pcd: B, 4, 3
            # goal_gripper_points: B, 4, 3
            # gripper_pcd_history: B, H, 4, 3
            # calculate the displacement from every point to the gripper to get the labels with shape B, N, 4, 3
            gripper_points = goal_gripper_pcd
            
            if not args.predict_two_goals:
                if args.add_one_hot_encoding:
                    # for pointcloud, we add (1, 0)
                    # for gripper_pcd, we add (0, 1)
                    pointcloud_one_hot = torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2)
                    pointcloud_one_hot[:, :, 0] = 1
                    pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
                    gripper_pcd_one_hot = torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2)
                    gripper_pcd_one_hot[:, :, 1] = 1
                    gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
                    inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1) # B, N+4, 5
                else:
                    inputs = torch.cat([pointcloud, gripper_pcd], dim=1) # B, N+4, 3
                    if args.n_obs_steps > 1:
                        B, H, _, _, = gripper_pcd_history.shape
                        gripper_pcd_history = gripper_pcd_history.reshape(B, -1, 3)
                        inputs = torch.cat([inputs, gripper_pcd_history], dim=1) # B, N+4+4*history, 3
            else:
                inputs = pointcloud

            labels = gripper_points.unsqueeze(1) - inputs[:, :, :3].unsqueeze(2)
            B, N, _, _ = labels.shape
            labels = labels.view(B, N, -1) # B, N, 12

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)
            optimizer.zero_grad()
            outputs = model(inputs) # B, N, 13
            weights = outputs[:, :, -1] # B, N
            outputs = outputs[:, :, :-1] # B, N, 12
            if args.output_obj_pcd_only:
                weights = weights[:, :-4]
                outputs = outputs[:, :-4, :]
                labels = labels[:, :-4, :]
                inputs = inputs[:, :, :-4]
                N = N - 4
            loss = criterion(outputs, labels)
            accumulated_displacement_loss += loss.item()

            if args.using_weight:
                inputs = inputs.permute(0, 2, 1)
                if not args.predict_two_goals:
                    outputs = outputs.view(B, N, 4, 3)
                else:
                    outputs = outputs.view(B, N, 8, 3)
                outputs = outputs + inputs[:, :, :3].unsqueeze(2) # B, N, 4, 3

                # softmax the weights
                weights = torch.nn.functional.softmax(weights, dim=1)
                
                # sum the displacement of the predicted gripper point cloud according to the weights
                outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
                outputs = outputs.sum(dim=1)
                avg_loss = criterion(outputs, gripper_points.to(device))

                loss = loss + avg_loss * args.weight_loss_weight
                accumulated_weighting_loss += (avg_loss * args.weight_loss_weight).item()

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
                }

                wandb_run.log(log_info, step=global_step)

                running_loss = 0.0
                accumulated_displacement_loss = 0.0
                accumulated_weighting_loss = 0.0

            global_step += 1

        if (epoch + 1) % args.save_freq == 0 and os.environ['LOCAL_RANK'] == '0':
            save_path = f"{args.exp_path}/model_{epoch + 1}.pth"
            torch.save(model.module.state_dict(), save_path)

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ddp_setup()
    train(args)
    destroy_process_group()