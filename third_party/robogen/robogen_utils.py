import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from torch.utils.data import DataLoader
from third_party.robogen.test_PointNet2.model_invariant import PointNet2_super
from matplotlib import pyplot as plt
import torch

ROOT_DIR = Path(__file__).parent.parent.parent

original_gripper_pcd = np.array([[ 0.10432111,  0.00228697,  0.8474241 ],
       [ 0.12816067, -0.04368229,  0.8114649 ],
       [ 0.08953098,  0.0484529 ,  0.80711854],
       [ 0.11198021,  0.00245327,  0.7828771 ]])
original_gripper_pos = np.array([0.1119802 , 0.00245327, 0.78287711])
original_gripper_orn = np.array([0.97841681, 0.19802945, 0.0581003 , 0.01045192])

def compute_plane_normal(gripper_pcd):
    x1 = gripper_pcd[0]
    x2 = gripper_pcd[1]
    x4 = gripper_pcd[3]
    v1 = x2 - x1
    v2 = x4 - x1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

original_gripper_normal = compute_plane_normal(original_gripper_pcd)

def quaternion_to_rotation_matrix(quat):
    rotation = R.from_quat(quat)
    return rotation.as_matrix()

def rotation_matrix_to_quaternion(R_opt):
    rotation = R.from_matrix(R_opt)
    return rotation.as_quat()

def rotation_matrix_from_vectors(v1, v2):
    """
    Find the rotation matrix that aligns v1 to v2
    :param v1: A 3d "source" vector
    :param v2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to v1, aligns it with v2.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0:
        axis = axis / axis_len
    angle = np.arccos(np.dot(v1, v2))

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def get_gripper_pos_orient_from_4_points(gripper_pcd):
    normal = compute_plane_normal(gripper_pcd)
    R1 = rotation_matrix_from_vectors(original_gripper_normal, normal)
    v1 = original_gripper_pcd[3] - original_gripper_pcd[0]
    v2 = gripper_pcd[3] - gripper_pcd[0]
    v1_prime = np.dot(R1, v1)
    R2 = rotation_matrix_from_vectors(v1_prime, v2)
    R = np.dot(R2, R1)
    gripper_pos = original_gripper_pos + gripper_pcd[3] - original_gripper_pcd[3]
    original_R = quaternion_to_rotation_matrix(original_gripper_orn)
    R = np.dot(R, original_R)
    gripper_orn = rotation_matrix_to_quaternion(R)
    return gripper_pos, gripper_orn

def rotation_transfer_matrix_to_6D_batch(rotate_matrix):

    # rotate_matrix.shape = (B, 9) or (B x 3, 3) rotation transpose (i.e., row vectors instead of column vectors)
    # return shape = (B, 6)

    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(-1, 9)
    rotate_matrix = rotate_matrix.reshape(-1, 9)

    return rotate_matrix[:,:6]

def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle):
    # original_gripper_pcd = np.array([[ 0.10432111,  0.00228697,  0.8474241 ],
    #         [ 0.12816067, -0.04368229,  0.8114649 ],
    #         [ 0.08953098,  0.0484529 ,  0.80711854],
    #         [ 0.11198021,  0.00245327,  0.7828771 ]])
    # original_gripper_orn = np.array([0.97841681, 0.19802945, 0.0581003 , 0.01045192])
    # original_gripper_pcd = np.array([[ 0.43856215, -0.40922496,  0.6756892 ],
    #    [ 0.3991713 , -0.42923108,  0.65513015 ],
    #    [ 0.45587012, -0.43078858,  0.6355644  ],
    #    [ 0.41987222, -0.44440767,  0.6243291 ]])
    # original_gripper_orn = np.array([ 0.69285525, -0.64422789,  0.08350163,  0.31296886])
    original_gripper_pcd = np.array([[ 0.5648266,   0.05482348,  0.34434554],
        [ 0.5642125,   0.02702148,  0.2877661 ],
        [ 0.53906703,  0.01263776,  0.38347825],
        [ 0.54250515, -0.00441092,  0.32957944]]
    )
    original_gripper_orn = np.array([0.21120763,  0.75430543, -0.61925177, -0.05423936])
    
    gripper_pcd_right_finger_closed = np.array([ 0.55415434,  0.02126799,  0.32605097])
    gripper_pcd_left_finger_closed = np.array([ 0.54912525,  0.01839125,  0.3451934 ])
    gripper_pcd_closed_finger_angle = 2.6652539383870777e-05
 
    original_gripper_pcd[1] = gripper_pcd_right_finger_closed + (original_gripper_pcd[1] - gripper_pcd_right_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
    original_gripper_pcd[2] = gripper_pcd_left_finger_closed + (original_gripper_pcd[2] - gripper_pcd_left_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
 
    # goal_R = R.from_quat(gripper_orn)
    # import pdb; pdb.set_trace()
    goal_R = R.from_quat(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos
    return gripper_pcd

def rotation_transfer_6D_to_matrix(orient):
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

    return rotate_matrix

# def get_goal_gripper_pos_eefs(actions, eef_pos, eef_quat, eef_qpos, closed_threshold=0.02, open_threshold=0.0375): # square_d2
# def get_goal_gripper_pos_eefs(actions, eef_pos, eef_quat, eef_qpos, closed_threshold=0.022, open_threshold=0.0375): # three_piece_assembly_d2
def get_goal_gripper_pos_eefs(actions, eef_pos, eef_quat, eef_qpos, closed_threshold=0.016, open_threshold=0.0375): # threading d2
# def get_goal_gripper_pos_eefs(actions, eef_pos, eef_quat, eef_qpos, closed_threshold=0.005, open_threshold=0.0375): # mug pickup d2
# def get_goal_gripper_pos_eefs(actions, eef_pos, eef_quat, eef_qpos, closed_threshold=0.01, open_threshold=0.0375): # hammer pickup d2
    # plt.plot(eef_qpos, marker='o', linestyle='-')
    # plt.title("eef_qpos trajectory")
    # plt.xlabel("Time step")
    # plt.ylabel("eef_qpos")
    # plt.show()
    is_closed = eef_qpos[:, 0] < closed_threshold
    is_open = eef_qpos[:, 0] > open_threshold
    is_closed_decision_boundary = np.where(np.diff(is_closed.astype(int)))[0]
    is_open_decision_boundary = np.where(np.diff(is_open.astype(int)))[0]
    while len(is_closed_decision_boundary) > 0 and is_closed_decision_boundary[0] < 20:
        is_closed_decision_boundary = is_closed_decision_boundary[1:]
    while len(is_open_decision_boundary) > 0 and is_open_decision_boundary[0] < 20:
        is_open_decision_boundary = is_open_decision_boundary[1:]
    switch_indices = [is_closed_decision_boundary[0]] if len(is_closed_decision_boundary) > 0 else []
    i_closed = 1
    while len(is_open_decision_boundary) > 0 and len(switch_indices) > 0 and is_open_decision_boundary[0] < switch_indices[0]:
        is_open_decision_boundary = is_open_decision_boundary[1:]
    for i_open in range(len(is_open_decision_boundary)):
        switch_indices.append(is_open_decision_boundary[i_open])
        while i_closed < len(is_closed_decision_boundary) and \
                is_closed_decision_boundary[i_closed] > switch_indices[-1]:
            i_closed += 1
    switch_indices.append(len(actions) - 1)
    switch_indices = np.array(switch_indices)
    repeat_count = np.insert(np.diff(switch_indices), 0, switch_indices[0])
    repeat_count[-1] += 1
    max_eef_qpos = np.max(np.abs(eef_qpos), axis=1, keepdims=True)
    goal_eef_pos = eef_pos[switch_indices]
    goal_eef_quat = eef_quat[switch_indices]
    goal_eef_qpos = max_eef_qpos[switch_indices]
    expanded_goal_eef_pos = np.repeat(goal_eef_pos, repeat_count, axis=0)
    expanded_goal_eef_quat = np.repeat(goal_eef_quat, repeat_count, axis=0)
    expanded_goal_eef_qpos = np.repeat(goal_eef_qpos, repeat_count, axis=0)
    # plt.plot(eef_qpos, marker='o', linestyle='-')
    # plt.plot(expanded_goal_eef_qpos, color='g', linestyle='-')
    # plt.axhline(y=closed_threshold, color='r', linestyle='-')
    # plt.axhline(y=open_threshold, color='r', linestyle='-')
    # plt.axhline(y=-closed_threshold, color='r', linestyle='-')
    # plt.axhline(y=-open_threshold, color='r', linestyle='-')
    # plt.axhline(y=open_goal_gripper_representation, color='g', linestyle='-')
    # plt.axhline(y=closed_goal_gripper_representation, color='g', linestyle='-')
    # plt.axhline(y=-open_goal_gripper_representation, color='g', linestyle='-')
    # plt.axhline(y=-closed_goal_gripper_representation, color='g', linestyle='-')
    # plt.title("eef_qpos trajectory")
    # plt.xlabel("Time step")
    # plt.ylabel("eef_qpos")
    # plt.show()
    assert expanded_goal_eef_pos.shape[0] == len(actions)
    assert expanded_goal_eef_quat.shape[0] == len(actions)
    assert expanded_goal_eef_qpos.shape[0] == len(actions)
    return expanded_goal_eef_pos, expanded_goal_eef_quat, expanded_goal_eef_qpos

def load_high_level_weighted_displacement_policy(model_name='model_60'):
    load_model_path = model_name
    # load_model_path = f"/data/minon/tax3d-conditioned-mimicgen/models/square-d2-weighted-displacement/{model_name}.pth"
    # load_model_path = f'/project_data/held/mnakuraf/RoboGen-sim2real/test_PointNet2/exps/pointnet2_super_model_invariant_2025-05-09_use_all_data_three_piece_assembly_d2_abs-obj_batch_norm/model_60.pth'
    # load_model_path = f"/data/minon/tax3d-conditioned-mimicgen/models/three-piece-assembly/{model_name}.pth"
    # load_model_path = f"/data/minon/tax3d-conditioned-mimicgen/models/threading-weighted-displacement/{model_name}.pth"
    # load_model_path = '/home/ktsim/Projects/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-15_use_all_data_threading_D2_abs-obj_threading_D2_abs/model_30.pth'
    load_model_path = '/home/ktsim/checkpoints/put_money_in_safe/pointnet2_super_model_invariant_2025-06-23_use_all_data_put_money_in_safe-obj_put_money_in_safe/model_100.pth'
    print(load_model_path)
    pointnet2_model = PointNet2_super(num_classes=13, use_in=False).to('cuda')
    pointnet2_model.load_state_dict(torch.load(load_model_path))
    pointnet2_model.eval()
    return pointnet2_model

def load_high_level_gmm_policy(epoch=30):
    load_model_path = f"/data/minon/tax3d-conditioned-mimicgen/models/gmm/square_d2/model_{epoch}.pth"
    pointnet2_model = PointNet2_super(num_classes=13, input_channel=3).to('cuda')
    pointnet2_model.load_state_dict(torch.load(load_model_path))
    pointnet2_model.eval()
    return pointnet2_model

def run_high_level_policy_inference(policy, batch, return_weights=False):
    policy.eval()
    pointcloud = batch['point_cloud'][:, -1, :, :]
    gripper_pcd = batch['gripper_pcd'][:, -1, :, :]
    inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1)
    outputs = policy(inputs_)
    weights = outputs[:, :-4, -1] # B, N
    outputs = outputs[:, :-4, :-1] # B, N, 12
    B, N, _ = outputs.shape
    outputs = outputs.view(B, N, 4, 3)
    outputs = outputs + inputs[:,:-4,:].unsqueeze(2)
    weights = torch.nn.functional.softmax(weights, dim=1)
    outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
    outputs = outputs.sum(dim=1)
    outputs = outputs.unsqueeze(1)
    if return_weights:
        return outputs, weights
    return outputs

def run_high_level_gmm_inference(policy, batch, return_weights=False, one_hot=False):
    pointcloud = batch['point_cloud'][:, -1, :, :]
    gripper_pcd = batch['gripper_pcd'][:, -1, :]
    inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
    if one_hot:
        input_onehots = torch.zeros(inputs.shape[0], inputs.shape[1], 2).to(inputs.device)
        input_onehots[:, :pointcloud.shape[1], 0] = 1
        input_onehots[:, pointcloud.shape[1]:, 1] = 1
        inputs = torch.cat([inputs, input_onehots], dim=2)

    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1)
    outputs = policy(inputs_)
    weights = outputs[:, :-4, -1] # B, N
    outputs = outputs[:, :-4, :-1] # B, N, 12
    inputs = inputs[:, :-4, :3]
    probabilities = weights  # Must sum to 1
    probabilities = torch.nn.functional.softmax(weights, dim=1)
    # import pdb; pdb.set_trace()
    sampled_index = torch.argmax(probabilities.squeeze(0))
    outputs = outputs.reshape(1, -1, 4, 3)
    displacement_mean = outputs[:, sampled_index, :, :] # B, 4, 3
    input_point_pos = inputs[:, sampled_index, :] # B, 3
    prediction = input_point_pos.unsqueeze(1) + displacement_mean # B, 4, 3
    outputs = prediction.unsqueeze(1)
    if return_weights:
        return outputs, weights
    return outputs

def get_dataloader(dataset_object, shuffle=False, batch_size=1):
    dataloader = DataLoader(dataset_object, 
                            shuffle=shuffle,
                            # sampler=DistributedSampler(dataset_object),
                            batch_size=batch_size,
                            num_workers=5,
                            pin_memory=True,
                            )
    return dataloader

def compute_new_goal_gripper_pcd(
        gripper_pcd:  np.ndarray,
        eef_qpos:     np.ndarray,
        actions:      np.ndarray,
    ) -> np.ndarray:
    # gripper_actions = actions[:,-1]
    T, N, _ = gripper_pcd.shape
    # # figure out if closing
    # derivative = np.gradient(np.abs(eef_qpos), axis=0)
    # deriv_right = derivative[:, 0]; deriv_left = derivative[:, 1]
    # is_closing_right = deriv_right < -1e-3; is_closing_right[:20] = False
    # is_closing_left = deriv_left < -1e-3; is_closing_left[:20] = False
    # is_closing = np.logical_and(is_closing_left, is_closing_right).astype(int)

    # # calculate indices
    # closing_last_indices = np.where((is_closing[1:] - is_closing[:-1]) == -1)[0] + 1
    # opening_first_indices = np.where(
    #     np.logical_and(
    #         np.sign(gripper_actions[:-1]) != np.sign(gripper_actions[1:]),
    #         np.sign(gripper_actions[1:]) == -1
    #         )
    # )[0] + 1

    # # 4) alternate between close and open, always moving forward in time
    # switches = []
    # last_t = -1
    # mode = 'close'
    # closes = closing_last_indices.tolist()
    # opens  = opening_first_indices.tolist()

    # while True:
    #     if mode == 'close':
    #         closes = [i for i in closes if i > last_t]
    #         if not closes:
    #             break
    #         t = closes.pop(0)
    #     else:
    #         opens = [i for i in opens if i > last_t]
    #         if not opens:
    #             break
    #         t = opens.pop(0)

    #     switches.append(t)
    #     last_t = t
    #     mode = 'open' if mode == 'close' else 'close'

    # if not switches or switches[-1] != T - 1:
    #     switches.append(T - 1)
    # switch_indices = np.array(switches, dtype=int)
    switch_indices = np.arange(30, T, 30)
    switch_indices[-1] = T-1
    print(switch_indices)
    # switch_indices = np.sort(np.concatenate([opening_first_indices, closing_last_indices, [T - 1]]))
    repeat_count = np.insert(np.diff(switch_indices), 0, switch_indices[0])
    repeat_count[-1] += 1
    goal_gripper_pcd = gripper_pcd[switch_indices]
    max_width = np.max(eef_qpos) * 2
    # expand the grippers of the opening indices
    # for idx, ggp in zip(switch_indices, goal_gripper_pcd): 
    #     if idx in opening_first_indices:
    #         distance = np.linalg.norm(ggp[1] - ggp[2])
    #         difference = (max_width - distance) / 2
    #         direction_vector = (ggp[1] - ggp[2]) / distance
    #         ggp[1] = ggp[1] + difference*direction_vector
    #         ggp[2] = ggp[2] - difference*direction_vector
    expanded_goal_gripper_pcd = np.repeat(goal_gripper_pcd, repeat_count, axis=0)
    assert expanded_goal_gripper_pcd.shape == gripper_pcd.shape
    return expanded_goal_gripper_pcd