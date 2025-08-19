import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from torch.utils.data import DataLoader
from third_party.robogen.test_PointNet2.model_invariant import PointNet2_super, PointNet2_Binary, PointNet2_text, PointNet2GripperBinary
from matplotlib import pyplot as plt
import torch
from termcolor import cprint

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

# def rotation_matrix_from_vectors(v1, v2):
#     """
#     Find the rotation matrix that aligns v1 to v2
#     :param v1: A 3d "source" vector
#     :param v2: A 3d "destination" vector
#     :return mat: A transform matrix (3x3) which when applied to v1, aligns it with v2.
#     """
#     v1 = v1 / np.linalg.norm(v1)
#     v2 = v2 / np.linalg.norm(v2)
#     axis = np.cross(v1, v2)
#     axis_len = np.linalg.norm(axis)
#     if axis_len != 0:
#         axis = axis / axis_len
#     angle = np.arccos(np.dot(v1, v2))

#     K = np.array([[0, -axis[2], axis[1]],
#                   [axis[2], 0, -axis[0]],
#                   [-axis[1], axis[0], 0]])

#     R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
#     return R

def rotation_matrix_from_vectors(v1, v2, eps=1e-8):
    """
    Find the rotation matrix that aligns v1 to v2
    :param v1: A 3D "source" vector
    :param v2: A 3D "destination" vector
    :return: A 3x3 rotation matrix that rotates v1 to v2
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    # normalize
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        # one vector is too small: return identity
        return np.eye(3)

    v1 = v1 / n1
    v2 = v2 / n2

    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)  # prevent invalid arccos

    # if vectors are almost identical -> no rotation needed
    if np.isclose(dot, 1.0, atol=1e-6):
        return np.eye(3)

    # if vectors are opposite -> need a 180° rotation around some orthogonal axis
    if np.isclose(dot, -1.0, atol=1e-6):
        # find an orthogonal vector
        orthogonal = np.array([1.0, 0.0, 0.0])
        if abs(v1[0]) > 0.9:
            orthogonal = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v1, orthogonal)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * np.dot(K, K)  # 180° rotation

    # general case
    axis = np.cross(v1, v2)
    axis /= np.linalg.norm(axis)

    angle = np.arccos(dot)

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def project_to_rotation_matrix(R):
    if not np.all(np.isfinite(R)):
        # fallback: identity or safe rotation
        return np.eye(3)

    try:
        U, _, Vt = np.linalg.svd(R)
    except np.linalg.LinAlgError:
        # fallback if SVD still fails
        return np.eye(3)

    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj


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
    R = project_to_rotation_matrix(R)
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

def load_high_level_weighted_displacement_policy(task_name, epoch):
    if task_name == 'put_money_in_safe':
        # load_model_path = '/home/ktsim/Projects/tax3d-conditioned-mimicgen/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-15_use_all_data_threading_D2_abs-obj_threading_D2_abs/model_30.pth'
        # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-26_use_all_data_put_money_in_safe-obj_use_gripper_open_use_collision_use_color_put_money_in_safe/model_100.pth' # Predictions gripper and collision too
        # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-06-30_use_all_data_put_money_in_safe-obj_use_gripper_open_use_collision_use_color_put_money_in_safe/model_100.pth' # Same as above but with weight adjusted
        # load_model_path = '/home/ktsim/checkpoints/put_money_in_safe/pointnet2_super_model_invariant_2025-06-23_use_all_data_put_money_in_safe-obj_put_money_in_safe/model_100.pth' # No gripper nor collision
        # load_model_path = '/home/ktsim/checkpoints/put_money_in_safe/pointnet2_super_model_invariant_2025-07-10_use_all_data_put_money_in_safe-obj_use_color_put_money_in_safe/best_model.pth' # No gripper nor collision
        # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-07-15_use_all_data_put_money_in_safe-obj_use_color_put_money_in_safe/best_model.pth' # Text embedding
        # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-07-18_use_all_data_put_money_in_safe-obj_use_color_new_dataset/best_model.pth'
        # load_model_path = '/home/ktsim/checkpoints/put_money_in_safe/pointnet2_super_model_invariant_2025-07-23_use_all_data_put_money_in_safe-obj_use_color_use_text_more_epochs/best_model.pth' # MOre epochs
        # load_model_path = '/home/ktsim/checkpoints/put_money_in_safe/pointnet2_super_model_invariant_2025-07-24_use_all_data_put_money_in_safe-obj_use_color_use_text_best_model/best_model.pth' # Best model
        # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-07-28_use_all_data_put_money_in_safe-obj_one_hot_use_color_use_text_one_hot_and_reduction/best_model.pth'
        # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-07-30_use_all_data_put_money_in_safe-obj_one_hot_use_color_use_text_colosseum/best_model.pth'
        # load_model_path = '/home/ktsim/checkpoints/put_money_in_safe/pointnet2_super_model_invariant_2025-07-29_use_all_data_put_money_in_safe-obj_one_hot_use_color_use_text_reduction_200/best_model.pth' # best one so far 7/31 84%
        
        # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-08-06_use_all_data_put_money_in_safe-obj_one_hot_use_color_use_text_/best_model.pth' # transforms
        # load_model_path = '/home/ktsim/checkpoints/pointnet2_super_model_invariant_2025-08-08_use_all_data_put_money_in_safe-obj_one_hot_use_color_use_text_so2_aug_dataset_fix/best_model.pth'
        # load_model_path = '/home/ktsim/checkpoints/pointnet2_super_model_invariant_2025-08-09_use_all_data_put_money_in_safe-obj_one_hot_use_color_use_text_dataset_fix/best_model.pth'
        load_model_path = '/home/ktsim/checkpoints/2025-08-15/pointnet2_super_model_invariant_2025-08-15_use_all_data_put_money_in_safe-obj_one_hot_use_color_use_text_500_epochs/model_{}.pth'.format(epoch)
    elif task_name == 'reach_and_drag':
        # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-08-02_use_all_data_reach_and_drag-obj_one_hot_use_color_use_text_/best_model.pth'
        # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_super_model_invariant_2025-08-13_use_all_data_reach_and_drag-obj_one_hot_use_color_use_text_keyframe_fixed/best_model.pth'
        # load_model_path = '/home/ktsim/checkpoints/pointnet2_super_model_invariant_2025-08-14_use_all_data_reach_and_drag-obj_one_hot_use_color_use_text_more_epochs/model_300.pth'
        load_model_path = '/home/ktsim/checkpoints/2025-08-15/pointnet2_super_model_invariant_2025-08-15_use_all_data_reach_and_drag-obj_one_hot_use_color_use_text_500_epochs/model_{}.pth'.format(epoch)
    elif task_name == 'place_cups':
        load_model_path = '/home/ktsim/checkpoints/place_cups/pointnet2_super_model_invariant_2025-08-01_use_all_data_place_cups-obj_one_hot_use_color_use_text_new_task/best_model.pth'
    elif task_name == 'insert_onto_square_peg':
        # load_model_path = '/home/ktsim/checkpoints/2025-08-16/pointnet2_super_model_invariant_2025-08-16_use_all_data_insert_onto_square_peg-obj_one_hot_use_color_use_text_/model_{}.pth'.format(epoch)
        load_model_path = '/home/ktsim/checkpoints/pointnet2_super_model_invariant_2025-08-18_use_all_data_insert_onto_square_peg-obj_one_hot_use_color_use_text_so2_/model_{}.pth'.format(epoch)
    elif task_name == 'place_shape_in_shape_sorter':
        load_model_path = '/home/ktsim/checkpoints/2025-08-16/pointnet2_super_model_invariant_2025-08-16_use_all_data_place_shape_in_shape_sorter-obj_one_hot_use_color_use_text_/model_{}.pth'.format(epoch)
    elif task_name == 'stack_cups':
        load_model_path = '/home/ktsim/checkpoints/2025-08-16/pointnet2_super_model_invariant_2025-08-16_use_all_data_stack_cups-obj_one_hot_use_color_use_text_/model_{}.pth'.format(epoch)
    elif task_name == 'put_groceries_in_cupboard':
        load_model_path = '/home/ktsim/checkpoints/2025-08-17/pointnet2_super_model_invariant_2025-08-16_use_all_data_put_groceries_in_cupboard-obj_one_hot_use_color_use_text_/model_{}.pth'.format(epoch)
    elif task_name == 'open_drawer':
        load_model_path = '/home/ktsim/checkpoints/2025-08-17/pointnet2_super_model_invariant_2025-08-17_use_all_data_open_drawer-obj_one_hot_use_color_use_text_/model_{}.pth'.format(epoch)
    cprint(load_model_path, color='blue')
    # pointnet2_model = PointNet2_super(num_classes=13, input_channel=6, use_in=False).to('cuda')
    pointnet2_model = PointNet2_text(num_classes=13, input_channel=8, use_text_embedding=True).to('cuda')

    pointnet2_model.load_state_dict(torch.load(load_model_path))
    pointnet2_model.eval()
    return pointnet2_model

def load_high_level_binary_prediction(gripper=False, collision=False, task_name=None, epoch=150):

    if collision:
        if task_name == 'put_money_in_safe':
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/collision_pointnet2_binary_model_invariant_2025-08-13_use_all_data_put_money_in_safe-obj_one_hot_no_weight_use_color_use_text_action_corrected/best_model.pth'
            load_model_path = '/home/ktsim/checkpoints/2025-08-15/collision_pointnet2_binary_model_invariant_2025-08-15_use_all_data_put_money_in_safe-obj_one_hot_no_weight_use_color_use_text_500_epochs/model_{}.pth'.format(epoch)
        elif task_name == 'reach_and_drag':
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/pointnet2_binary_model_invariant_2025-08-02_use_all_data_reach_and_drag-obj_one_hot_no_weight_use_color_use_text/best_model.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/collision_pointnet2_binary_model_invariant_2025-08-13_use_all_data_reach_and_drag-obj_one_hot_no_weight_use_color_use_text/best_model.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/collision_pointnet2_binary_model_invariant_2025-08-13_use_all_data_reach_and_drag-obj_one_hot_no_weight_use_color_use_textkeyframe_fixed/best_model.pth'
            # load_model_path = '/home/ktsim/checkpoints/pointnet2_binary_model_invariant_2025-08-14_use_all_data_reach_and_drag-obj_one_hot_no_weight_use_color_use_text_keyframe_fixed/best_collision_model.pth'
            load_model_path = '/home/ktsim/checkpoints/2025-08-15/collision_pointnet2_binary_model_invariant_2025-08-15_use_all_data_reach_and_drag-obj_one_hot_no_weight_use_color_use_text_500_epochs/model_{}.pth'.format(epoch)
        elif task_name == 'place_cups':
            load_model_path = '/home/ktsim/checkpoints/place_cups/pointnet2_binary_model_invariant_2025-08-01_use_all_data_place_cups-obj_one_hot_no_weight_use_color_use_textnew_task/best_model.pth'
            load_model_path = '/home/ktsim/checkpoints/2025-08-17/pointnet2_super_model_invariant_2025-08-16_use_all_data_place_cups-obj_one_hot_use_color_use_text_gmm_trying_gmm/model_{}.pth'.format(epoch)
        elif task_name == 'insert_onto_square_peg':
            load_model_path = '/home/ktsim/checkpoints/2025-08-16/collision_pointnet2_binary_model_invariant_2025-08-16_use_all_data_insert_onto_square_peg-obj_one_hot_no_weight_use_color_use_text_/model_{}.pth'.format(epoch)
        elif task_name == 'place_shape_in_shape_sorter':
            load_model_path = '/home/ktsim/checkpoints/2025-08-16/collision_pointnet2_binary_model_invariant_2025-08-16_use_all_data_place_shape_in_shape_sorter-obj_one_hot_no_weight_use_color_use_text_/model_{}.pth'.format(epoch)
        elif task_name == 'stack_cups':
            load_model_path = '/home/ktsim/checkpoints/2025-08-16/collision_pointnet2_binary_model_invariant_2025-08-16_use_all_data_stack_cups-obj_one_hot_no_weight_use_color_use_text_/model_{}.pth'.format(epoch)
        elif task_name == 'put_groceries_in_cupboard':
            load_model_path = '/home/ktsim/checkpoints/2025-08-17/collision_pointnet2_binary_model_invariant_2025-08-16_use_all_data_put_groceries_in_cupboard-obj_one_hot_no_weight_use_color_use_text_/model_{}.pth'.format(epoch)
        elif task_name == 'open_drawer':
            load_model_path  = '/home/ktsim/checkpoints/2025-08-17/collision_pointnet2_binary_model_invariant_2025-08-17_use_all_data_open_drawer-obj_one_hot_no_weight_use_color_use_text_/model_{}.pth'.format(epoch)
        pointnet2_model = PointNet2_Binary(num_classes=1, input_channel=9, use_text_embedding=True).to('cuda')

    elif gripper:
        if task_name == 'put_money_in_safe':
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-12_use_all_data_put_money_in_safe-obj_no_weight_use_text_keyframes_goal_classifier/best_model.pth' # 84%, we should use this
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-13_use_all_data_put_money_in_safe-obj_one_hot_no_weight_use_color_use_textaction_corrected_goal_classifier/model_10.pth' # 80%
            load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-16_use_all_data_put_money_in_safe-obj_no_weight_use_text_/model_20.pth'
        elif task_name == 'reach_and_drag':
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/binary_pointnet2_binary_model_invariant_2025-08-04_use_all_data_reach_and_drag-obj_no_weight_distance/best_model.pth' # best one so far
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-05_use_all_data_reach_and_drag-obj_one_hot_no_weight_distance_random/model_100.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-06_use_all_data_reach_and_drag-obj_one_hot_no_weight_distance/model_100.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-09_use_all_data_reach_and_drag-obj_no_weight_use_text_just_goal/model_50.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-09_use_all_data_reach_and_drag-obj_no_weight_use_text_just_goal_inconsistent/model_50.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-09_use_all_data_reach_and_drag-obj_no_weight_use_text_dataset_fix_just_goal/best_model.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-13_use_all_data_reach_and_drag-obj_no_weight_use_textaction_corrected_goal_classifier/best_model.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-13_use_all_data_reach_and_drag-obj_no_weight_use_textkeyframes_goal_classifier/best_model.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-13_use_all_data_reach_and_drag-obj_one_hot_no_weight_use_color_use_text_/best_model.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-13_use_all_data_reach_and_drag-obj_no_weight_use_text_keyframes_fixed/best_model.pth'
            # load_model_path = '/home/ktsim/checkpoints/pointnet2_binary_model_invariant_2025-08-14_use_all_data_reach_and_drag-obj_one_hot_no_weight_use_color_use_text_keyframe_fixed/best_gripper_model.pth'
            load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-14_use_all_data_reach_and_drag-obj_no_weight_use_text_keyframes_fixed/model_50.pth'
        elif task_name == 'place_cups':
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/binary_pointnet2_binary_model_invariant_2025-08-04_use_all_data_place_cups-obj_no_weight_distance/best_model.pth'
            # load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-05_use_all_data_place_cups-obj_one_hot_no_weight_distance_random/model_100.pth'
            load_model_path = '/home/ktsim/checkpoints/2025-08-17/collision_pointnet2_binary_model_invariant_2025-08-16_use_all_data_place_cups-obj_one_hot_no_weight_use_color_use_text_/model_{}.pth'.format(epoch)
        elif task_name == 'insert_onto_square_peg':
            load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-17_use_all_data_insert_onto_square_peg-obj_no_weight_use_text_/model_20.pth'
        elif task_name == 'place_shape_in_shape_sorter':
            load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-17_use_all_data_place_shape_in_shape_sorter-obj_no_weight_use_text_/model_20.pth'
        elif task_name == 'stack_cups':
            load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-17_use_all_data_stack_cups-obj_no_weight_use_text_/model_20.pth'
        elif task_name == 'put_groceries_in_cupboard':
            load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-18_use_all_data_put_groceries_in_cupboard-obj_no_weight_use_text_/model_20.pth'
        elif task_name == 'open_drawer':
            load_model_path = '/home/ktsim/Projects/SAM2Act/third_party/robogen/test_PointNet2/exps/gripper_pointnet2_binary_model_invariant_2025-08-18_use_all_data_open_drawer-obj_no_weight_use_text_/model_20.pth'
        pointnet2_model = PointNet2GripperBinary(num_classes=1, input_channel=3, use_text_embedding=True).to('cuda')
    
    cprint(load_model_path, color='yellow')
    pointnet2_model.load_state_dict(torch.load(load_model_path))
    pointnet2_model.eval()
    return pointnet2_model

def load_high_level_gmm_policy(epoch=30):
    load_model_path = f"/data/minon/tax3d-conditioned-mimicgen/models/gmm/square_d2/model_{epoch}.pth"
    pointnet2_model = PointNet2_super(num_classes=13, input_channel=3).to('cuda')
    pointnet2_model.load_state_dict(torch.load(load_model_path))
    pointnet2_model.eval()
    return pointnet2_model

def run_high_level_policy_inference(policy, batch, text_embedding=None, return_weights=False, binary_prediction=False):
    policy.eval()
    pointcloud = batch['point_cloud'][:, -1, :, :]
    gripper_pcd = batch['gripper_pcd'][:, -1, :, :]

    inputs = torch.cat([pointcloud, gripper_pcd], dim=1).float()
    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1)
    
    if text_embedding is not None:
        outputs = policy(inputs_, text_embedding)
    else:
        outputs = policy(inputs_)
    if outputs.shape[-1] == 15:
        collision = outputs[:, :-4, -1] # B, N
        gripper_open = outputs[:, :-4, -2] # B, N
        weights = outputs[:, :-4, -3] # B, N
        outputs = outputs[:, :-4, :-3] # B, N, 12
    elif outputs.shape[-1] == 13:
        weights = outputs[:, :-4, -1] # B, N
        outputs = outputs[:, :-4, :-1] # B, N, 12

    B, N, _ = outputs.shape
    outputs = outputs.view(B, N, 4, 3)
    outputs = outputs + inputs[:,:-4,:3].unsqueeze(2)
    weights = torch.nn.functional.softmax(weights, dim=1)
    outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
    outputs = outputs.sum(dim=1)
    outputs = outputs.unsqueeze(1)

    if binary_prediction:
        gripper_open = torch.sigmoid(gripper_open)
        collision = torch.sigmoid(collision)

        gripper_open = (gripper_open * weights).sum(dim=1, keepdim=True)
        collision = (collision * weights).sum(dim=1, keepdim=True)
        gripper_open = gripper_open.unsqueeze(1)
        collision = collision.unsqueeze(1)

        gripper_open = (gripper_open > 0.5).float()
        collision = (collision > 0.5).float()

        return gripper_open, collision

    if return_weights:
        return outputs, weights
    else:
        return outputs

def collision_binary_inference(policy, batch, return_weights=False, text_embedding=None):
    policy.eval()
    pointcloud = batch['point_cloud'][:, -1, :, :]
    gripper_pcd = batch['gripper_pcd'][:, -1, :, :]
    goal_gripper_pcd = batch['goal_gripper_pcd'][:, -1, :, :]

    inputs = torch.cat([pointcloud, gripper_pcd, goal_gripper_pcd], dim=1).float()
    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1)
    outputs = policy(inputs_, text_embedding)
    
    # weights = outputs[:, :-4, 0]
    # gripper_open = outputs [:, :-4, 1] # B
    # collision = outputs [:, :-4, 2] # B
    # weights = torch.nn.functional.softmax(weights, dim=1)

    collision = outputs[:, 0]
    collision = torch.sigmoid(collision)
    collision = (collision > 0.5).float()

    return collision
    
    # Weighted average
    # gripper_open = (gripper_open * weights).sum(dim=1, keepdim=True)
    # collision = (collision * weights).sum(dim=1, keepdim=True)


    # gripper_open = gripper_open.unsqueeze(1)
    # collision = collision.unsqueeze(1)

    # return gripper_open, collision


def gripper_binary_inference(policy, batch, return_weights=False, text_embedding=None):
    policy.eval()
    # pointcloud = batch['point_cloud'][:, -1, :, :]
    # gripper_pcd = batch['gripper_pcd'].float() #[:, -1, :, :]
    goal_gripper_pcd = batch['goal_gripper_pcd'].unsqueeze(0).to('cuda') #[:, -1, :, :]
    centroid = torch.mean(goal_gripper_pcd, dim=1, keepdim=True) # B, 1, 3
    goal_gripper_pcd = goal_gripper_pcd - centroid # B, 4
    inputs = goal_gripper_pcd.float()

    # inputs = torch.cat([pointcloud, goal_gripper_pcd], dim=1).float()
    # distance = torch.norm(goal_gripper_pcd[:, 1, :3] - goal_gripper_pcd[:, 2, :3], keepdim=True, dim=1) # B, 1
    inputs = inputs.to('cuda')
    inputs_ = inputs.permute(0, 2, 1)
    # outputs = policy(delta, goal_gripper_pcd, text_embedding)
    outputs = policy(inputs_, text_embedding)

    # weights = outputs[:, :-4, 0]
    # gripper_open = outputs [:, :-4, 1] # B
    # collision = outputs [:, :-4, 2] # B
    # weights = torch.nn.functional.softmax(weights, dim=1)

    gripper_open = outputs[:, 0]
    gripper_open = torch.sigmoid(gripper_open)
    gripper_open = (gripper_open > 0.5).float()
    
    return gripper_open

    # Weighted average
    # gripper_open = (gripper_open * weights).sum(dim=1, keepdim=True)
    # collision = (collision * weights).sum(dim=1, keepdim=True)


    # gripper_open = gripper_open.unsqueeze(1)
    # collision = collision.unsqueeze(1)

    # return gripper_open, collision

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