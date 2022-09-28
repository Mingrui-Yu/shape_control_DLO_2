import numpy as np
from matplotlib import pyplot as plt
import torch
from torch._C import dtype
import rospy
from scipy.spatial.transform import Rotation as sciR

try:
    from utils.state_index import I
except:
    from state_index import I


env_dim = rospy.get_param("env/dimension")
num_fps = rospy.get_param("DLO/num_FPs")


# --------------------------------------------------------------------------------
def dataRandomTransform(state_input, fps_vel=None, ends_vel=None):
    if fps_vel is not None:
        assert ends_vel is not None
        if type(state_input) is not np.ndarray:
            state_input = state_input.numpy()
            fps_vel = fps_vel.numpy()
            ends_vel = ends_vel.numpy()
    else:
        assert type(state_input) is np.ndarray
    
    batch_size = state_input.shape[0]

    # rotate around Z-axis
    random_axis = np.concatenate([np.zeros((batch_size, 2)), -1 + 2 * np.random.rand(batch_size, 1)], axis=1)
    random_rotvec =  random_axis / np.linalg.norm(random_axis, axis=1, keepdims=True) * (np.pi * np.random.rand(batch_size, 1)) # axis * angle, any rotation

    if env_dim == '2D':
        random_translation = np.concatenate([-0.5 + 1 * np.random.rand(batch_size, 2), np.zeros((batch_size, 1))], axis=1)
    elif env_dim == '3D':
        random_translation =  -0.5 + 1 * np.random.rand(batch_size, 3)

    R = (sciR.from_rotvec(random_rotvec)).as_matrix().reshape(-1, 3, 3).astype('float32')
    t = random_translation.reshape(-1, 1).reshape(-1, 3, 1).astype('float32')

    # original data
    fps_pos = state_input[:, 0 : 3*num_fps]
    left_end_pos = state_input[:, 3*num_fps : 3*num_fps+3]
    left_end_quat = state_input[:, 3*num_fps+3 : 3*num_fps+7]
    right_end_pos = state_input[:, 3*num_fps+7 : 3*num_fps+10]
    right_end_quat = state_input[:, 3*num_fps+10 : 3*num_fps+14]

    left_end_ori = sciR.from_quat(left_end_quat)
    right_end_ori = sciR.from_quat(right_end_quat)
    left_end_rotmat = left_end_ori.as_matrix()
    right_end_rotmat = right_end_ori.as_matrix()

    # transform the data
    fps_pos_l = np.matmul(R.reshape(-1, 1, 3, 3), fps_pos.reshape(-1, num_fps, 3, 1)) + t.reshape(-1, 1, 3, 1)

    left_end_pos_l = np.matmul(R.reshape(-1, 3, 3), left_end_pos.reshape(-1, 3, 1)) + t.reshape(-1, 3, 1)
    left_end_quat_l = (sciR.from_matrix(np.matmul(R, left_end_rotmat))).as_quat()
    right_end_pos_l = np.matmul(R.reshape(-1, 3, 3), right_end_pos.reshape(-1, 3, 1)) + t.reshape(-1, 3, 1)
    right_end_quat_l = (sciR.from_matrix(np.matmul(R, right_end_rotmat))).as_quat()

    if fps_vel is not None:
        fps_vel_l = np.matmul(R.reshape(-1, 1, 3, 3), fps_vel.reshape(-1, num_fps, 3, 1)).reshape(-1, 3*num_fps)
        ends_vel_l = np.matmul(R.reshape(-1, 1, 3, 3), ends_vel.reshape(-1, 4, 3, 1)).reshape(-1, 12)

    state_input_l = np.zeros(state_input.shape, dtype='float32')
    state_input_l[:, 0 : 3*num_fps] = fps_pos_l.reshape(batch_size, -1)
    state_input_l[:, 3*num_fps+0 : 3*num_fps+3] = left_end_pos_l.reshape(batch_size, -1)
    state_input_l[:, 3*num_fps+3 : 3*num_fps+7] = left_end_quat_l.reshape(batch_size, -1)
    state_input_l[:, 3*num_fps+7 : 3*num_fps+10] = right_end_pos_l.reshape(batch_size, -1)
    state_input_l[:, 3*num_fps+10 : 3*num_fps+14] = right_end_quat_l.reshape(batch_size, -1)

    if fps_vel is not None:
        return torch.tensor(state_input_l), torch.tensor(fps_vel_l), torch.tensor(ends_vel_l)
    else:
        return state_input_l



# # --------------------------------------------------------------------------------
# def fdmDataRandomTransform(fps_pos, ends_pose, ends_vel, output):
#     fps_pos = fps_pos.numpy()
#     ends_pose = fps
#     output = output.numpy()
    
#     batch_size = input.shape[0]

#     # rotate around Z-axis
#     random_axis = np.concatenate([np.zeros((batch_size, 2)), -1 + 2 * np.random.rand(batch_size, 1)], axis=1)
#     random_rotvec =  random_axis / np.linalg.norm(random_axis, axis=1, keepdims=True) * (np.pi * np.random.rand(batch_size, 1)) # axis * angle, any rotation

#     if env_dim == '2D':
#         random_translation = np.concatenate([-0.5 + 1 * np.random.rand(batch_size, 2), np.zeros((batch_size, 1))], axis=1)
#     elif env_dim == '3D':
#         random_translation =  -0.5 + 1 * np.random.rand(batch_size, 3)

#     R = (sciR.from_rotvec(random_rotvec)).as_matrix().reshape(-1, 3, 3).astype('float32')
#     t = random_translation.reshape(-1, 1).reshape(-1, 3, 1).astype('float32')

#     # original data
#     fps_pos = input[:, 0 : 3*num_fps]
#     left_end_pos = input[:, 3*num_fps : 3*num_fps+3]
#     left_end_quat = input[:, 3*num_fps+3 : 3*num_fps+7]
#     right_end_pos = input[:, 3*num_fps+7 : 3*num_fps+10]
#     right_end_quat = input[:, 3*num_fps+10 : 3*num_fps+14]
#     ends_vel = input[:, 3*num_fps+14 :  3*num_fps+14+12]

#     left_end_ori = sciR.from_quat(left_end_quat)
#     right_end_ori = sciR.from_quat(right_end_quat)
#     left_end_rotmat = left_end_ori.as_matrix()
#     right_end_rotmat = right_end_ori.as_matrix()

#     # transform the data
#     fps_pos_l = np.matmul(R.reshape(-1, 1, 3, 3), fps_pos.reshape(-1, num_fps, 3, 1)) + t.reshape(-1, 1, 3, 1)

#     left_end_pos_l = np.matmul(R.reshape(-1, 3, 3), left_end_pos.reshape(-1, 3, 1)) + t.reshape(-1, 3, 1)
#     left_end_quat_l = (sciR.from_matrix(np.matmul(R, left_end_rotmat))).as_quat()
#     right_end_pos_l = np.matmul(R.reshape(-1, 3, 3), right_end_pos.reshape(-1, 3, 1)) + t.reshape(-1, 3, 1)
#     right_end_quat_l = (sciR.from_matrix(np.matmul(R, right_end_rotmat))).as_quat()

#     output_l = np.matmul(R.reshape(-1, 1, 3, 3), output.reshape(-1, num_fps, 3, 1)).reshape(-1, 3*num_fps)
#     ends_vel_l = np.matmul(R.reshape(-1, 1, 3, 3), ends_vel.reshape(-1, 4, 3, 1)).reshape(-1, 12)

#     input_l = np.zeros(input.shape, dtype='float32')
#     input_l[:, 0 : 3*num_fps] = fps_pos_l.reshape(batch_size, -1)
#     input_l[:, 3*num_fps+0 : 3*num_fps+3] = left_end_pos_l.reshape(batch_size, -1)
#     input_l[:, 3*num_fps+3 : 3*num_fps+7] = left_end_quat_l.reshape(batch_size, -1)
#     input_l[:, 3*num_fps+7 : 3*num_fps+10] = right_end_pos_l.reshape(batch_size, -1)
#     input_l[:, 3*num_fps+10 : 3*num_fps+14] = right_end_quat_l.reshape(batch_size, -1)
#     input_l[:, 3*num_fps+14 :3*num_fps+14+12] = ends_vel_l.reshape(batch_size, -1)

#     return torch.tensor(input_l), torch.tensor(output_l)


# ----------------------------------------------------------------------------------------------------------------
def desiredShapesRandomTransform(desired_shapes):
    batch_size = desired_shapes.shape[0]

    # rotate around Z-axis
    random_axis = np.concatenate([np.zeros((batch_size, 2)), -1 + 2 * np.random.rand(batch_size, 1)], axis=1)
    random_rotvec =  random_axis / np.linalg.norm(random_axis, axis=1, keepdims=True) * (1/2 * np.pi *  np.random.rand(batch_size, 1)) # 最多转90度
    # random_rotvec = np.array([[0, 0, np.pi]])

    random_translation =  -0.3 + 0.6 * np.random.rand(batch_size, 3)
    # random_translation =  0 * np.random.rand(batch_size, 3)
    if env_dim == '2D':
        random_translation[:, 2] = 0

    R = (sciR.from_rotvec(random_rotvec)).as_matrix().reshape(-1, 3, 3).astype('float32')
    t = random_translation.reshape(-1, 1).reshape(-1, 3, 1).astype('float32')

    # transform the data
    desired_shapes_aug = np.matmul(R.reshape(-1, 1, 3, 3), desired_shapes.reshape(-1, num_fps, 3, 1)) + t.reshape(-1, 1, 3, 1)
    desired_shapes_aug.reshape(-1, num_fps*3)

    return desired_shapes_aug


# ----------------------------------------------------------------------------------------------------------------
def datasetRandomTransform(state):
    state = np.array(state, dtype='float32')
    num_data = state.shape[0]

    # state data in world frame
    state_input = state[:, I.state_input_idx]
    fps_vel = state[:, I.fps_vel_idx]
    ends_vel = state[:, I.ends_vel_idx]

    state_input_l, fps_vel_l, ends_vel_l = dataRandomTransform(state_input, fps_vel, ends_vel)
    state_input_l = state_input_l.numpy()
    fps_vel_l = fps_vel_l.numpy().reshape(-1, num_fps*3)
    ends_vel_l = ends_vel_l.numpy().reshape(-1, 12)

    state_l = state.copy()
    state_l[:, I.state_input_idx] = state_input_l
    state_l[:, I.fps_vel_idx] = fps_vel_l
    state_l[:, I.ends_vel_idx] = ends_vel_l

    return state_l




# --------------------------------------------------------------------------------------------
# transform state given rotation and translation
def statesTransform(state, rotvec, translation):
    state = np.array(state, dtype='float32')
    data_size = state.shape[0]

    b_input_is_one = False
    if len(state.shape) == 1:
        state = np.expand_dims(state, axis=0)
        b_input_is_one = True

    # state data in world frame
    fps_pos = state[:, I.fps_pos_idx]
    left_end_pos = state[:, I.left_end_pos_idx]
    left_end_quat = state[:, I.left_end_quat_idx]
    right_end_pos = state[:, I.right_end_pos_idx]
    right_end_quat = state[:, I.right_end_quat_idx]
    fps_vel = state[:, I.fps_vel_idx]
    left_end_vel = state[:, I.left_end_vel_idx]
    right_end_vel = state[:, I.right_end_vel_idx]

    left_end_ori = sciR.from_quat(left_end_quat)
    right_end_ori = sciR.from_quat(right_end_quat)
    left_end_rotmat = left_end_ori.as_matrix()
    right_end_rotmat = right_end_ori.as_matrix()

    rotmat = sciR.from_rotvec(rotvec).as_matrix().reshape(1, 3, 3)

    # transformer
    R_l2w = rotmat
    p_l_in_w = np.array(translation).reshape(-1, 3, 1)
    R_w2l = np.transpose(R_l2w, axes=[0, 2, 1])
    p_w_in_l = np.matmul(-R_w2l, p_l_in_w)

    # transform data to local frame
    fps_pos_l = np.matmul(R_w2l.reshape(-1, 1, 3, 3), fps_pos.reshape(-1, num_fps, 3, 1)) + p_w_in_l.reshape(-1, 1, 3, 1)
    fps_vel_l = np.matmul(R_w2l.reshape(-1, 1, 3, 3), fps_vel.reshape(-1, num_fps, 3, 1))

    left_end_pos_l = np.matmul(R_w2l.reshape(-1, 3, 3), left_end_pos.reshape(-1, 3, 1)) + p_w_in_l.reshape(-1, 3, 1)
    left_end_quat_l = (sciR.from_matrix(np.matmul(R_w2l, left_end_rotmat))).as_quat()
    right_end_pos_l = np.matmul(R_w2l.reshape(-1, 3, 3), right_end_pos.reshape(-1, 3, 1)) + p_w_in_l.reshape(-1, 3, 1)
    right_end_quat_l = (sciR.from_matrix(np.matmul(R_w2l, right_end_rotmat))).as_quat()

    left_end_vel_l = np.matmul(R_w2l.reshape(-1, 1, 3, 3), left_end_vel.reshape(-1, 2, 3, 1))
    right_end_vel_l = np.matmul(R_w2l.reshape(-1, 1, 3, 3), right_end_vel.reshape(-1, 2, 3, 1))

    state_l = state
    state_l[:, I.fps_pos_idx] = fps_pos_l.reshape(data_size, -1)
    state_l[:, I.left_end_pos_idx] = left_end_pos_l.reshape(data_size, -1)
    state_l[:, I.left_end_quat_idx] = left_end_quat_l.reshape(data_size, -1)
    state_l[:, I.right_end_pos_idx] = right_end_pos_l.reshape(data_size, -1)
    state_l[:, I.right_end_quat_idx] = right_end_quat_l.reshape(data_size, -1)
    state_l[:, I.fps_vel_idx] = fps_vel_l.reshape(data_size, -1)
    state_l[:, I.left_end_vel_idx] = left_end_vel_l.reshape(data_size, -1)
    state_l[:, I.right_end_vel_idx] = right_end_vel_l.reshape(data_size, -1)

    if b_input_is_one:
        return state_l.reshape(-1, )
    else:
        return state_l





# -------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    from my_plot import plot3dState

    project_dir = rospy.get_param("project_dir")
    env_dim = rospy.get_param("env/dimension")
    num_fps = rospy.get_param("DLO/num_FPs")

    states_world = np.load(project_dir + "data/train_data/"+ env_dim + "/state.npy").astype("float32")

    states_aug = datasetRandomTransform(states_world)

    plot3dState(states_aug[100, :])


    plt.show()