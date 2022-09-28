import rospy

num_fps = rospy.get_param("DLO/num_FPs")


class Index(object):
    def __init__(self):
        length_start = 0
        length_end = 1

        fps_pos_start = length_end
        fps_pos_end = fps_pos_start + 3*num_fps

        end_pose_start = fps_pos_end
        end_pose_end = end_pose_start + 14

        fps_vel_start = end_pose_end
        fps_vel_end = fps_vel_start + 3*num_fps

        ends_vel_start = fps_vel_end
        ends_vel_end = ends_vel_start + 12

        desired_pos_start = ends_vel_end
        desired_pos_end = desired_pos_start +3*num_fps

        self.state_dim = desired_pos_end

        self.length_idx = list(range(length_start, length_end))
        self.fps_pos_idx = list(range(fps_pos_start, fps_pos_end))
        self.end_pose_idx = list(range(end_pose_start, end_pose_end))
        self.fps_vel_idx = list(range(fps_vel_start, fps_vel_end))
        self.ends_vel_idx = list(range(ends_vel_start, ends_vel_end))
        self.desired_pos_idx = list(range(desired_pos_start, desired_pos_end))

        self.left_end_pos_idx = self.end_pose_idx[0:3]
        self.left_end_quat_idx = self.end_pose_idx[3:7]
        self.right_end_pos_idx = self.end_pose_idx[7:10]
        self.right_end_quat_idx = self.end_pose_idx[10:14]

        self.left_end_vel_idx = self.ends_vel_idx[0:6]
        self.right_end_vel_idx = self.ends_vel_idx[6:12]
        self.left_end_lvel_idx = self.ends_vel_idx[0:3]
        self.left_end_avel_idx = self.ends_vel_idx[3:6]
        self.right_end_lvel_idx = self.ends_vel_idx[6:9]
        self.right_end_avel_idx = self.ends_vel_idx[9:12]

        self.state_input_idx = self.fps_pos_idx + self.end_pose_idx


I = Index()

# print(I.length_idx)
# print(I.fps_pos_idx)
# print(I.end_pose_idx)
# print(I.fps_vel_idx)
# print(I.ends_vel_idx)
# print(I.desired_pos_idx)

