#!/usr/bin/env python

# compare the control results of methods
# load the control results and calcualte the success rate, average task time, averaget task error

import os, sys
parrentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parrentdir)

import numpy as np
from matplotlib import pyplot as plt
import rospy
from utils.state_index import I


project_dir = rospy.get_param("project_dir")
num_fps = rospy.get_param("DLO/num_FPs")
env = rospy.get_param("env/sim_or_real")
target_points_idx = rospy.get_param("controller/object_fps_idx")


# ------------------------------------------------------------------------------
def evaluateControlResults(names, num_case=100, delta_t=0.1):

    env_dim = rospy.get_param("env/dimension")
    
    threshold = 0.05
    if env == 'sim':
        sequence_length = 30 * 10
        control_rate = 10

    target_dim = []
    for target_point_idx in target_points_idx:
        target_dim += [3*target_point_idx, 3*target_point_idx+1, 3*target_point_idx+2]

    all_success = np.zeros((num_case, len(names)))
    k = 0
    for name in names:
        task_error_success = []
        task_error_all = []
        task_time = []
        success = 0
        for i in range(num_case):
            if env == 'sim':
                state = np.load(project_dir + "results/" + env + "/control/" + name + "/" + env_dim  + "/state_" + str(i) + ".npy")

            desired_positions = state[-1, I.desired_pos_idx] 
            positions = state[:, I.fps_pos_idx]

            error = np.linalg.norm((positions - desired_positions)[:, target_dim], axis=1)
            
            # if doesn't overstretch and the final error is less than the threshold
            if(state.shape[0] >= sequence_length - control_rate and np.all(error[-control_rate : -1] < threshold)):
                if env == 'sim' and env_dim == '2D' and np.any(positions.reshape(-1, 10, 3)[1:, :, 2] > 0.005): # 2D if the DLO left the table
                    continue
                success += 1
                all_success[i, k] = 1
                time = np.min(np.where(error < threshold)) * delta_t
                task_time.append(time)
                task_error_success.append( np.mean(error[-control_rate : -1]))

            # for all cases (not just successful cases)
            task_error_all.append( np.mean(error[-control_rate : -1]))

        if task_error_success == []:
              ave_task_error_success = 0
              ave_task_time = 0  
        else:
            ave_task_error_success = np.mean(np.array(task_error_success))
            ave_task_time = np.mean(np.array(task_time))
        ave_task_error_all = np.mean(np.array(task_error_all))
        
        print(name, " Success: ", success, ", Task time (s): ", ave_task_time, ", Success  Task error  (cm): ", ave_task_error_success * 100, ", All Task error  (cm): ", ave_task_error_all * 100)

        ave_result = np.array([success,  ave_task_time, ave_task_error_success * 100, ave_task_error_all * 100])
        np.save(project_dir + "results/" + env + "/control/" + name + "/" + env_dim + "/ave.npy", ave_result)

        k += 1



# -------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    methods = ['ours']
    evaluateControlResults(methods)