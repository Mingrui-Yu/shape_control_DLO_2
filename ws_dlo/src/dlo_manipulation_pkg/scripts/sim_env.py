#!/usr/bin/env python

# main node of the simulation
# bridge between the Unity Simulator and the controller scripts, based on 'mlagents' and 'gym'

import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import rospy
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import gym
from gym_unity.envs import UnityToGymWrapper
from utils.state_index import I
from geometry_msgs.msg import Vector3

control_method = rospy.get_param("controller/control_law")
if control_method == 'ours':
    from controller_ours import Controller



class Environment(object):
    def __init__(self):
        self.project_dir = rospy.get_param("project_dir")
        env_dim = rospy.get_param("env/dimension")
        self.num_fps = rospy.get_param("DLO/num_FPs")

        engine_config_channel = EngineConfigurationChannel()
        env_params_channel = EnvironmentParametersChannel()

        # use the built Unity environment
        env_file = self.project_dir + "env_dlo/env_" + env_dim
        unity_env = UnityEnvironment(file_name=env_file, seed=1, side_channels=[engine_config_channel, env_params_channel])
        engine_config_channel.set_configuration_parameters(width=640, height=360, time_scale=2.0)  # speed x2
        
        self.env = UnityToGymWrapper(unity_env)
        self.controller = Controller()
        self.control_input = np.zeros((12, ))



    
    # -------------------------------------------------------------------
    def mainLoop(self):
        # the first second in unity is not stable, so we do nothing in the first second
        for k in range(10):
            state, reward, done, _ = self.env.step(self.control_input)
            state[I.left_end_avel_idx + I.right_end_avel_idx] /= 2*np.pi  # change the unit of the input angular velocity from rad/s  to 2pi*rad/s

        while not rospy.is_shutdown():
            self.control_input = self.controller.generateControlInput(state).copy()
            self.control_input[[3, 4, 5, 9, 10, 11]] *= 2*np.pi  # change the unit of the output angular velocity from 2pi*rad/s  torad/s

            state, reward, done, _ = self.env.step(self.control_input)
            state[I.left_end_avel_idx + I.right_end_avel_idx] /= 2*np.pi # change the unit of the input angular velocity from rad/s  to 2pi*rad/s

            if done: # Time up (30s), the env and the controller are reset. Next case with different desired shapes.
                self.controller.reset(state)
                state = self.env.reset()


  

# --------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        rospy.init_node("sim_env_node")
        env = Environment()
        env.mainLoop()

    except rospy.ROSInterruptException:
        print("program interrupted before completion.")