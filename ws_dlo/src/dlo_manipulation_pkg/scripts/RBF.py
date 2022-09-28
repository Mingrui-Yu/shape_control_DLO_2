#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import os
import time
from sklearn.cluster import KMeans
import copy
import rospy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as sciR

import torch_rbf as rbf # reference: https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer

from utils.data_augmentation import dataRandomTransform 
from utils.state_index import I

params_online_window_time = 2  # unit: second
params_online_max_valid_fps_vel = 0.3
params_online_fps_vel_thres = 0.01
params_online_min_valid_fps_vel = 0.00
params_update_if_window_full = False


# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
class NNDataset(Dataset):
    def __init__(self, state):
        self.data_num = state.shape[0]
        self.length = state[:, I.length_idx]
        self.state_input = state[:, I.state_input_idx]
        self.fps_vel = state[:, I.fps_vel_idx]
        self.ends_vel = state[:, I.ends_vel_idx]
    
    def __getitem__(self, index):
        return self.length[index], self.state_input[index], self.fps_vel[index], self.ends_vel[index]

    def __len__(self):
        return self.data_num



# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
class Net_J(nn.Module):
    def __init__(self, nFPs, bTrainMuBeta, num_hidden_unit):
        super(Net_J, self).__init__()
        self.nFPs = nFPs
        self.numHidden = num_hidden_unit
        lw = [3*self.nFPs + 3+4+4, self.numHidden, (nFPs * 3) * 12]
        basis_func = rbf.gaussian

        self.fc1 = rbf.RBF(lw[0], lw[1], basis_func, bTrainMuBeta=bTrainMuBeta)
        self.fc2 = nn.Linear(lw[1], lw[2], bias=False)


    def forward(self, x):
        theta = (self.fc1(x))
        output = (self.fc2(theta)) 
        output = torch.reshape(torch.reshape(output, (-1, self.nFPs, 12, 3)).transpose(2, 3), (-1, 3 * self.nFPs, 12)) # J: dimension: 30 * 12
        return output

    # use kmeans to calculate the initial value of mu and sigma in RBFN
    def GetMuAndBetaByKMeans(self, full_data):
        max_data_size = 600 * 60
        if(full_data.shape[0] > max_data_size): 
            # randomly choose a subset of train data for kmeans
            index = np.random.choice(np.arange(full_data.shape[0]), size=max_data_size, replace=False)
            data = full_data[index, :]
        else:
            data = full_data

        print("start kmeans ... ")
        kmeans = KMeans(n_clusters=self.numHidden, n_init=2, max_iter=100).fit(data)
        print("finish kmeans ... ")
        nSamples = np.zeros((self.numHidden, ), dtype='float32')
        variance = np.zeros((self.numHidden, ), dtype='float32')
        for i, label in enumerate(kmeans.labels_):
            variance[label] += np.linalg.norm(data[i, :] - kmeans.cluster_centers_[label, :])**2
            nSamples[label] += 1
        variance = variance / nSamples
        sigma = np.sqrt(variance) * np.sqrt(2) * 10 #  mannually set initial value which is better for the following training
        invSigma = np.clip(invSigma, 0, 1)

        self.fc1.centres.data = torch.tensor(kmeans.cluster_centers_).cuda()
        self.fc1.sigmas.data = torch.tensor(invSigma).cuda()



# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
class JacobianPredictor(object):

    numFPs = rospy.get_param("DLO/num_FPs")
    projectDir = rospy.get_param("project_dir")
    online_learning_rate = rospy.get_param("controller/online_learning/learning_rate")
    lr_task_e = online_learning_rate
    lr_approx_e = online_learning_rate * rospy.get_param("controller/online_learning/weight_ratio")
    env = rospy.get_param("env/sim_or_real")
    env_dim = rospy.get_param("env/dimension")
    control_rate = rospy.get_param("ros_rate/env_rate")
    online_update_rate = rospy.get_param("ros_rate/online_update_rate")
    
    # ------------------------------------------------------
    def __init__(self, num_hidden_unit=256):
        device = torch.device("cuda:0")
        
        self.bTrainMuBeta = True
        self.model_J = Net_J(self.numFPs, self.bTrainMuBeta, num_hidden_unit).to(device)
        # for offline learning
        learningRate = 0.01
        self.optimizer = torch.optim.Adam([{'params': self.model_J.parameters()}], learningRate)
        torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99, last_epoch=-1, verbose=False)
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=0.001)
        # for online learning
        self.online_optimizer = torch.optim.SGD([{'params': self.model_J.fc2.parameters()}], lr=1.0/self.online_update_rate)
        self.mse_criterion = torch.nn.MSELoss(reduction='sum')

        if rospy.get_param("learning/is_test"):
            self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models_test/rbfWeights/' + self.env_dim + '/'
        else:
            self.nnWeightDir = self.projectDir + 'ws_dlo/src/dlo_manipulation_pkg/models/rbfWeights/' + self.env_dim + '/'
        self.resultsDir = self.projectDir + 'results/' + self.env + '/'
        self.dataDir = self.projectDir +'data/'

        self.online_dataset = []

        
    # ------------------------------------------------------
    def LoadDataForTraining(self, train_dataset=None):
        # trainset
        if train_dataset is None:
            train_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state_0.npy').astype(np.float32)[600*2 : 600*10, :]
        self.trainDataset = NNDataset(train_dataset.astype(np.float32))
        self.trainDataLoader = DataLoader(self.trainDataset, batch_size=512, shuffle=True, num_workers=4)

    # ------------------------------------------------------
    def LoadDataForTest(self, test_dataset=None):
        # testset
        if test_dataset is None:
            test_dataset = np.load(self.dataDir + 'train_data/' + self.env_dim + '/state_0.npy').astype(np.float32)[600*0 : 600*2, :]
        self.testDataset = NNDataset(test_dataset.astype(np.float32))
        self.testDataLoader = DataLoader(self.testDataset, batch_size=test_dataset.shape[0], shuffle=False, num_workers=4)


    # ------------------------------------------------------
    def LoadModelWeights(self, file=None):
        if file is not None:
            if os.path.exists(self.nnWeightDir  + "/" + file):
                self.model_J.load_state_dict(torch.load(self.nnWeightDir  + "/" + file))
                # print('Load previous model.')
            else:
                print('Warning: no model exists !')
        else:
            offline_model = rospy.get_param("controller/offline_model")
            if rospy.get_param("learning/is_test"):
                if os.path.exists(self.nnWeightDir  + "/model_J.pth"):
                    self.model_J.load_state_dict(torch.load(self.nnWeightDir + "/model_J.pth"))
                    # print('Load previous model.')
                else:
                    print('Warning: no model exists !')
            else:
                if os.path.exists(self.nnWeightDir + offline_model + "/model_J.pth"):
                    self.model_J.load_state_dict(torch.load(self.nnWeightDir + offline_model + "/model_J.pth"))
                    # print('Load previous model.')
                else:
                    print('Warning: no model exists !')

            self.n_count = 0
            self.online_dataset = []

    
    # ------------------------------------------------------
    def SaveModelWeights(self):
        torch.save(self.model_J.state_dict(), self.nnWeightDir + "model_J.pth")
        # print("Save models to ", self.nnWeightDir)

    
    # ------------------------------------------------------
    def Train(self, loadPreModel=False, n_epoch=50, save_model=True, rotation_augmentation=True):

        if loadPreModel == False:  # use kmeans to calculate the initial value of mean and sigma of gaussian kernels
            if rotation_augmentation:
                self.model_J.GetMuAndBetaByKMeans(
                        self.relativeStateRepresentationTorch(dataRandomTransform(self.trainDataset.state_input)))
            else:
                self.model_J.GetMuAndBetaByKMeans(
                        self.relativeStateRepresentationTorch(self.trainDataset.state_input))
        else:
            self.LoadModelWeights()

        # training
        for epoch in range(0, n_epoch):
            accumLoss = 0.0
            numBatch = 0
            for batch_idx, (length, state_input, fps_vel, ends_vel) in enumerate(self.trainDataLoader):       
                # data augmentation
                if rotation_augmentation:
                    state_input, fps_vel, ends_vel = dataRandomTransform(state_input, fps_vel, ends_vel)

                state_input = self.relativeStateRepresentationTorch(state_input)

                # normalization
                ends_vel /= (torch.linalg.norm(fps_vel, dim=1).unsqueeze(1) + 1e-8)
                fps_vel /= (torch.linalg.norm(fps_vel, dim=1).unsqueeze(1) + 1e-8) # avoid division by zero

                # data to GPU
                length = length.cuda()
                state_input = state_input.cuda()
                fps_vel = fps_vel.cuda()
                ends_vel = ends_vel.cuda()
                
                bmm_ends_vel = torch.reshape(ends_vel, (-1, 1, 12))
                bmm_fps_vel = torch.reshape(fps_vel, (-1, 1, self.numFPs * 3))

                J_pred = self.model_J(state_input)
                J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.reshape(-1, 1, 1)  # N * T to get the final Jacobian

                J_pred_T = J_pred.transpose(1, 2)
                loss = self.criterion(bmm_fps_vel, torch.bmm(bmm_ends_vel, J_pred_T))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                accumLoss += loss.item()
                numBatch += 1

            print("epoch: ", epoch, " , Loss/train: ", accumLoss/numBatch)

        # save model
        if save_model:
            self.SaveModelWeights()


    # ------------------------------------------------------
    def TestAndSaveResults(self):

        self.LoadModelWeights("model_J.pth")

        accumLoss = 0.0
        numBatch = 0
        for batch_idx, (length, state_input, fps_vel, ends_vel) in enumerate(self.testDataLoader):
            state_input = self.relativeStateRepresentationTorch(state_input)

            length = length.cuda()
            state_input = state_input.cuda()
            fps_vel = fps_vel.cuda()
            ends_vel = ends_vel.cuda()
            
            bmm_ends_vel = torch.reshape(ends_vel, (-1, 1, 12))
            bmm_fps_vel = torch.reshape(fps_vel, (-1, 1, self.numFPs * 3))

            J_pred = self.model_J(state_input)
            J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length.reshape(-1, 1, 1) 

            J_pred_T = J_pred.transpose(1, 2)
            fps_vel_pred = torch.bmm(bmm_ends_vel, J_pred_T)

            testLoss = self.mse_criterion(fps_vel_pred, bmm_fps_vel)

            accumLoss += testLoss.item()
            numBatch += 1

        print("Loss/test: ", accumLoss/numBatch)

        # test result 数据保存
        np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_truth.npy", bmm_fps_vel.cpu().detach().numpy()) 
        np.save(self.resultsDir + "nn_test/rbf/" + self.env_dim + "/dot_x_pred.npy", fps_vel_pred.cpu().detach().numpy())



    # ------------------------------------------------------
    def OnlineLearningAndPredictJ(self, state, task_error=None):
        state = copy.copy(state)
        task_error = copy.copy(task_error)
        
        self.n_count  += 1
        # parameters
        window_size = params_online_window_time * self.control_rate

        length = state[I.length_idx]
        state_input = state[I.state_input_idx].reshape(1, -1) # one row matrix
        if task_error is None:
            task_error = np.zeros((self.numFPs * 3, ), dtype='float32')

        # Because of the imperfection of the simulator, sometimes the DLO will wiggle to the other side very fast.
        # We don't want to include these outlier data in training, so we just discard the online data with too fast speed.
        fps_vel_norm = np.linalg.norm(state[I.fps_vel_idx])
        if fps_vel_norm > params_online_max_valid_fps_vel or  fps_vel_norm < params_online_min_valid_fps_vel:
            # return the Jacobian without online updating
            length_torch = torch.tensor(length).cuda()
            state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).cuda()
            J_pred = self.model_J(state_input_torch)
            J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)  # no normalize 要改 # RBF_abs 要改
            return J_pred.cpu().detach().numpy().reshape(3 * self.numFPs, 12)

        # normalize the velocities
        elif fps_vel_norm > params_online_fps_vel_thres:
            state[I.fps_vel_idx]  /= (fps_vel_norm)
            state[I.ends_vel_idx]  /= (fps_vel_norm)
            task_error *= (fps_vel_norm)
        else:
            state[I.fps_vel_idx] /= params_online_fps_vel_thres
            state[I.ends_vel_idx]  /= params_online_fps_vel_thres
            task_error *= (params_online_fps_vel_thres) 

        fps_vel = state[I.fps_vel_idx]
        ends_vel = state[I.ends_vel_idx]
            
        # --------------------------------------------------
        # update the NN weights
        # we use the SGD optimizer in PyTorch for online learning implementation to achieve faster computing speed. 
        # Note that the following computing is mathematically equivalent to the online updating law in the paper.

        # learning rate: transform the learning rates  to the weights in the loss function
        weight_approx_e = np.sqrt(self.lr_approx_e /  window_size / 2)
        if weight_approx_e == 0:
            weight_task_e = 0
        else:
            weight_task_e = self.lr_task_e / 2 /  weight_approx_e

        # data preparation
        # latest step data
        length_torch = torch.tensor(length).cuda()
        state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).cuda()
        ends_vel_torch = torch.reshape(torch.tensor(ends_vel), (1, 1, 12)).cuda()
        fps_vel_torch = torch.reshape(torch.tensor(fps_vel), (1, 1, 3 * self.numFPs)).cuda()
        task_error_torch = torch.reshape(torch.tensor(task_error), (1, 1, 3 * self.numFPs)).cuda()

        # previous data in sliding window
        if len(self.online_dataset) > 1:
            online_dataset = np.array(self.online_dataset)
            length_batch = online_dataset[:, I.length_idx]
            state_input_batch = online_dataset[:, I.state_input_idx]
            if len(state_input_batch.shape) == 1:
                state_input_batch = state_input_batch.reshape(1, -1)
            fps_vel_batch = online_dataset[:, I.fps_vel_idx]
            ends_vel_batch = online_dataset[:, I.ends_vel_idx]
            length_batch_torch = torch.tensor(length_batch).cuda()
            state_input_batch_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input_batch)).cuda()
            ends_vel_batch_torch = torch.reshape(torch.tensor(ends_vel_batch), (-1, 1, 12)).cuda()
            fps_vel_batch_torch = torch.reshape(torch.tensor(fps_vel_batch), (-1, 1, 3 * self.numFPs)).cuda()

        # updating
        if (params_update_if_window_full is False) or (len(self.online_dataset) == window_size - 1):

            for epoch in range(int(self.online_update_rate / self.control_rate)):

                self.online_optimizer.zero_grad()

                # data at the current time
                J_pred = self.model_J(state_input_torch)
                J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1) 
                J_pred_T = J_pred.transpose(1, 2)
                task_e = task_error_torch
                approx_e = fps_vel_torch - torch.bmm(ends_vel_torch, J_pred_T)
                loss = self.mse_criterion(weight_approx_e * approx_e  + weight_task_e * task_e,   torch.zeros(approx_e.shape).cuda())

                loss.backward()
                
                # previous data stored in the sliding window
                if len(self.online_dataset) > 1:
                    J_pred = self.model_J(state_input_batch_torch)
                    J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_batch_torch.reshape(-1, 1, 1)
                    J_pred_T = J_pred.transpose(1, 2)
                    loss =  self.lr_approx_e / window_size / 2 * self.mse_criterion(fps_vel_batch_torch, torch.bmm(ends_vel_batch_torch, J_pred_T))
                    
                    loss.backward()

                # do the update
                self.online_optimizer.step()
        # --------------------------------------------------

        # store the data at the current time in the sliding window
        self.online_dataset.append(state)
        # remove the earliest data in the sliding window
        if len(self.online_dataset) > window_size - 1:
            self.online_dataset.pop(0)

        # return the updated Jacobian matrix
        J_pred = self.model_J(state_input_torch)
        J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)

        return J_pred.cpu().detach().numpy().reshape(3 * self.numFPs, 12)


    # --------------------------------------------------------------------
    def calcNextEndsPose(self, current_ends_pose, ends_vel, delta_t=0.1):
        left_end_pos = current_ends_pose[:, 0:3]
        left_end_quat = current_ends_pose[:, 3:7]
        right_end_pos = current_ends_pose[:, 7:10]
        right_end_quat = current_ends_pose[:, 10:14]

        left_end_lvel = ends_vel[:, 0:3]
        left_end_avel = ends_vel[:, 3:6]
        right_end_lvel = ends_vel[:, 6:9]
        right_end_avel = ends_vel[:, 9:12]

        next_left_end_pos = left_end_pos + left_end_lvel * delta_t
        next_right_end_pos = right_end_pos + right_end_lvel * delta_t

        left_end_ori = sciR.from_quat(left_end_quat)
        left_end_delta_ori = sciR.from_rotvec(left_end_avel * delta_t)
        next_left_end_ori = left_end_delta_ori * left_end_ori
        next_left_end_quat = next_left_end_ori.as_quat()

        right_end_ori = sciR.from_quat(right_end_quat)
        right_end_delta_ori = sciR.from_rotvec(right_end_avel * delta_t)
        next_right_end_ori = right_end_delta_ori * right_end_ori
        next_right_end_quat = next_right_end_ori.as_quat()

        next_ends_pose = np.concatenate([next_left_end_pos, next_left_end_quat, next_right_end_pos, next_right_end_quat], axis=1)
        return next_ends_pose


    # ------------------------------------------------------
    def predNextFPsPositions(self, length, fps_pos, ends_pose, ends_vel, delta_t):
        if fps_pos.ndim == 1 or ends_pose.ndim == 1 or ends_vel.ndim==1:
            length = length.reshape(1, -1)
            fps_pos = fps_pos.reshape(1, -1)
            ends_pose = ends_pose.reshape(1, -1)
            ends_vel = ends_vel.reshape(1, -1)

        fps_vel_pred = self.predFPsVelocities(length, fps_pos, ends_pose, ends_vel)
        next_fps_pos = fps_pos + delta_t * fps_vel_pred
        return next_fps_pos


    # ------------------------------------------------------
    def predFPsVelocities(self, length, fps_pos, ends_pose, ends_vel):
        state_input = np.concatenate([fps_pos, ends_pose], axis=1).astype('float32') # one row matrix
        # np array to torch tensor
        state_input_torch = self.relativeStateRepresentationTorch(torch.tensor(state_input)).cuda()
        ends_vel_torch = torch.reshape(torch.tensor(ends_vel.astype('float32')), (-1, 1, 12)).cuda()
        length_torch = torch.tensor(length.astype('float32')).cuda()

        # predict the feature velocities
        J_pred = self.model_J(state_input_torch)
        J_pred[:, :, [3, 4, 5, 9, 10, 11]] *= length_torch.reshape(-1, 1, 1)

        J_pred_T = J_pred.transpose(1, 2)
        fps_vel_pred = torch.bmm(ends_vel_torch, J_pred_T).cpu().detach().numpy().reshape(-1, self.numFPs * 3)

        return fps_vel_pred


    # ------------------------------------------------------
    # state representation preprocess
    def relativeStateRepresentationTorch(self, state_input):
        b_numpy = False
        if type(state_input) is np.ndarray:
            state_input = torch.tensor(state_input)
            b_numpy = True

        numFPs = self.numFPs
        left_end_pos = state_input[:, 3*numFPs : 3*numFPs + 3]
        left_end_quat = state_input[:,  3*numFPs + 3 : 3*numFPs +7]
        right_end_pos = state_input[:,  3*numFPs + 7 : 3*numFPs +10]
        right_end_quat = state_input[:,  3*numFPs +10 : 3*numFPs +14]

        fps_pos = state_input[:, 0 : 3*numFPs].reshape(-1, numFPs, 3)
        if len(fps_pos.shape) == 1: # reshape fps_pos from vector to one-row matrix
            fps_pos = fps_pos.unsqueeze(0)

        fps_pos_r = torch.zeros(fps_pos.shape)
        fps_pos_r[:, 1:, :] = (fps_pos[:, 1:, :] - fps_pos[:, 0:-1, :]) 
        fps_pos_r[:, 1:, :] /= torch.linalg.norm(fps_pos_r[:, 1:, :], dim=2).unsqueeze(2)
        fps_pos_r = fps_pos_r.reshape(-1, 3*numFPs)
        
        right_end_pos_r = (right_end_pos - left_end_pos) 
        right_end_pos_r /= torch.linalg.norm(right_end_pos_r, dim=1).unsqueeze(1)

        relative_state_input = torch.cat((fps_pos_r, right_end_pos_r, left_end_quat, right_end_quat), dim=1)
        if b_numpy:
            return relative_state_input.numpy()
        else:
            return  relative_state_input
    


# --------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    project_dir = rospy.get_param("project_dir")
    env_dim = rospy.get_param("env/dimension")

    # training dataset (from 10 DLOs)
    train_dataset = np.empty((0, I.state_dim)).astype("float32")
    for j in range(1, 11):
        state = np.load(project_dir + "data/train_data/"+ env_dim + "/state_" + str(j) + ".npy").astype("float32")[: 6000, :]
        train_dataset = np.concatenate([train_dataset, state], axis=0)


    trainer = JacobianPredictor()
    trainer.LoadDataForTraining(train_dataset)
    trainer.Train(loadPreModel=False, n_epoch=100)

