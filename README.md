# shape_control_DLO_2

[Project website](https://mingrui-yu.github.io/shape_control_DLO_2/)

Repository for the IEEE T-RO Paper "Global Model Learning for Large Deformation Control of Elastic Deformable Linear Objects: An Efficient and Adaptive Approach".

Here we provide:
* the code for the model learning and controller
* the offline training data
* the offline learned deformation model
* the built simulation environment


## Dependencies
* Ubuntu 18.04
* ROS Melodic
* Nvidia driver & CUDA
* PyTorch in python3 env
* [Unity](https://unity.com/) for Linux 2020.03
* [Obi](http://obi.virtualmethodstudio.com/): for simulating the DLOs
* [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents): for the communication between the Unity and Python scripts
* [PyTorch-Radial-Basis-Function-Layer](https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer): we use the code for the implementation of RBFN in pytorch

## Installation

Install ROS Melodic on Ubuntu 18.04.

Install Unity for Linux 2020.03 [doc](https://docs.unity3d.com/2020.2/Documentation/Manual/GettingStartedInstallingHub.html).

Install Unity ML-Agents Toolkit [doc](https://github.com/Unity-Technologies/ml-agents/blob/release_18_docs/docs/Installation.md).

Install the following dependeces in your python3 env:
```
# about pytorch
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# about ROS
pip install rospkg

# about Unity ML-Agents
pip install mlagents==0.27.0
pip install gym
pip install gym_unity

# others
pip install numpy
pip install matplotlib
pip install sklearn
pip install empy
pip install PyYAML
pip install scipy
```

Clone the repo:
```
git clone https://github.com/Mingrui-Yu/shape_control_DLO_2.git
```

Build the catkin workspaces:

```
cd <YOUR_PATH>/shape_control_DLO_2/ws_dlo
catkin_make
```

Change the variable "project_dir" in *ws_dlo/src/dlo_system_pkg/config/params_sim.yaml* to '<YOUR_PATH>/shape_control_DLO_2/'.

## Usage

Give permissions to the simulation environment:

```
chmod -R 755 <YOUR_PATH>/shape_control_DLO/env_dlo/env_2D
chmod -R 755 <YOUR_PATH>/shape_control_DLO/env_dlo/env_3D
```

### Parameter Setting

Modifiable parameters in *dlo_system_pkg/config/params_sim.yaml*:

* "project dir": change it to your path to the project.
* "env/dimension": '2D' or '3D'.
* "controller/offline_model": we provide three pre-trained offline models: '10\*0.2', '10\*1' and '10\*6'. The numbers refer to the training data amount.
* "controller/online_learning/learning_rate": the online learning rate, corresponding to $\eta$ in the paper.

### Shape Control Tasks

Source the workspace:

```
cd <YOUR_PATH>/shape_control_DLO_2/ws_dlo
source devel/setup.bash
```

Upload the ROS params:
```
roslaunch dlo_system_pkg upload_sim_params.launch
```

Then, **activate your python3 env**, and run:

```
rosrun dlo_manipulation_pkg sim_env.py
```

After the running, **activate your python3 env** and run the following script to evaluate the performance:
```
rosrun dlo_manipulation_pkg control_results_compare.py
```

### Offline Train the Jacobian model
Upload the ROS params:
```
roslaunch dlo_system_pkg upload_sim_params.launch
```

Then, **activate your python3 env**, and run:

```
rosrun dlo_manipulation_pkg RBF.py
```

To change the training dataset, you can change the code in '\_\_main\_\_' of *dlo_manipulation_pkg/scripts/RBF.py*.

## Details

### Built Simulation Environment

The simulation environment is an executable file built by Unity. In both 2D and 3D environment, the ends of the DLO are grasped by two grippers which can translate and rotate.

Both the control rate and the data collection rate are 10 Hz.

The manipulated DLO is with a length of 0.5m and a diameter of 1cm.

#### Coordinate

We use a standard right-hand coordinate in the simulation environment.

In 2D tasks, the x axis is towards the bottom of the screen. The y axis is towards the right of the screen. The z axis is towards the outside of the screen.

In 3D tasks, the x axis is towards the outside of the screen. The y axis is towards the right of the screen. The z axis is towards the top of the screen.

Actually, the coordinates in 2D and 3D environment are the same. Only the position of the camera is changed.

#### Feature points along DLOs

Ten features are uniformly distributed along the DLO, which are represented by blue points. Note that the first and last features are the left and right ends, so they actually represent the positions of the robot end-effectors.

#### Desired Shape

The desired shape is represented by the desired positions of 10 features. All desired shapes are chosen from the training dataset, so they are ensured to be feasible. The 100 2D desired shapes are stored in *env_dlo/env_2D_Data/StreamingAssets/desired_shape/2D/desired_positions.npy* and the 100 3D desired shapes are stored in *env_dlo/env_2D_Data/StreamingAssets/desired_shape/3D/desired_positions.npy* .

Note that in control tasks, only the desired positions of the internal 8 features are considered (set as the target points).

#### State

The state is a 117-dimension vector.

- 0: the length of the DLO
- 1~30: the positions of the 10 features (10*3)
- 31~44: the pose of the two end-effectors
  - left end positions (3) + left end orientation (4) + right end position (3) + right end orientation (4)
  - The representation of the orientations is quaternion.
- 45~74: the velocities of the 10 features (10*3)
- 75~86: the velocities of the end-effectors
  - left end linear velocity (3) + left end angular velocity (3) + right end linear velocity (3) + right end angular velocity (3)
  - The representation of the angular velocities is rotation vector.
- 87~116: the desired positions of the 10 features (10*3)

Note that in 2D environment, the dimension of the position of one feature is still three, but the value in the z axis is always zero.

#### Action

The action is a 12-dimension vector.

- 0~2: the linear velocity of the left end effector
- 3~5: the angular velocity of the left end effector
- 6~8: the linear velocity of the right end effector
- 9~11: the angular velocity of the right end effector

In our implementation, the action is formulated as the above format in both 2D and 3D environment. Thus, in 2D tasks, we need to output valid control input in the controller script, where the [2, 3, 4, 8, 9, 10] dimension of the control input must be zero.

### Training Dataset

Our offline collected data are in *shape_control_DLO/data/train_data*.

- state_0.py: the data of the manipulated DLO. Data amount: 60k.
- state_1.npy ~ state_10.npy: the data of 10 different DLOs. Data amount: 6k for each DLO. The parameters of the DLOs are listed in the paper.

The format of the training dataset is (-1, 117). Each row is a 117-dimension state vector.

## Contact

If you have any question, feel free to raise an issue (recommended) or contact the authors: Mingrui Yu, [ymr20@mails.tsinghua.edu.cn](mailto:ymr20@mails.tsinghua.edu.cn)