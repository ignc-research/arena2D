# rl-ros-agents(#TODO need a better name!)
## Introduction
rl-ros-agents is a package for training a local planner with the [stable-baseline](https://github.com/hill-a/stable-baselines) reinforcement learning approaches. It basically does the following two things:
    1. Defined the a environment wrapper which use ros messages to communicate with our arena simultor 
    2. Providing some handy scripts for training
![Working manner](/../img/Working_manner_rl_ros_agent.png)
When the training get started, seveal instances of the environment wrapper will be created and each runs on a single process. The number of the environment and other parameters are loaded from the ROS parameter server, which are register by the arena simulator. For each environment a pair of request message and response message will be created. For example if we create four environment, they may look like this:
```
/arena2d/env_0/request
/arena2d/env_0/response
/arena2d/env_1/request
/arena2d/env_1/response
/arena2d/env_2/request
/arena2d/env_2/response
/arena2d/env_3/request
/arena2d/env_3/response
```
The definitions of the request and response message can be found [here](../arena2d_msgs/msg). In each training step all environments will send the request in parallel and will be blocked util response messages are received. The main thread in the arena simulator will firstly collect the request messages and send the response messages back after finishing the updates.

## Prerequisites
1. Standard ROS setup. The link can be found [here](http://wiki.ros.org/melodic/Installation/Ubuntu) (Code has been tested with Melodic on Ubuntu 18.04)
2. Python3 and dependencies. its recommended to use conda to install necessary packages listed in the `envoronment.yml`.
    ```
     conda env create -f environment.yml
    ```
## Building
1. Create a catkin workspace:
    ```bash
    $ mkdir -p ~/catkin_ws_foo/src
    $ cd ~/catkin_ws_foo/
    $ rosws update
    ```
2. Clone this repository in the src-folder of your catkin workspace and checkout the branch `arena-ros`. Compile tehe code and add the workspace to the ROS environment with:
    ```bash
    $ catkin_make -DSUPPORT_ROS_AGENT=ON
    $ source devel/setup.bash
    ```
3. Make sure `rl_ros_agents`is included in the PYTHONPATH. Append following command to your `.bashrc`:
    ```bash
    $ export PYTHONPATH=${path_to_this_package}/rl_ros_agents:${PYTHONPATH}
    ```
## Useage
---
#### Training
1. open the simulator by:
    ```bash
    $ roslaunch arena2d arena_sim_video_off.launch
    OR
    $ roslaunch arena2d arena_sim_video_on.launch 
    ```
    Turning the Video on or off have no significant difference in the training speed.
2. running the training script:
    ```bash
    $ conda activate arena2d
    $ python scripts/train_a3c.py
    ```