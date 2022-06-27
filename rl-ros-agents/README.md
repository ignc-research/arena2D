# rl-ros-agents(#TODO need a better name!)
## Introduction
rl-ros-agents is a package for training a local planner with the [stable-baseline](https://github.com/hill-a/stable-baselines) reinforcement learning approaches. It basically does the following two things:
    1. Defined the a environment wrapper which use ros messages to communicate with our arena simultor 
    2. Providing some handy scripts for training
![Working manner](/img/Working_manner_rl_ros_agent.png)
When the training get started, seveal instances of the environment wrapper will be created and each runs on a single process. The number of the environment and other parameters are loaded from the ROS parameter server, which are register by the arena simulator. Which means if number of the environments you want to use need to be changed, you need to modify the param **num_envs** in the file `settings.st` in the package *arena2d-sim*  .For each environment a pair of request message and response message will be created. For example if we create four environment, they may look like this:
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
   - Standard ROS setup. The link can be found [here](http://wiki.ros.org/noetic/Installation/Ubuntu) (currently we use ubuntu 20.04 and ros-noetic).
   - install conda and necessary dependencies:
   1. install conda see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
   2. `$ sudo apt-get install cmake libsdl2-dev libfreetype-dev`
   3. in this repository `$ conda env create -f environment.yml`

## Building
1. Create a catkin workspace:
    ```
    $ mkdir -p ~/ARENA2d_ws/src
    $ cd ~/ARENA2d_ws/
    ```
2. Clone this repository in the src-folder of your catkin workspace and checkout the branch `arena-ros`.  
 
    Compile the code and add the workspace to the ROS environment with:
    ```
    $ cd ~/ARENA2d_ws/src
    $ git clone https://github.com/ignc-research/arena2D.git # this repository
    $ cd arena2D
    $ git checkout arena-ros 

    you will see this:

    Branch 'arena-ros' set up to track remote branch 'arena-ros' from 'origin'.
    Switched to a new branch 'arena-ros'

    $ cd ~/ARENA2d_ws
    $ pip install empy
	$ vim vim ~/.bashrc
	$ export PYTHONPATH=$PYTHONPATH:~/anaconda3/envs/arena2d/lib/python3.6/site-packages # copy this in the bottom, then use wq! to quit
	$ source ~/.bashrc
    $ catkin_make -DUSE_ROS=ON
    $ source devel/setup.bash
    ```
    You may encounter some compilation problems when using `catkin_make`, e.g.:
   ```
   $ /usr/bin/ld: /lib/x86_64-linux-gnu/libapr-1.so.0: undefined reference to `uuid_generate@UUID_1.0'
   ```
   to fix that, you can use following commands:
   ```
   $ ls ~/anaconda3/lib/libuuid*
   $ mkdir ~/anaconda3/libuuid
   $ mv ~/anaconda3/lib/libuuid* ~/anaconda3/libuuid
   $ catkin_make -DUSE_ROS=ON
   ```

3. Make sure rl_ros_agents and further dependences is included in the PYTHONPATH.
    ```
    $ vim ~/.bashrc
    $ export PYTHONPATH=~/ARENA2d_ws/src/arena2D/rl-ros-agents:${PYTHONPATH} # copy this in the bottom, then use wq! to quit
    $ source ~/.bashrc
    ```
	You can check if your pythonpath is correct by using `echo $PYTHONPATH`.
    
## Additional work (not necessary)
Offically ros packages are only built for python2. In some cases mixing useage of python3 with python2 may cause problems. Therefore we recommended building two more packages specifically.
1. Create a python3 workspace:
    ```bash
    $ mkdir -p ~/python3_ws/src
    $ cd ~/python3_ws/src
    $ git clone --depth=1 https://github.com/ros/geometry.git
    $ git clone --depth=1 https://github.com/ros/geometry2.git
    ``` 
2. When this README document is created, there is a duplicate name error. If you have the same issue, please solve it 
refer to the methods introduced [here](https://github.com/ros/geometry/issues/213#issuecomment-643552794)
3. Activate your virtual environment. e.g
    ```bash
    $ conda activate arena2d
    ```
4. complile the workspace:
    ```bash
    $ catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$(which python)
    ```
5. append following command to the `.bashrc`file under your HOME folder.
    ```bash
    $ source ~/python3_ws/devel/setup.bash
    ```

## Training

1. open a terminal and run the simulator:  
    ```
    $ cd ~/ARENA2d_ws
    $ source devel/setup.bash
    $ roslaunch arena2d arena_sim_video_off.launch
    OR
    $ roslaunch arena2d arena_sim_video_on.launch 
    ```
- Turning the Video on or off have no significant difference in the training speed.
- Use `robot_model:=<model_name> robot_mode:=<mode_name>` to choose robot and action mode  
e.g `roslaunch arena2d arena_sim_video_on.launch robot_model:=burger robot_mode:=continuous`  
(default: burger + continuous)

2. open a new terminal and run the training script:
    ```
    $ cd ~/ARENA2d_ws
    $ source devel/setup.bash
    $ cd ~/ARENA2d_ws/src/arena2D/rl-ros-agents
    $ conda activate arena2d
    $ python scripts/training/train_a3c.py
    ```

3. open a new termial and visualize robot position & laserscan:
    ```
    $ cd ~/ARENA2d_ws
    $ source devel/setup.bash
    $ cd ~/ARENA2d_ws/src/arena2D/rl-ros-agents
    $ pip install rospkg
    $ python scripts/rviz_visualize_helper.py
    ```

4. open the rviz and start it with a config file
    ```
    $ rviz -d launch/rviz_config_new.rviz
    ```
