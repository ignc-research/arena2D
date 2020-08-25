# arena2D-agent


## Training
The training with a given agent script can be started with the command `run_agent <agent_script>`. You can pass several optional parameters to this command:
* `--device <device_name>`: Specify a device to perform training on, mostly `cpu` or `cuda` (parameter `device_name` in `__init__()`, see *Agent Implementation*).
* `--model <model_name>`: Specifies a path to a model file to be loaded as initial model weights by the agent (parameter `model_name` in `__init__()`, see *Agent Implementation*).
* `--no_record`: If this flag is set then no training folder is created and no metrics are recorded for later evaluation.

Upon starting a training session, the simulator will create a folder (if flag `--no_record` is not set) at the current location called `training_X/` with `X` being the time and date the training was started.
For reproducibility purposes the agent script and the current settings file are copied to the training folder along with some other evaluation scripts.
During training several metrics (e.g. success rate) are recorded to the file `data.csv` in the training folder.
These metrics can be monitored with tensorboard by running the `monitor.py` script from the training folder.
When training is done a summary file will be created containing some basic information about the training session.
The metrics can then be plotted easily by running the `plot.py` script from the training folder.

While the training is running you should disable the video mode by pressing `F3` (if you are starting arena2d with GUI enabled).
This will stop the rendering and remove the FPS lock to ensure maximum performance.

## Agent Implementation
Agents are realized through a python class containing several methods that are called by the simulator during training.
When executing the command ```run_training <py-script>``` a training session will be started using the callback functions defined in the given python-script.
These functions must be defined in a class called ```Agent``` (can be changed in settings file with option ```training.agent_class```).
The simulator will automatically create an instance of the ```Agent```-class and call its functions during the training:
* `def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path)`: Constructor, called once before training starts.
	* `device_name`: String defining the device to use for training (mostly 'cpu' or 'cuda'), passed to the agent with the option `--device <device_name>`. Default: `cpu`.
	* `model_name`: Path of model to load from file and to initialize the net with, set to `None` if not specified by user with option `--model <model_name>`.
	* `num_observations`: Number of scalar values per environment passed to pre_step/post_step functions.
	* `num_envs`: Number of parallel environments in the simulator.
	* `num_threads`: Number of threads (cpu cores) the simulator was initilized with.
	* `training_data_path`: Path to a folder (ending with '/') were all the training data is stored (plots, settings, etc.) by the simulator. You should store any data created by this agent during training (e.g. model weights) at this location.
* `def pre_step(self, observations)`: This function is called before each simulation step.
	* `observations`: A list containing the observations from the simulator `[distance, angle, laser0, ..., laserN-1, additional_data0, ..., additional_dataN-1]`. If more than one environment is used, observations is a list containing a list for each environment.
	* `return`: Return a single action that is performed accross all environments or a list of actions (dedicated action for each environment). An action is either a single integer referring to a specific action type: `0:FORWARD`, `1:LEFT`, `2:RIGHT`, `3:STRONG_LEFT`, `4:STRONG_RIGHT`, `5:BACKWARD`, `6:STOP` (discrete action space) or a tuple containing two float numbers (linear_velocity, angular_velocity) (continuous action space).
* `def post_step(self, new_observations, rewards, dones, mean_reward, mean_success)`: This function is called after each simulation step.
	* `new_observations`: A list containing the new observations (same type as in function `pre_step()`)
	* `rewards`: The rewards that have been received in each environment, single value (one environment) or list (multiple environments).
	* `dones`: 1 if episode is over, 0 if episode is not finished yet, single value (one environment) or list (multiple environments).
	* `mean_reward`: Scalar value, mean reward from last 100 episodes accross all environments.
	* `mean_success` Scalar value, mean success rate from last 100 episodes accross all environments.
	* `return`: Return 0 to continue training normally, 1 to stop training, -1 to continue training but environments are not reset if episodes are done.
* `def get_stats(self)`: This function is called on the end of every episode and can be used to return metrics to be recorded by the simulator for later evaluation.
	* `return`: Return list of tuples `(name, value)`, an empty list or None. The value must be of type float or int, the name must be a string. Do not use `,` in the name for as this symbol is used as delimiter in the output csv file. Also please make sure to not change the order or number of metrics in the returned list between multiple calls to `get_stats()`.
* `def stop(self)`: Called when training has been stopped by the user in the simulator or by the agent (return 1 in function `post_step()`).
	* `return`: Optionally a string can be returned to be written to the results-file.

Please have a look at the [random agent](../arena2d-agents/random/agent.py) for more implementation details.

## Observation Data
The observations passed to the agent through `pre_step()`/`post_step()` functions consist of three parts:
* Distance and angle to the goal
* Laser scan data
* Additional data

The values in the observation list are packed as follows:
```
[distance, angle, laser0, ..., laserN-1, additional_data0, ..., additional_dataN-1]
```

### Distance and Angle
The distance and angle is measured from the robot to the goal in meters and degree.
The angle `(-180, +180]` is calculated from the current robot front direction to the vector pointing from robot to goal with a positive angle in CCW (counter clockwise) direction. So if the robot is looking up and the goal is on the right, the angle to the goal will be negative. Keep in mind that the angle has a discontinuity at 180 degrees (goal is directly behind the robot), at that point a very small change in the robot position can flip the sign of the angle.

### Laser Scan Data
Initially and after each simulation step a laser scan is performed by the simulator, retrieving a fixed number of distance samples to nearby obstacles around the robot in meters. 
The number of samples, maximum scan distance, as well as the start/end angle can be specified in the settings file in the `robot` section.
The distance samples are clamped to the maximum scan distance.
The start angle is the angle relative to the robot front direction (CCW) of the first laser sample. Similarly the end angle is the angle of the last laser sample.
With the setttings parameter `robot.laser_noise` a random noise can be applied to the laser samples. The `laser_noise` value is multiplied with the distance of each sample to determine a maximum +/- deviation of that sample. Example: `laser_noise = 0.1`, `laser_sampleX = 2.5`:  The resulting sample with noise applied will be an equally distributed random value in interval `2.25 <= laser_sampleX_noise <= 2.75`.

### Additional Data
You can specify additional observational data by overriding the function `getAgentData()` of the `Level`-class in your custom levels. The data can consist of any number of scalar float values. Additional data can be, for example, the velocity of moving obstacles, the safety distance to keep to certain objects, or the position of environmental markers.

## Network Design
you can explore different network architectures. Currently there are FC, CNN, LSTM implemented 
- what is input output of the networks, maybe some abstraction such that user can easily try out different combinations
- what to consider when making new architecture (input, output, interface between network and agent)
- interface between different agents
- example code


## Test and Evaluate Agents
To test and evaluate the trained models of your agent, run the agent_play.py scripts in the respective folders and specify the model path. We provide evaluation scripts in [evaluation_scripts](../arena2d-sim/scripts/)


