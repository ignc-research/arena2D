# arena2d-sim
arena2d-sim is a simple yet efficient 2D simulator build on C++, with configurable agents, highly randomization of levels, sensors and handling of python-based DRL agents.


## Updates
* 01/06/2020:** including asynchronous training. It is now possible to run multiple training instances in parallel. See ... for template
* 01/07/2020:** including A3C agent, based on [Lapan et al. (2018)][lapan]
...

# Instructions

## Installation
Please follow the instruction from the main arena2d [README](../README.md) file for installation of arena2d.

## Commandline arguments
There are several commandline arguments that can be passed when launching the application.
* ```--help``` show help
* ```--disable-video``` disable window creation and video rendering (useful when starting the simulator over an ssh session)
* ```--run <command>``` run simulator commands separated by ```;```
* ```--logfile <file>``` log output to given file

For example
```
arena2d --logfile log.txt --disable-video --run "run_agent my_agent.py"
```
starts the simulator without GUI, then starts training with the agent script ```my_agent.py``` while logging all the output to the file ```log.txt```.

### Start Asynchronous Training from Commandline
To start asynchronous training, you have to specify at least 2 parallel number of environments (num_envs). When using the commandline, you can only change these parameters inside the settings.st file where you should set num_envs to a value greater then 1.
Afterwards run:
```
arena2d --disable-video --run "level "random" --dynamic; run_agent ../../arena2d-agents/A3C_LSTM/agent_train.py --device cuda"
```
to train an A3C with LSTM agent on the random, dynamic environment with GPU. For more options see the agent descriptions below. 
Currently, we provide the levels "empty", "random", and "svg". More information is provided below.

### Change training settings
Changing the training settings outside the settings.st file is only possible when using GUI in build command tool. If you are running this terminal-only, you have to change the settings inside the settings.st. Recompilation is not neccessary. 


## GUI Commands and in app console

## Controls
Hold down the right mouse button while moving the mouse to pan the view. Scroll to zoom in and out.
For testing purposes, the robot can be controlled with the arrow keys. The keys are mapped to the discrete actions as follows:
* ```UP``` - forward
* ```LEFT``` - strong left
* ```RIGHT``` - strong right
* ```LEFT + UP``` - left
* ```RIGHT + UP``` - right
* ```DOWN``` - backwards

The velocities of the discrete actions, as well as the key binding can be configured in the settings file. The user controls are blocked if an agent is running.

## Hot Keys
* ```F1```: open/close in-app console
* ```F2```: show/hide stats display showing several metrics such as the simulation time
* ```F3```: activate/deactivate video mode (deactivating video mode ensures maximum performance during training)
* ```F5```: take screenshot
* ```Ctrl+S```: save current settings to default settings file (```settings.st```)
* ```Ctrl+L```: load settings from default settings file 
* ```R```: reset stage (rebindable in settings file)
* ```SPACE```: run/pause simulation when training is currently inactive (this might be useful to test the behaviour of moving obstacles)

## Simulation
The 2D physics simulation is run when either:
* the user controls the robot manually
* an agent is running (command `run_agent`)
* or the simulation is set to run-mode by pressing `SPACE`

When the simulation is running with the GUI enabled, a fixed amount of steps is performed per second that can be set with the settings parameter `physics.fps` or with the command `fps physics <fps>`.
The time interval simulated in one step is determined by the time step `physics.time_step` and by the number of sub steps `physics.step_iterations`.
So if you have `physics.time_step = 0.1` and `physics.step_iterations = 5` each step simulates 0.5 seconds.
If you want to increase the overall time per step (which is also the time between two subsequent observations received by the agent) you should not increase `physics.time_step` for as the simulation might get inaccurate, but rather increase the number of sub steps `physics.step_iterations`.
To increase the simulation accuracy (which is not necessary in most cases) the parameters `physics.position_iterations` and `physics.velocity_iterations` can be adjusted.

## In-App console
The in-app console is used to send commands to the simulator at runtime.
Press ```F1``` to toggle the In-App console. Commands can be executed by hitting ```Enter```.
Use the up and down keys to browse the command history. The command history will be stored in a file called ```.arena_command_history.txt``` in your home directory.
The command ```help``` displays all available commands. Run ```help <command>``` to get more detailed information about a given command.
The In-App console is only available with the GUI enabled.
To run a command without the GUI pass the option ```--run <command>``` when launching the simulator from the console to specify commands that are executed on startup.
Here is a list of the most relevant commands:
* ```level <level_name>```: Load level with given name, e.g. *empty*, *random* (see section *Built-in Levels* for more details).
* ```run_agent <py-script> [--model <initial_model>] [--device <cpu/cuda>] [--no_record]```: Start training session using given agent script (see section *Training* for more details).
* ```stop_agent```: Stop current training session.
* ```set <settings-expression>```: Adjust current settings, e.g. ```set physics.time_step=0.01``` (note that some changes will only take effect on restart).

**NOTE:** You should not execute the command ```set``` during training, for as it might break the reproducibility and possibly makes the training unstable (e.g. when changing the simulation time step).

## Global Settings
The settings file `settings.st` contains global options that can be accessed by all software components in the simulator. When running the application for the first time a default settings file is created at the current location. The settings file is always loaded from the directory where arena2d is executed. The settings file has a simple custom format that specifies options in a nested form like so:
```
robot{
	laser_noise = 0.0
	laser_num_samples = 360
	forward_speed{
		linear = 0.2
		angular = 0.0
	}
}
```
To access a given option you need to specify the categories separated by `.`, for example `robot.forward_speed.linear`.

## Built-in Levels
The simulator comes with a few built-in levels which are listed below:
* ```empty```: An empty level with just a border surrounding the level area.
* ```random```: Randomly generated obstacles. Add flag ```--dynamic``` to add moving obstacles.
* ```svg```: Custom SVG-levels that are loaded from the folder ```svg_levels/```. The path to the folder can be specified in the settings file with the option ```stage.svg_path```.

A level can be loaded with the command ```level <level_name>``` or by setting the option ```stage.initial_level``` in the settings file. Further parameters for built-in levels can be configured in the ```stage``` section of the settings file. For example, the parameter ```stage.level_size``` specifies the size of the square-shaped *empty* and *random* levels.

### SVG levels
There is the possiblity to include SVG-based maps that are created with the vector graphic program [INKSCAPE](https://inkscape.org/). When designing a level it is important to export the SVG-files as **Inkscape SVG** (not **Plain SVG**). Please have a look at [svg_levels/example_level.svg](./svg_levels/example_level.svg) for an SVG-level example.

## Custom Levels
The workflow for creating custom levels is as follows:
1. Generate new header from template `cd level/ && ./generate_level.sh MyLevel` ->  a new file `level/MyLevel.hpp` will be created
2. Open file `level/LevelFactory.cpp`, add include directive `#include "MyLevel.hpp"` and register the level by adding the following to the constructor `LevelFactory()`:
```
LevelFactory::LevelFactory()
{
	...
	REGISTER_LEVEL(MyLevel, "my_level", "", "My new custom level.");
}
```
3. Implement level functionalities in `level/MyLevel.hpp`
4. Your level can be loaded in the simulator with the command `level my_level` (the level id `my_level` is the second parameter in `REGISTER_LEVEL()`)

Please refer to our tutorial [How to create custom levels in Arena2D?](tutorials/custom_levels.md) or the built-in levels for implementation details.

## Extending Global Settings
Adding new options that can be accessed from C++ code requires three simple steps:
1. Adding new options to the settings struct in file [settings/SettingsStructs.h](./settings/SettingsStructs.h)
1. Set a default value for the new options in file [settings/SettingsDefault.cpp](./settings/SettingsDefault.cpp)
1. Recompile the application

Now you can access the settings from anywhere in the code.
Once you remove the old settings file `settings.st` and run the application, the new settings file should containing your options set to their default values.
Please refer to our tutorial [How to create custom levels in Arena2D?](tutorials/custom_levels.md) for implementation details.

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

## Semantics
You can incorporate classes into the arena for semantic based navigation e.g. human, corridor, door, hallway, etc. 
- explain how one could include different classes (human wanderer, robot wanderer, dynamic door, etc..) and example of how to utilize these semantic information to be considered for training e.g. get closest distance to humna ,etc. .. and where to change it in the reward and penalties.



[lapan]: https://books.google.de/books?hl=en&lr=&id=xKdhDwAAQBAJ&oi=fnd&pg=PP1&dq=lapan+reinforcement+learning&ots=wTgggiYhaD&sig=VjRRQF20if5gCTVjFiuLkw_5mbk#v=onepage&q=lapan%20reinforcement%20learning&f=false
