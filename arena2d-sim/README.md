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

## Controls
Hold down the right mouse button while moving the mouse to pan the view. Scroll to zoom in and out.
For testing purposes, the robot can be controlled with the arrow keys. The keys are mapped to the discrete actions as follows:
* ```UP``` - forward
* ```LEFT``` - strong left
* ```RIGHT``` - strong right
* ```LEFT + UP``` - left
* ```RIGHT + UP``` - right
* ```DOWN``` - backwards

The velocities of the discrete actions can be configured in the settings file.

## Hot Keys
* ```F1```: open/close in-app console
* ```F2```: show/hide stats display showing several metrics such as the simulation time
* ```F3```: activate/deactivate video mode (deactivating video mode ensures maximum performance during training)
* ```F5```: take screenshot
* ```Ctrl+S```: save current settings to default settings file (```settings.st```)
* ```Ctrl+L```: load settings from default settings file 
* ```R```: reset stage (rebindable in settings file)
* ```SPACE```: run/pause simulation when training is currently inactive (this might be useful to test the behaviour of moving obstacles)

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
To access a given option you need to specify the categories separated by `.`, for the example `robot.forward_speed.linear`.

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
1. Open file `level/LevelFactory.cpp`, add include directive `#include "MyLevel.hpp"` and register the level by adding the following to the constructor `LevelFactory()`:
```
LevelFactory::LevelFactory()
{
	...
	REGISTER_LEVEL(MyLevel, "my_level", "", "My new custom level.");
}
```
1. Implement level functionalities in `level/MyLevel.hpp`
1. Your level can be loaded in the simulator with the command `level my_level` (the level id `my_level` is the second parameter in `REGISTER_LEVEL()`)

Please refer to our tutorial [How to create custom levels in Arena2D?](tutorials/custom_levels.md) or the built-in levels for implementation details.

## Extending Global Settings
Adding new options that can be accessed from C++ code requires three simple steps:
1. Adding new options to the settings struct in file [settings/SettingsStructs.h](./settings/SettingsStructs.h)
1. Set a default value for the new options in file [settings/SettingsDefault.cpp](./settings/SettingsDefault.cpp)
1. Recompile the application

Now you can access the settings from anywhere in the code.
Once you remove the old settings file `settings.st` and run the application, the new settings file should containing your options set to their default values.
Please refer to our tutorial [How to create custom levels in Arena2D?](tutorials/custom_levels.md) for implementation details.

## Sensor Data
- explain how lidar scan data is constructed, where it can be retrieved by agent. API call or similar, --evtl abstraktion vornehmen
- explanation about how to set noise

## Semantics
You can incorporate classes into the arena for semantic based navigation e.g. human, corridor, door, hallway, etc. 
- explain how one could include different classes (human wanderer, robot wanderer, dynamic door, etc..) and example of how to utilize these semantic information to be considered for training e.g. get closest distance to humna ,etc. .. and where to change it in the reward and penalties.



[lapan]: https://books.google.de/books?hl=en&lr=&id=xKdhDwAAQBAJ&oi=fnd&pg=PP1&dq=lapan+reinforcement+learning&ots=wTgggiYhaD&sig=VjRRQF20if5gCTVjFiuLkw_5mbk#v=onepage&q=lapan%20reinforcement%20learning&f=false
