/* Author: Cornelius Marx */

#ifndef ARENA_H
#define ARENA_H

#include <iostream>
#include <engine/zFramework.hpp>
#include <engine/GlobalSettings.hpp>
#include <engine/Camera.hpp>
#include <engine/Timer.hpp>
#include <signal.h>
#include <time.h>
#include <sys/stat.h>
#include "Command.hpp"
#include "CommandRegister.hpp"
#include "Console.hpp"
#include "PhysicsWorld.hpp"
#include "ConsoleParameters.hpp"
#include "MeanBuffer.hpp"
#include "Environment.hpp"
#include "StatsDisplay.hpp"
#include "CSVWriter.hpp"
#include <Python.h>
#include <script_sources.generated.h>

// python callback functions
enum PyAgentFunction{PYAGENT_FUNC_PRE_STEP, PYAGENT_FUNC_POST_STEP, PYAGENT_FUNC_GET_STATS, PYAGENT_FUNC_STOP, PYAGENT_FUNC_NUM};
extern const char * PYAGENT_FUNC_NAMES[PYAGENT_FUNC_NUM];

/* hotkeys */
#define ARENA_HOTKEY_CONSOLE 		SDLK_F1 // open console
#define ARENA_HOTKEY_SHOW_STATS		SDLK_F2 // show/hide stats display
#define ARENA_HOTKEY_DISABLE_VIDEO 	SDLK_F3 // temporarily disable video
#define ARENA_HOTKEY_SCREENSHOT 	SDLK_F5 // take screenshot

/* application main class */
class Arena
{
public:
	/* constructor
	 * NOTE: initialization is done in function init()
	 */
	Arena();

	/* destructor
	 * NOTE: memory is freed in function quit()
	 */
	~Arena(){}

	/* initialize Arena2D with parameters from commandline
	 * called by main() function
	 * @return -1 on error, 0 on success
	 */
	int init(int argc, char ** argv);

	/* start main loop of application, blocking
	 * called by main() function
	 */
	void run();

	/* shut down application, free all memory
	 * called by main() function
	 */
	void quit();

	/* request application shutdown
	 * prompt user with a dialog if training is currently running
	 */
	void exitApplication();

private:
	/* execute full command
	 * @param c command to be executed
	 * @return command status
	 */
	CommandStatus command(const char * c);

	/* execute command separated into command name and parameters
	 * called by command(const char*)
	 * @param name command name
	 * @param params parameters to pass to command function
	 */
	CommandStatus command(const char * name, const ConsoleParameters & params);

	/* get command description of a given command
	 * @param name the name of the command
	 * @return command description
	 */
	const CommandDescription* getCommand(const char * name); /* get command description */

	/* get command description of all command
	 * @param cmd command descriptions are pushed into this list
	 */
	void getAllCommands(list<const CommandDescription*> & cmds); 

	/* command functions
	 * the following functions are called once a specific command is executed
	 * more information on each command can be found in Arena_cmd.cpp
	 * @param params command parameters 
	 * @return command status
	 */
	CommandStatus cmdHelp(const ConsoleParameters & params);
	CommandStatus cmdExit(const ConsoleParameters & params);
	CommandStatus cmdShow(const ConsoleParameters & params);
	CommandStatus cmdHide(const ConsoleParameters & params);
	CommandStatus cmdHideShow(const ConsoleParameters & params, bool show);
	CommandStatus cmdLoadLevel(const ConsoleParameters & params);
	CommandStatus cmdReset(const ConsoleParameters & params);
	CommandStatus cmdSaveSettings(const ConsoleParameters & params);
	CommandStatus cmdLoadSettings(const ConsoleParameters & params);
	CommandStatus cmdSet(const ConsoleParameters & params);
	CommandStatus cmdStartTraining(const ConsoleParameters & params);
	CommandStatus cmdStopTraining(const ConsoleParameters & params);
	CommandStatus cmdSetFPS(const ConsoleParameters & params);
	CommandStatus cmdSetCameraFollow(const ConsoleParameters & params);
	CommandStatus cmdRealtime(const ConsoleParameters & params);

	/* reset current level in all environments
	 */
	void reset(bool robot_position_reset);

	/* pack observation of given environment into new PyObject
	 * @param env_index index of environment to get observations from
	 * @return new list containing all observations from the environment
	 */
	PyObject* packPyObservation(int env_index);

	/* get number of scalar values in one observation
	 * @return observation size
	 */
	int getPyObservationSize();

	/* pack observation of all environments into new PyObject
	 * @return 	new PyList containing an observation (list of float) for each environment
	 * 			or a single observation if #envs==1 (similiar to packPyObservation(int))
	 */
	PyObject* packAllPyObservation();

	/* pack rewards from all environments into new PyObject
	 * @return 	new PyList containing rewards (float) from all environments or
	 			a single PyFloat if #envs==1
	 */
	PyObject* packAllPyRewards();

	/* pack rewards from all environments into new PyObject
	 * @return 	new PyList containing done flags (long: 0 or 1) from all environments or
	 			a single done flag if #envs==1
	 */
	PyObject* packAllPyDones();

	/* create directory to put training data (e.g. plots in)
	 * @param agent_path path to agent script
	 * @return -1 on error, 0 on success
	 */
	int createTrainingDir(const char * agent_path);

	/* make screenshot and save it to given path (BMP)
	 * @param path path to save screenshot to
	 */
	void screenshot(const char * path);

	/* initialize commands
	 */
	void initCommands();

	/* initialize stats display
	 */
	void initStats();

	/* render physics world (e.g. robot, obstacles, laser)
	 * called in main loop
	 */
	void render();

	/* render GUI (e.g. in-app console, stats)
	 * called in main loop
	 */
	void renderGUI();

	/* render black screen showing 'video disabled'- message
	 * this screen is rendered if the user disables rendering for faster training
	 * NOTE: standalone render function, calls render() and renderGUI()
	 */
	void renderVideoDisabledScreen();

	/* pass events to console to update typing
	 * issues command when entered by user
	 */
	void updateConsole(zEventList & evtList);

	/* update user/agent robot control
	 * call agent callbacks in python script
	 * update physics
	 * called in main loop
	 */
	void update();

	/* handle events (e.g. window, keyboard, mouse)
	 * called in main loop 
	 */
	void processEvents(zEventList & evtList);

	/* called by processEvents() if window resize event occurs
	 */
	void resize();

	/* update visual fps counter metrics (e.g. video, physics)
	 * called by processEvents() on every TICK-event
	 */
	void refreshFPSCounter();

	/* update visual episode counter metric
	 * called whenever number of episodes changes
	 */
	void refreshEpisodeCounter();

	/* update visual reward counter metric
	 * called when new episode starts
	 */
	void refreshRewardCounter();

	/* update visual level reset time metric
	 * called when level is reset or a new level is loaded
	 */
	void refreshLevelResetTime();

	/* initialize training, reset environment, episode counter
	 * called when training starts
	 */
	void initializeTraining();

	/* warn user if action is not be performed due to active training mode
	 */
	static void showTrainingModeWarning()
	{
		WARNING("TRAINING MODE active. Action might interfere with training process and thus will not be performed!");
	}

	/* print several metrics to console
	 * called if an episode ends
	 */
	void printEpisodeResults(float total_reward);

	/* regular font */
	zFont * _monospaceRegular;

	/* bold font */
	zFont * _monospaceBold;

	/* in-app console */
	Console *_console;

	/* command register mapping command-names to member functions */
	CommandRegister<Arena, CommandStatus, const ConsoleParameters&> _commands;

	/* stats display showing several metrics */
	StatsDisplay * _statsDisplay;

	/* metrics index in array */
	enum Metrics{	VIDEO_FPS,
					PHYSICS_FPS,
					STEPS_PER_SECOND,
					SIMULATION_TIME,
					REALTIME,
					MEAN_REWARD,
					MEAN_SUCCESS,
					AGENT_TIME,
					LEVEL_RESET_TIME,
					NUM_EPISODES,
					TIME_ELAPSED,
					NUM_METRICS
	};

	/* array holding metric handles to update values, enum Metrics used as indicies */
	MetricHandle * _metricHandles[NUM_METRICS];

	/* application still running? 'volatile' because this value is written by sigint handler */
	volatile bool _run;

	/* array of environments (_numEnvs), dynamically allocated in init() */
	Environment * _envs;

	/* total number of environments (in array _envs) */
	int _numEnvs;

	/* number of environments displayed accross window width */
	int _envsX; 

	/* number of environments displayed accross window height */
	int _envsY; 

	/* array of abstract threads (_numThreads) responsible for performing simulation steps in specific environments */
	EnvironmentThread * _threads;

	/* number of threads accross which to calculate simulation steps for all environments */
	int _numThreads;

	/* buffer keeping track of total rewards from last 100 episodes */
	MeanBuffer _meanReward;

	/* buffer keeping track success rate */
	MeanBuffer _meanSuccess;

	/* total number of episodes since training start */
	int _episodeCount;

	/* keeping track of update time (FPS) */
	Timer _updateTimer;

	/* keeping track of render time (FPS) */
	Timer _videoTimer;

	/* keeping track of time for physics step (FPS) */
	Timer _physicsTimer;
	
	/* measuring time, the whole simulation step accross all environments takes */
	MeanTimeBuffer _simulationMeasure; 

	/* measuring time for post_step callback function */
	MeanTimeBuffer _agentPostMeasure;

	/* measuring time for pre_step callback function */
	MeanTimeBuffer _agentMeasure; 

	/* time it takes to reset the level */
	MeanTimeBuffer _levelResetMeasure;

	/* array of continuous actions to perform in all environments */
	Twist* _actions;
	
	/* array for keeping track of episode ends */
	bool * _dones;

	/* console currently enabled (user can type in commands)? */
	bool _consoleEnabled;

	/* should camera be translated according to mouse movement? */
	bool _translateCamera;

	/* should camera be rotated according to mouse movement? */
	bool _rotateCamera;

	/* global camera for all environments */
	Camera _camera;

	/* keeping track of the keys pressed for user robot control */
	bool _keysPressed[4];

	/* are simulation steps performed without the user pressing any robot control keys? */
	bool _playSimulation;

	/* training mode active? (restricted user input) */
	bool _trainingMode;

	/* temporarily disable rendering to increase performance of training */
	bool _videoDisabled;

	/* message shown when video is temporarily disabled during runtime */
	zTextView * _videoDisabledText;

	/* number of ticks (SDL_GetTicks()) when training has started */
	Uint32 _trainingStartTime;

	/* gl buffer for rendering grid to visually separate environments */
	GLuint _envGridBuffer;

	/* number of verticies in grid buffer */
	GLuint _envGridBufferCount;

	/* set to true if python callback functions are used for training */
	bool _pyAgentUsed;
	#ifdef ARENA_PYTHON_VERSION_3
		wchar_t
	#else// old python <= 2.7
		char
	#endif
		*_pyProgName,
		*_pyArg0;

	/* PyMethods from agent class, callbacks for training */
	PyObject * _agentFuncs[PYAGENT_FUNC_NUM];

	/* main module from python script */
	PyObject * _agentModule;

	/* name of directory to put training data in */
	char _trainingDir[128];

	/* string representing training start time */
	char _trainingStartString[128];

	/* path to agent script */
	std::string _agentPath;

	/* csv writer for writing out metrics */
	CSVWriter _csvWriter;

	/* if set to true no training data is recorded*/
	bool _noTrainingRecord;
};


#endif
