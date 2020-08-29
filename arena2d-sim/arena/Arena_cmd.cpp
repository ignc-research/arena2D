#include "Arena.hpp"

#define NUM_SHOW_OBJECTS 7
static const char* SHOW_OBJECT_STRINGS[NUM_SHOW_OBJECTS*2] = {
	"stage", "gui.show_stage",
	"robot", "gui.show_robot",
	"laser", "gui.show_laser",
	"stats", "gui.show_stats",
	"goal", "gui.show_goal",
	"goal_spawn", "gui.show_goal_spawn",
	"trail", "gui.show_trail"
};

void Arena::initCommands()
{
	// get all objects that can be hidden/shown
	std::string show_hide_objects;
	for(int i = 0; i < NUM_SHOW_OBJECTS; i++){
		show_hide_objects += SHOW_OBJECT_STRINGS[i*2];
		if(i < NUM_SHOW_OBJECTS-1)
			show_hide_objects += ", ";
	}

	// get all level names
	std::string levels;
	std::list<const CommandDescription*> level_descriptions;
	LEVEL_FACTORY->getLevelDescriptions(level_descriptions);
	for(auto it = level_descriptions.begin(); it != level_descriptions.end(); it++){
		levels += "\t" + (*it)->name;
		if(!(*it)->hint.empty())
			levels += " " + (*it)->hint;
		levels += " - " + (*it)->descr + "\n"; 
	}

	// register commands
	_commands.registerCommand(&Arena::cmdExit, "exit", "",
								"exit the application");

	_commands.registerCommand(&Arena::cmdShow, "show", "<o1> <o2> ...",
								std::string("show specified gui objects (") + show_hide_objects + std::string(")"));

	_commands.registerCommand(&Arena::cmdHide, "hide", "<o1> <o2> ...",
								std::string("hide specified gui objects (") + show_hide_objects + std::string(")"));

	_commands.registerCommand(&Arena::cmdHelp, "help", "[command]",
								"display information about available commands");

	_commands.registerCommand(&Arena::cmdLoadLevel, "level", "<level> [options]",
								"load level with specified name and level specific options. Available levels:\n"+levels);

	_commands.registerCommand(&Arena::cmdReset, "reset", "",
								"reset current level to initial state");

	_commands.registerCommand(&Arena::cmdSaveSettings, "save_settings", "[filename]",
								"save current settings to given file, if no filename is specified, the settings are stored to last location");

	_commands.registerCommand(&Arena::cmdLoadSettings, "load_settings", "[filename]",
								"load current settings from given file, if no filename is specified, the settings are loaded from last location");

	_commands.registerCommand(&Arena::cmdSet, "set", "<settings_expression>",
								"set an option in the settings, e.g. set \"robot.laser_noise = 0.01\"; note that some changes (e.g. video) only take effect after restarting");

	_commands.registerCommand(&Arena::cmdStartTraining, "run_agent", "<agent.py> [--model <path>] [--device <device>] [--no_record]",
								"start training with specified python script; if flag --no_record is set then no training directory will be created and no training data is recorded");

	_commands.registerCommand(&Arena::cmdStopTraining, "stop_agent", "",
								"stop agent script");

	_commands.registerCommand(&Arena::cmdSetCameraFollow, "camera_follow", "<0:disabled/1:enabled/2:rotation>",
								"Enable/Disable the camera mode to follow robot position.");

	_commands.registerCommand(&Arena::cmdSetFPS, "fps", "<video/physics> <fps>",
								"set frames per second of video or physics thread");

	_commands.registerCommand(&Arena::cmdRealtime, "realtime", "<factor>",
								"set realtime factor of simulation, e.g. 2 for double speed, 0.5 for half speed");
}


CommandStatus Arena::command(const char * c)
{
	CommandStatus s = CommandStatus::SUCCESS;
	while(c != NULL && *c != '\0'){
		int n = 0;
		/* split command */
		char ** tokens;
		c = CommandTools::splitCommand(c, &n, &tokens);

		s = command(tokens[0], ConsoleParameters(n-1, (const char **)(tokens + 1)));
		
		/* free tokens */
		CommandTools::splitCommand_free(tokens, n);

		if(s != CommandStatus::SUCCESS)
			break;

	}
	return s;
}

CommandStatus Arena::command(const char * c, const ConsoleParameters & params){
	CommandStatus s = _commands.execCommand(this, c, params);
	switch(s)
	{
	case CommandStatus::UNKNOWN_COMMAND:{
		ERROR_F("Unknown command '%s'!", c);
	}break;
	case CommandStatus::EXEC_FAIL:{
		ERROR_F("Failed to Execute command '%s'!", c);
	}break;
	case CommandStatus::INVALID_ARG:{
		ERROR_F("Invalid arguments for command '%s'!", c);
	}break;
	case CommandStatus::SUCCESS:{
		INFO(" ");
	}break;
	}
	return s;
}


/* get description of a command given by name */
const CommandDescription* Arena::getCommand(const char * cmd){
	auto c = _commands.getCommand(cmd);
	if(c == NULL)
		return NULL;
	
	return &c->getDescription();
}

 /* get all command descriptions */
void Arena::getAllCommands(list<const CommandDescription*> & cmds){
	_commands.getAllCommandDescriptions(cmds);
}


CommandStatus Arena::cmdExit(const ConsoleParameters & params)
{
	exitApplication();
	return CommandStatus::SUCCESS;
}

CommandStatus Arena::cmdShow(const ConsoleParameters & params){
	return cmdHideShow(params, true);
}

CommandStatus Arena::cmdHide(const ConsoleParameters & params){
	return cmdHideShow(params, false);
}

CommandStatus Arena::cmdHideShow(const ConsoleParameters & params, bool show)
{
	if(params.argc == 0) return CommandStatus::INVALID_ARG;
	for(int i = 0; i < params.argc; i++){
		std::string expr = "";
		for(int o = 0; o < NUM_SHOW_OBJECTS; o++){
			if(!strcmp(params.argv[i], SHOW_OBJECT_STRINGS[o*2])){
				expr = SHOW_OBJECT_STRINGS[o*2 +1];
				break;
			}
		}
		if(expr.empty())
		{
			ERROR_F("Unknown object name '%s'", params.argv[i]);
			return CommandStatus::INVALID_ARG;
		}
		if(show)
			expr +="=1";
		else
			expr +="=0";
		_SETTINGS_OBJ->loadFromString(expr.c_str());
	}
	return CommandStatus::SUCCESS;
}

CommandStatus Arena::cmdHelp(const ConsoleParameters & params)
{
	if(params.argc == 0){
		list<const CommandDescription*> cmds;
		getAllCommands(cmds);
		INFO("Commands:");
		for(list<const CommandDescription*>::iterator it = cmds.begin(); it != cmds.end(); it++) {
			INFO_F("\t%s", (*it)->name.c_str());
		}
		INFO("Type 'help <command>' to get information about a specific command.");
	}
	else if(params.argc > 0){
		const CommandDescription * cmd_desc = getCommand(params.argv[0]);
		if(cmd_desc == NULL){
			INFO_F("No such command '%s'", params.argv[0]);
			cmdHelp(ConsoleParameters(0, NULL));
		}
		else{
			INFO_F("%s %s", cmd_desc->name.c_str(), cmd_desc->hint.c_str());
			INFO_F("-> %s", cmd_desc->descr.c_str());
		}
	}
	return CommandStatus::SUCCESS;
}


CommandStatus Arena::cmdLoadLevel(const ConsoleParameters & params)
{
	if(_trainingMode){
		showTrainingModeWarning();
		return CommandStatus::EXEC_FAIL;
	}

	// no arguments given
	if(params.argc < 1){
		return CommandStatus::INVALID_ARG;
	}

	// load level in every environment
	_levelResetMeasure.reset();
	for(int i = 0; i < _numEnvs; i++)
	{
		int res = _envs[i].loadLevel(params.argv[0], ConsoleParameters(params.argc-1, params.argv+1));
		if(res < 0)
			return CommandStatus::EXEC_FAIL;
	}
	INFO_F("Level '%s' loaded!", params.argv[0]);

	// set new initial level
	_SETTINGS->stage.initial_level = params.argv[0];
	for(int i = 1; i < params.argc; i++){
		_SETTINGS->stage.initial_level += " "+std::string(params.argv[i]);
	}
	initializeTraining();
	return CommandStatus::SUCCESS;
}

CommandStatus Arena::cmdReset(const ConsoleParameters & params)
{
	if(_trainingMode){
		showTrainingModeWarning();
		return CommandStatus::EXEC_FAIL;
	}else{
		reset(true);
		return CommandStatus::SUCCESS;	
	}
}

CommandStatus Arena::cmdSaveSettings(const ConsoleParameters & params)
{
	int ret = 0;
	std::string path;// std::string is used because _lastSettingsPath gets overwritten during saving
	if(params.argc > 0){
		path = params.argv[0];
	}
	else{
		path = _SETTINGS_OBJ->getLastSettingsPath();
	}
	ret = _SETTINGS_OBJ->saveToFile(path.c_str());

	if(ret){
		return CommandStatus::EXEC_FAIL;
	}
	else{
		INFO_F("Settings saved to '%s'", path.c_str());
		return CommandStatus::SUCCESS;
	}
}

CommandStatus Arena::cmdLoadSettings(const ConsoleParameters & params)
{
	if(_trainingMode){
		showTrainingModeWarning();
		return CommandStatus::EXEC_FAIL;
	}
	int ret = 0;
	std::string path;// std::string is used because _lastSettingsPath gets overwritten during loading
	if(params.argc > 0){
		path = params.argv[0];
	}
	else{
		path = _SETTINGS_OBJ->getLastSettingsPath();
	}
	ret = _SETTINGS_OBJ->loadFromFile(path.c_str(), 0);

	if(ret){
		return CommandStatus::EXEC_FAIL;
	}
	else{
		INFO_F("Settings loaded from '%s'", path.c_str());
		return CommandStatus::SUCCESS;
	}
}

CommandStatus Arena::cmdSet(const ConsoleParameters & params){

	if(params.argc != 1){
		return CommandStatus::INVALID_ARG;
	}

	if(_SETTINGS_OBJ->loadFromString(params.argv[0])){
		return CommandStatus::EXEC_FAIL;
	}

	// update environment params
	for(int i = 0; i < _numEnvs; i++)
		_envs[i].refreshSettings();
	
	return CommandStatus::SUCCESS;
}

CommandStatus Arena::cmdStartTraining(const ConsoleParameters & params)
{
	if(params.argc < 1){return CommandStatus::INVALID_ARG;}
	CommandStatus status = CommandStatus::SUCCESS;
	if(!_trainingMode){
		_pyAgentUsed = true;
		// running python script
		FILE * file = fopen(params.argv[0], "rb");
		if(file == NULL){
			status = CommandStatus::EXEC_FAIL;
			ERROR_F("Failed to open file '%s': %s", params.argv[0], strerror(errno));
		}
		else{
			// check for additional parameter
			const char * model = NULL;
			const char * device = "cpu";
			params.getString("--model", model);
			params.getString("--device", device);
			_noTrainingRecord = false;
			if(params.getFlag("--no_record")){
				_noTrainingRecord = true;
			}
			// set argv (absolute path of script that is beeing executed)
			char buffer[256];	
			buffer[0] = '\0';
			getcwd(buffer, 256);
			int cwd_len = strlen(buffer);
			buffer[cwd_len] = '/';
			strcpy(&buffer[cwd_len+1], params.argv[0]);
			cwd_len = strlen(buffer);
			#ifdef ARENA_PYTHON_VERSION_3
				PyMem_RawFree(_pyArg0);
				_pyArg0 = Py_DecodeLocale(buffer, NULL);
			#else
				delete[](_pyArg0);
				_pyArg0 = new char[cwd_len+1];
				strcpy(_pyArg0, buffer);
			#endif
			PySys_SetArgv(1, &_pyArg0);
			if(PyRun_SimpleFileEx(file, params.argv[0], 1)){
				status = CommandStatus::EXEC_FAIL;
				ERROR_F("Failed to execute script '%s'!", params.argv[0]);
			}
			else
			{
				const char * module_name = "__main__";
				_agentModule = PyImport_ImportModule(module_name);
				if(_agentModule == NULL)
				{
					status = CommandStatus::EXEC_FAIL;
					ERROR_F("Could not get module '%s'!", module_name);
				}
				else
				{
					// get Agent class
					const char * agent_class_name = _SETTINGS->training.agent_class.c_str();
					PyObject* agent_class = PyObject_GetAttrString(_agentModule, agent_class_name);
					PyObject* agent_instance = NULL;
					if(agent_class == NULL)
					{
						Py_DECREF(_agentModule);
						status = CommandStatus::EXEC_FAIL;
						ERROR_F("Failed to get agent class '%s'", agent_class_name);
					}
					else
					{
						// instanciate agent
						INFO_F("Initializing '%s' object...", agent_class_name);
						const char * training_path = "./";
						if(!_noTrainingRecord){
							if(createTrainingDir(params.argv[0]) == 0){// directory creation success
								training_path = _trainingDir;
							}
						}
						// setting arguments for constructor
						PyObject * init_args = PyTuple_New(6);
						PyTuple_SetItem(init_args, 0, PyUnicode_FromString(device));
						if(model)
							PyTuple_SetItem(init_args, 1, PyUnicode_FromString(model));
						else
							PyTuple_SetItem(init_args, 1, Py_None);
						PyTuple_SetItem(init_args, 2, PyLong_FromLong(getPyObservationSize()));
						PyTuple_SetItem(init_args, 3, PyLong_FromLong(_numEnvs));
						PyTuple_SetItem(init_args, 4, PyLong_FromLong(_numThreads));
						PyTuple_SetItem(init_args, 5, PyUnicode_FromString(training_path));

						agent_instance = PyObject_CallObject(agent_class, init_args);
						Py_DECREF(init_args);
						if(agent_instance == NULL){
							PyErr_Print();
							ERROR_F("Instancing of '%s' object failed!", agent_class_name);
							status = CommandStatus::EXEC_FAIL;
						}
						else
						{
							// getting functions
							for(int i = 0; i < PYAGENT_FUNC_NUM; i++){
								_agentFuncs[i] = PyObject_GetAttrString(agent_instance, PYAGENT_FUNC_NAMES[i]);
								if(!(_agentFuncs[i] && PyMethod_Check(_agentFuncs[i]))){
									if(_agentFuncs[i] == NULL)
										PyErr_Print();
									WARNING_F("Function '%s' was not found in class '%s'!", PYAGENT_FUNC_NAMES[i], agent_class_name);
								}
							}
						}

						// free references
						Py_XDECREF(agent_instance);
						Py_XDECREF(agent_class);
					}
				}
			}
		}
		if(status == CommandStatus::SUCCESS){
			_trainingMode = true;
			initializeTraining();
			INFO("TRAINING MODE. Critical actions performed by user are blocked!");
		}
	}
	else{
		INFO("Already in TRAINING MODE. Run command 'stop_agent' to stop the training.");
	}
	return status;
}

CommandStatus Arena::cmdStopTraining(const ConsoleParameters & params)
{
	if(_trainingMode){
		_trainingMode = false;
		INFO("TRAINING MODE deactivated.");
		const char * agent_results = "-";
		PyObject * ret = NULL;
		if(_pyAgentUsed){
			// training stop callback
			PyObject * f = _agentFuncs[PYAGENT_FUNC_STOP];
			if(f){
				ret = PyObject_CallObject(f, NULL);
				if(ret == NULL){
					PyErr_Print();
					ERROR_F("Call to Function '%s' failed!", PYAGENT_FUNC_NAMES[PYAGENT_FUNC_STOP]);
				}
				else if(PyUnicode_Check(ret)){
					agent_results = PyUnicode_AsUTF8(ret);
				}
				else if(ret != Py_None)
				{
					ERROR_F("Expected string or None in return value of function '%s'!", PYAGENT_FUNC_NAMES[PYAGENT_FUNC_STOP]);
				}
			}
		}
		// writing results file
		if(!_noTrainingRecord){
			std::string results_path = std::string(_trainingDir) + "summary.txt";
			FILE * f = fopen(results_path.c_str(), "w");
			if(f){
				char time_buffer[32];
				Timer::getTimeString(SDL_GetTicks()-_trainingStartTime, time_buffer, 32);
				fprintf(f,
					"### Training Summary ###\n"
					"Script:   %s\n"
					"Started:  %s\n"
					"Episodes: %d\n"
					"Time:     %s\n"
					"\n"
					"# Agent Info #\n%s",
					_agentPath.c_str(),
					_trainingStartString,
					_episodeCount,
					time_buffer,
					agent_results);
				fclose(f);
			}
			else{
				ERROR_F("Failed write '%s': %s", results_path.c_str(), strerror(errno));
			}
		}
		if(_pyAgentUsed){
			// throw away last return value
			Py_XDECREF(ret);
			for(int i = 0; i < PYAGENT_FUNC_NUM; i++){
				Py_XDECREF(_agentFuncs[i]);
				_agentFuncs[i] = NULL;
			}
			Py_XDECREF(_agentModule);
			_agentModule = NULL;
			_pyAgentUsed = false;
			INFO("Agent stopped!");
		}

		// close csv file
		_csvWriter.close();
	}
	else{
		INFO("Currently not in TRAINING MODE.");
	}
	return CommandStatus::SUCCESS;
}

CommandStatus Arena::cmdSetCameraFollow(const ConsoleParameters & params)
{
	if(params.argc != 1)
	{
		return CommandStatus::INVALID_ARG;
	}
	int mode;
	params.getIntAt(0, mode);
	if(mode < 0 || mode > 2)
	{
		ERROR_F("Invalid mode for '%d' for camera follow!", mode);
		return CommandStatus::INVALID_ARG;
	}

	_SETTINGS->gui.camera_follow = mode;
	return CommandStatus::SUCCESS;
}

CommandStatus Arena::cmdSetFPS(const ConsoleParameters & params)
{
	if(params.argc != 2){
		return CommandStatus::INVALID_ARG;
	}

	int fps;
	params.getIntAt(1, fps);
	if(fps < 1)
		fps = 1;
	else if(fps > 1000)
		fps = 1000;
	if(!strcmp(params.argv[0], "physics")){
		_SETTINGS->physics.fps = fps;
	}
	else if(!strcmp(params.argv[0], "video")){
		_SETTINGS->video.fps = fps;
	}
	else{
		ERROR("First Parameter must be 'video' or 'physics'");
		return CommandStatus::INVALID_ARG;
	}

	return CommandStatus::SUCCESS;
}

CommandStatus Arena::cmdRealtime(const ConsoleParameters & params)
{
	float factor = 1.0f;
	params.getFloatAt(0, factor);
	_SETTINGS->physics.fps = factor/(_SETTINGS->physics.time_step*_SETTINGS->physics.step_iterations);
	return CommandStatus::SUCCESS;
}
