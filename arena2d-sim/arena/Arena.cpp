/* Author: Cornelius Marx */
#include "Arena.hpp"

// sigint handler
static Arena* ARENA =  NULL;// arena instance
void sigintHandler(int state){
	ARENA->exitApplication();
}

static const char* ARENA_HELP_STRING =
	"options:\n"
	"--help -h                      display this info\n"
	"--disable-video                commandline only\n"
	"--run <commands>               run simulator commands on startup (separated by ;)\n"
	"--logfile <filename>           log output to file\n";

Arena::Arena(): _levelResetMeasure(10)
{
	_console = NULL;
	_monospaceRegular = NULL;
	_monospaceBold = NULL;
	_consoleEnabled = false;
	_episodeCount = 0;
	for(int i = 0; i < PYAGENT_FUNC_NUM; i++){
		_agentFuncs[i] = NULL;
	}
	_agentModule = NULL;
	_pyAgentUsed = false;
}

int Arena::init(int argc, char ** argv)
{
	_run = false;
	/* parse cmd options */
	bool disable_video = false;
	const char* log_file = NULL;
	_trainingMode = false;
	int command_index = -1;
	int arg_i = 1;
	while(arg_i < argc){
		if(!strcmp(argv[arg_i], "--disable-video")){// no video 
			disable_video = true;
		}
		else if(!strcmp(argv[arg_i], "--logfile")){// logfile
			if(arg_i < argc-1){
				log_file = argv[arg_i+1];
				arg_i++;// skip next argument (logfile path)
			}
			else{
				printf("No path given for logfile!\n");
				exit(1);
			}
		}
		else if((!strcmp(argv[arg_i], "--help") ||
						!strcmp(argv[arg_i], "-h"))){ // display help
			puts(ARENA_HELP_STRING);
			exit(0);
		}
		else if(!strcmp(argv[arg_i], "--run")){
			if(arg_i+1 < argc){
				command_index = arg_i+1;
				arg_i++;// skip next argument run command
			}
			else{
				printf("No commands given for option --run!\n");
				exit(0);
			}
		}
		else{
			printf("Unknown commandline option '%s'\n", argv[arg_i]);
			exit(1);
		}
		arg_i++;
	}
	if(log_file)
		printf("Logging to '%s'.\n", log_file);

	/* create logger */
	Z_LOG->createLog(true, log_file);
	initCommands();

	/* init global settings*/
	if(_SETTINGS_OBJ->init("./settings.st")){
		return -1;
	}

	/* random seed */
	srand((unsigned int)_SETTINGS->stage.random_seed);

	if(disable_video){
		_SETTINGS->video.enabled = false;
	}

	if(_SETTINGS->video.enabled)
		INFO("Starting Application in VIDEO MODE.");
	else
		INFO("Starting Application in COMMANDLINE MODE.");
	/* video setup */
	if(_SETTINGS->video.enabled) {

		/* initialize framework */
		SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");// we have our own sigint handler
		unsigned int flags = Z_RESIZABLE;
		if(_SETTINGS->video.vsync)
			flags |= Z_V_SYNC;
		if(_SETTINGS->video.fullscreen)
			flags |= Z_FULLSCREEN;
		if(_SETTINGS->video.maximized)
			flags |= Z_MAXIMIZED;
		if(Z_FW->init(	_SETTINGS->video.resolution_w,
						_SETTINGS->video.resolution_h,
						_SETTINGS->video.window_x,
						_SETTINGS->video.window_y,
						_SETTINGS->video.msaa,
						flags, false)){
			ERROR("Failed to initialize Framework!");
			return -1;
		}
		
		/* initialize opengl renderer */
		if(_RENDERER->init(false)){
			ERROR("Could not initialize Renderer!");
			return -1;
		}

		/* setting background color */
		glClearColor(1,1,1,1);

	
		INFO("Loading Fonts...");
		
		std::string dir = "./data/fonts/";
		std::string sys_dir = "/usr/share/fonts/TTF/";
		std::string font_regular_name = "Bitstream_Regular.ttf";
		std::string font_bold_name = "Bitstream_Bold.ttf";
		std::string font_regular_path = dir + font_regular_name;
		std::string font_bold_path = dir + font_bold_name;
		// check if file exists in default path
		FILE *f = fopen((font_regular_path).c_str(), "rb");
		if(f){
			fclose(f);
		}
		else{// does not exist locally, try system path
			font_regular_path = sys_dir + font_regular_name;
		}
		f = fopen((font_bold_path).c_str(), "rb");
		if(f){
			fclose(f);
		}
		else{// does not exist locally, try system path
			font_bold_path = sys_dir + font_bold_name;
		}

		/* loading fonts */
		_monospaceRegular = new zFont();
		_monospaceBold = new zFont();
		std::string p = dir+"Bitstream_Regular.ttf";
		INFO_F("-> loading %s", font_regular_path.c_str());
		if(_monospaceRegular->loadFromFile(font_regular_path.c_str())){
			return -1;
		}
		_monospaceRegular->renderMap(Z_NORMAL, _SETTINGS->gui.font_size);
		_monospaceBold = new zFont();
		p = dir + "Bitstream_Bold.ttf";
		INFO_F("-> loading %s", font_bold_path.c_str());
		if(_monospaceBold->loadFromFile(font_bold_path.c_str())){
			return -1;
		}
		_monospaceBold->renderMap(Z_NORMAL, _SETTINGS->gui.font_size);
		_monospaceBold->renderMap(Z_BIG, _SETTINGS->gui.font_size*1.5f);

		/* creating console */
		std::string home_path = getenv("HOME");
		std::string cmd_history_path =  home_path + "/.arena2d_command_history.txt";
		_console = new Console(_monospaceRegular, Z_NORMAL, cmd_history_path.c_str());
		_console->enable();

		/* initialize stats */
		initStats();

		/* creating video disabled text */
		_videoDisabledText = new zTextView(_monospaceBold, Z_BIG, "Video mode disabled! Press 'F3' to enable.");
		_videoDisabledText->setAlignment(0, 0, 0, true);
		_videoDisabledText->setColor(1,1,1,1);

		/* initial resize call */
		_videoDisabled = false;

		/* initial camera settings */
		_translateCamera = false;
		_rotateCamera = false;
		_camera.set(zVector2D(_SETTINGS->gui.camera_x, _SETTINGS->gui.camera_y),
						_SETTINGS->gui.camera_zoom, f_rad(_SETTINGS->gui.camera_rotation));
		_camera.setZoomFactor(_SETTINGS->gui.camera_zoom_factor);
		_camera.refresh();

	}
	/* initialize physics */
	_PHYSICS->init();
	_PHYSICS->setIterations(_SETTINGS->physics.velocity_iterations,
							_SETTINGS->physics.position_iterations);

	/* register this object as contact listener */
	_PHYSICS_WORLD->SetContactListener((b2ContactListener*)this);

	/* creating environments */
	_numEnvs = _SETTINGS->training.num_envs;
	if(_numEnvs <= 0){
		_numEnvs = 1;
	}
	_envs = new Environment[_numEnvs];

	// find closest square
	_envsX = ceil(sqrt(_numEnvs));
	_envsY = _envsX;

	/* creating environment grid */
	_envGridBuffer = 0;
	_envGridBufferCount = 0;
	if(_SETTINGS->video.enabled && _numEnvs > 1){
		glGenBuffers(1, &_envGridBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, _envGridBuffer);
		_envGridBufferCount	= 2*((_envsX-1)+(_envsY-1));
		float * verts = new float[_envGridBufferCount*2];
		int count = 0;
		float x_per_env = 2.0f/_envsX;
		float y_per_env = 2.0f/_envsY;
		for(int x = 1; x < _envsX; x++) {
			float x_pos = -1 + x_per_env*x;
			verts[count*4 + 0] = x_pos;
			verts[count*4 + 1] = 1;
			verts[count*4 + 2] = x_pos;
			verts[count*4 + 3] = -1;
			count+=1;
		}
		for(int y = 1; y < _envsY; y++) {
			float y_pos = -1 + y_per_env*y;
			verts[count*4 + 0] = 1;
			verts[count*4 + 1] = y_pos;
			verts[count*4 + 2] = -1;
			verts[count*4 + 3] = y_pos;
			count+=1;
		}
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*_envGridBufferCount*2, verts, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		delete[] verts;
	}

	// create actions, rewards and _dones array
	_actions = new Twist[_numEnvs];
	_dones = new bool[_numEnvs];

	_numThreads = _SETTINGS->training.num_threads;
	if(_numThreads == 0){// at least 1 thread
		_numThreads = 1;
	}else if(_numThreads < 0){// automatically determine number of threads
		_numThreads = SDL_GetCPUCount();
	}
	if(_numThreads > _numEnvs){// more threads than environments does not make any sense
		_numThreads = _numEnvs;
	}
	INFO_F("Simulating %d environment(s) accross %d thread(s).", _numEnvs, _numThreads);
	_threads = new EnvironmentThread[_numThreads];
	const int envs_per_thread = _numEnvs/_numThreads;
	const int envs_remain = _numEnvs%_numThreads;
	int env_index = 0;
	for(int i = 0; i < _numThreads; i++){
		int current_num_envs = envs_per_thread;
		if(i < envs_remain){
			current_num_envs++;
		}
		_threads[i].init(_envs, current_num_envs, env_index, _actions);
		env_index += current_num_envs;
	}

	// init key status
	memset(_keysPressed, 0, sizeof(_keysPressed));
	 
	// setup timer if video is disabled
	if(_SETTINGS->video.enabled == false){
		// init timer
		SDL_Init(SDL_INIT_TIMER);
	}

	// initialize sigint-handler
	ARENA = this;
	signal(SIGINT, &sigintHandler);

	char * py_path = Py_EncodeLocale(Py_GetPath(), NULL);
	char * py_prefix = Py_EncodeLocale(Py_GetPrefix(), NULL);
	char * py_exec_prefix = Py_EncodeLocale(Py_GetExecPrefix(), NULL); 
	INFO_F("Python path: %s", py_path);
	INFO_F("Python prefix: %s", py_prefix);
	INFO_F("Python exec prefix: %s", py_exec_prefix);
	PyMem_Free(py_path);
	PyMem_Free(py_prefix);
	PyMem_Free(py_exec_prefix);
	/* initialize python interpreter */
	#ifdef ARENA_PYTHON_VERSION_3
		_pyProgName = Py_DecodeLocale(argv[0], NULL);
	#else
		_pyProgName = new char[len+1];
		strcpy(_pyProgName, argv[0]);
	#endif
	Py_SetProgramName(_pyProgName);
	_pyArg0 = NULL;
	Py_InitializeEx(0);

	INFO("Initialization done! Running Arena...\n");
	_run = true;

	/* load initial level */
	std::string level_cmd = std::string("level ") + _SETTINGS->stage.initial_level.c_str();
	if((command(level_cmd.c_str())) != CommandStatus::SUCCESS){
		/* init training if not already done through loadLevel*/
		initializeTraining();
	}

	if(command_index >= 0){
		INFO_F("Running initial command: > %s", argv[command_index]);
		if(command(argv[command_index]) != CommandStatus::SUCCESS){
			return -1;
		}
	}

	/* initial resize */
	if(_SETTINGS->video.enabled){
		resize();

		refreshEpisodeCounter();
		refreshRewardCounter();
		refreshFPSCounter();
	}

	_playSimulation = false;

	return 0;
}

void Arena::initStats()
{
	zColor text_color(0.1, 0.1, 0.1, 1.0);
	zColor bg_color(0.8, 0.8, 0.8, 0.9);
	_statsDisplay = new StatsDisplay(_monospaceRegular, Z_NORMAL, text_color, text_color, bg_color);
	_metricHandles[VIDEO_FPS] = _statsDisplay->addMetric("Video FPS");
	_metricHandles[PHYSICS_FPS] = _statsDisplay->addMetric("Physics FPS");
	_metricHandles[STEPS_PER_SECOND] = _statsDisplay->addMetric("Time Step (s)");
	_metricHandles[SIMULATION_TIME] = _statsDisplay->addMetric("Simulation Time (ms)");
	_metricHandles[REALTIME] = _statsDisplay->addMetric("Realtime");
	_metricHandles[MEAN_REWARD] = _statsDisplay->addMetric("Mean Reward");
	_metricHandles[MEAN_SUCCESS] = _statsDisplay->addMetric("Mean Success");
	_metricHandles[AGENT_TIME] = _statsDisplay->addMetric("Agent Time (ms)");
	_metricHandles[LEVEL_RESET_TIME] = _statsDisplay->addMetric("Level Reset Time (ms)");
	_metricHandles[NUM_EPISODES] = _statsDisplay->addMetric("Episodes");
	_metricHandles[TIME_ELAPSED] = _statsDisplay->addMetric("Time Elapsed");
}

void Arena::exitApplication()
{
	INFO("\nApplication shutdown requested!");
	if(_run == false)// already tried to quit -> force
		exit(1);
	static const char * prompt_text = "TRAINING MODE active. Do you really want to exit the application?";
	if(_trainingMode && _SETTINGS->video.enabled){
		// creating message box and ask user whether he really wants to exit
		static const SDL_MessageBoxButtonData buttons[2] = {
				{0, 0, "NO, keep on training."},// type, button_id, text
				{0, 1, "YES, stop training!"}
		};
		static const SDL_MessageBoxColorScheme colors = {
			{
				{210, 210, 210},// message box background color
				{20, 20, 20},	// text color
				{20, 20, 20},	// button border
				{170, 170, 170},// button background
				{255, 255, 255} // button text selected
			}
		};
		const SDL_MessageBoxData data = {
			SDL_MESSAGEBOX_WARNING,
			Z_FW->getMainWindow(),
			"Exit?",
			prompt_text,
			2,
			buttons,
			&colors
		};
		int buttonid;
		if(SDL_ShowMessageBox(&data, &buttonid)){
			ERROR("Failed to show message box!");
		}
		else if(buttonid == 1){ /// user clicked yes
			_run = false;
		}
	}
	else{
		if(_trainingMode){
			printf("%s (y/n): ", prompt_text);
			std::string str;
			getline(std::cin, str);
			zStringTools::toLower(str);
			if(str == "yes" || str == "y"){
				_run = false;
			}
		}
		else{
			_run = false;
		}
	}
	if(_run){
		INFO("Exiting cancelled!");
	}
}

void Arena::quit()
{
	INFO("\nShutting down application...");

	// shutting down environment threads
	delete[]_threads;

	delete[](_actions);
	delete[](_dones);

	/* shutting down python interpreter */

	if(_trainingMode)
		cmdStopTraining(ConsoleParameters(0, NULL));
	Py_Finalize();
	#ifdef ARENA_PYTHON_VERSION_3
		PyMem_RawFree(_pyProgName);
		PyMem_RawFree(_pyArg0);
	#else
		delete[](_pyArg0);
		delete[]_pyProgName;
	#endif

	/* free console */
	delete _console;

	/* shutting down physics */
	_PHYSICS->del();

	if(_SETTINGS->video.enabled){
		/* free display */
		delete _statsDisplay;

		/* free fonts */
		delete _monospaceRegular;
		delete _monospaceBold;

		/* shutting down renderer */
		_RENDERER->del();

		/* shutting down framework */
		Z_FW->del();
	}

	INFO("Done!");
	/* close logger */
	Z_LOG->del();
}

void Arena::reset(bool robot_position_reset){
	for(int i = 0; i < _numEnvs; i++)
	{
		_levelResetMeasure.startTime();
		_envs[i].reset(robot_position_reset);
		_levelResetMeasure.endTime();
	}

	// refresh reward counter
	if(_SETTINGS->video.enabled)
	{
		refreshRewardCounter();
		refreshLevelResetTime();
	}
}

void Arena::initializeTraining()
{
	_meanSuccess.reset();
	_meanReward.reset();
	_trainingStartTime = SDL_GetTicks();
	_episodeCount = 0;
	if(_SETTINGS->video.enabled)
	{
		refreshEpisodeCounter();
		refreshLevelResetTime();
	}
}

int Arena::createTrainingDir(const char * agent_path)
{
	// get local time as string
	time_t t;
	time(&t);
	tm * timeinfo;
	timeinfo = localtime(&t);
	strftime(_trainingDir, sizeof(_trainingDir), "training_%a%d_%H-%M-%S/", timeinfo); 
	strftime(_trainingStartString, sizeof(_trainingStartString), "%c", timeinfo); 

	INFO_F("Creating folder '%s'", _trainingDir);
	// create directory with user write, read, exec permission
	if(mkdir(_trainingDir, S_IRWXU))
	{
		ERROR_F("Could not create training folder '%s': %s", _trainingDir, strerror(errno));
		return -1;
	}
	
	std::string dir = _trainingDir;
	// write current settings
	std::string settings_path = dir + "settings.st";
	_SETTINGS_OBJ->saveToFile(settings_path.c_str(), false);

	// write scripts for plotting and monitoring
	for(int i = 0; i < SCRIPT_NUM; i++)
	{
		std::string script_path = dir + SCRIPT_NAME_ARRAY[i];
		FILE * f = fopen(script_path.c_str(), "w");
		if(f){
			fputs(SCRIPT_ARRAY[i], f);
			fclose(f);
		}
		else{
			ERROR_F("Unable to write script '%s': %s", script_path.c_str(), strerror(errno));
		}
	}

	_agentPath = agent_path;
	std::string agent_text;
	if(zStringTools::loadFromFile(agent_path, &agent_text) < 0){
		ERROR_F("Unable to open agent script '%s' for copying!", agent_path);
	}
	else{
		std::string agent_copy_path = dir + "agent.py";
		if(zStringTools::storeToFile(agent_copy_path.c_str(), agent_text.c_str()) < 0){
			ERROR_F("Unable to copy agent script to '%s'!", agent_copy_path.c_str());
		}
	}

	// open csv writer
	std::string data_path = dir + "data.csv";
	if(_csvWriter.open(data_path.c_str()))
	{
		ERROR_F("Could not open csv file '%s' for writing (%s)!", data_path.c_str(), strerror(errno));
		return -1;
	}

	return 0;
}

void Arena::run(){
	_updateTimer.reset();
	_physicsTimer.reset();
	_videoTimer.reset();
	int iteration = 1;
	int next_video_update_it = 1;// iteration at which to perform the next video update
	int remainder = 0;
	if(_SETTINGS->video.enabled){/* video */
		while(_run){
			_updateTimer.setTargetFPS(_SETTINGS->physics.fps);
			_videoTimer.setTargetFPS(_SETTINGS->video.fps);
			_physicsTimer.setTargetFPS(_SETTINGS->physics.fps);

			if(iteration == next_video_update_it){
				zEventList evt;	
				if(Z_FW->update(&evt) == 0)/* user closed window */
					exitApplication();

				if(_consoleEnabled)
					updateConsole(evt);
				processEvents(evt);

				if(!_videoDisabled){
					Z_FW->clearScreen();
					/* render arena */
					render();

					/* render GUI (console, stats, etc)*/
					renderGUI();
					Z_FW->flipScreen();
				}

				iteration = 0;
				next_video_update_it = 1;
				int update_fps = f_round(_updateTimer.getCurrentFPS());
				if(_SETTINGS->video.fps < update_fps){
					int f = remainder+update_fps;
					next_video_update_it = f/_SETTINGS->video.fps;
					remainder = (f%_SETTINGS->video.fps);
				}
				// measure fps
				_videoTimer.update(false);
			}
			update();
			_updateTimer.update(!((_trainingMode || _playSimulation) && _videoDisabled));// no fps limit if in training mode and video disabled
			iteration++;
		}
	}
	else{/* no video */
		while(_run){
			_updateTimer.setTargetFPS(_SETTINGS->physics.fps);
			_physicsTimer.setTargetFPS(_SETTINGS->physics.fps);
			update();
			_updateTimer.update(!_trainingMode);// no fps limit when in training mode
		}
	}
}

void Arena::updateConsole(zEventList & evt){
	if(_console->update(&evt)){
		const char * cmd = _console->getCommand();
		INFO_F("> %s", cmd);
		command(cmd);
	}
}

void Arena::screenshot(const char * path)
{
	int windowW = Z_FW->getWindowW();
	int windowH = Z_FW->getWindowH();
	SDL_Surface *s = SDL_CreateRGBSurface(0, windowW, windowH, 32,
											0x000000FF,
											0x0000FF00,
											0x00FF0000,
											0xFF000000);
	glReadPixels(0, 0, windowW, windowH, GL_RGBA, GL_UNSIGNED_BYTE, s->pixels);
	int size = windowW*windowH;
	Uint32 * pixels = ((Uint32*)s->pixels);
	for(int i =0; i < size/2; i++){
		int x = i%windowW;
		int y = i/windowW;
		int mirrored_index = (windowH-y-1)*windowW + x;
		Uint32 p = pixels[mirrored_index];
		pixels[mirrored_index] = pixels[i];
		pixels[i] = p;
		pixels[i] |= 0xFF000000;// setting alpha to 255
	}

	for(int i =size/2; i < size; i++){
		pixels[i] |= 0xFF000000;// setting alpha to 255
	}
	if(SDL_SaveBMP(s, path)){
		puts(SDL_GetError());
	}
	SDL_FreeSurface(s);
}

void Arena::printEpisodeResults(float total_reward)
{
	_agentMeasure.calculateMean();
	_agentPostMeasure.calculateMean();
	_simulationMeasure.calculateMean();
	_levelResetMeasure.calculateMean();
	char time_buffer[32];
	Timer::getTimeString(SDL_GetTicks()-_trainingStartTime, time_buffer, 32);
	INFO_F(	"\nEpisode %d over:\n"
			"  Reward:           %.3f\n"
			"  Mean reward:      %.3f (last 100 episodes)\n"
			"  Speed:            %.1f steps/sec (x%d)\n"
			"  Agent Time:       %.1fms = (%.1f + %.1f)ms\n"
			"  Simulation Time:  %.1fms\n"
			"  Level Reset Time: %.1fms\n"
			"  Time elapsed:     %s\n"
			"  Success Rate:     %.0f%%",
		_episodeCount,
		total_reward,
		_meanReward.getMean(),
		_physicsTimer.getCurrentFPS(), _numEnvs,
		_agentMeasure.getMean() + _agentPostMeasure.getMean(),
		_agentMeasure.getMean(), _agentPostMeasure.getMean(),
		_simulationMeasure.getMean(),
		_levelResetMeasure.getMean(),
		time_buffer, _meanSuccess.getMean()*100);
}


void Arena::refreshFPSCounter()
{
	const float realtime = _SETTINGS->physics.time_step*_physicsTimer.getCurrentFPS();
	_metricHandles[VIDEO_FPS]->setRatioValue(_videoTimer.getCurrentFPS(), _SETTINGS->video.fps);
	_metricHandles[PHYSICS_FPS]->setRatioValue(_updateTimer.getCurrentFPS(), _SETTINGS->physics.fps);
	char buffer[32];
	sprintf(buffer, "%.3f x%d", _SETTINGS->physics.time_step, _SETTINGS->physics.step_iterations);
	_metricHandles[STEPS_PER_SECOND]->setStringValue(buffer);
	_simulationMeasure.calculateMean();
	_metricHandles[SIMULATION_TIME]->setFloatValue(_simulationMeasure.getMean(), 1);
	_metricHandles[REALTIME]->setFloatValue(realtime, 2);

	// set python agent time
	if(_pyAgentUsed){
		_agentMeasure.calculateMean();
		_agentPostMeasure.calculateMean();
		const float agent_time = _agentMeasure.getMean() + _agentPostMeasure.getMean();
		_metricHandles[AGENT_TIME]->setFloatValue(agent_time, 1);
	}
	else{
		_metricHandles[AGENT_TIME]->setStringValue("-");
	}
	
	// set training time
	char time_buffer[32];
	Timer::getTimeString(SDL_GetTicks()-_trainingStartTime, time_buffer, 32);
	if(_trainingMode){
		_metricHandles[TIME_ELAPSED]->setStringValue(time_buffer);
	}
	else{
		_metricHandles[TIME_ELAPSED]->setStringValue("-");
	}
}

void Arena::refreshEpisodeCounter(){
	_metricHandles[NUM_EPISODES]->setIntValue(_episodeCount);
}

void Arena::refreshRewardCounter(){
	_metricHandles[MEAN_REWARD]->setFloatValue(_meanReward.getMean(), 2);
	char buffer[32];
	sprintf(buffer, "%d%%", (int)(100*_meanSuccess.getMean()));
	_metricHandles[MEAN_SUCCESS]->setStringValue(buffer);
}

void Arena::refreshLevelResetTime()
{
	_levelResetMeasure.calculateMean();
	_metricHandles[LEVEL_RESET_TIME]->setFloatValue(_levelResetMeasure.getMean(), 1);
}

void Arena::resize(){
	/* update projection matrix */
	int windowW = Z_FW->getWindowW();
	int windowH = Z_FW->getWindowH();
	_RENDERER->refreshProjectionMatrizes(windowW, windowH);

	zRect r;
	_RENDERER->getScreenRect(&r);
	_videoDisabledText->setAlignBox(r);
	_videoDisabledText->align();

	_statsDisplay->refresh();

	// redraw screen if in video disabled mode
	if(_videoDisabled) {
		renderVideoDisabledScreen();
	}
}

