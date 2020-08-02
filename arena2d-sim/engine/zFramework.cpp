#include "zFramework.hpp"

zFramework::zFramework()
{
	_windowW =0;
	_windowH =0;
	_mainWindow = NULL;
	_glContext = 0;
	_isFullscreen = false;
	_multisamples = 0;
	_currentTime = 0.f;
	_lastTime = 0.f;
	_desiredFPS = 60;
	_currentFPS = 0.f;
	_currentTicks = 0;
	_frameCount = 0;
	_delayRemainder = 0;
	_lastMeasureTime = 0;
}

void zFramework::shutDown()
{
	delete[] _gamepads;
	SDL_Quit();
}

int zFramework::init(int windowW, int windowH, int windowX, int windowY, int multisamples, unsigned int flags, bool verbose)
{
	INFO("Initializing Framework...");
	//initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		ERROR_F("Failed to initialize SDL: %s", SDL_GetError());
		return -1;
	}

	if(SDL_InitSubSystem(SDL_INIT_JOYSTICK) < 0){
		ERROR_F("Could not initialize SDL Joystick subsystem: %s", SDL_GetError());
		// do not return, for as the game might run even without joysticks
	}

	if(SDL_InitSubSystem(SDL_INIT_HAPTIC) < 0){
		ERROR_F("Could not initialize SDL Haptic subsystem: %s", SDL_GetError());
		// do not return, for as the game might run even without joysticks
	}

	//setting flags
	_isFullscreen = false;
	unsigned int sdl_flags = SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_OPENGL;
	//getting current screen resolution if fullscreen activated
	if (flags & Z_FULLSCREEN)
	{
		sdl_flags |= SDL_WINDOW_FULLSCREEN;
		_isFullscreen = true;

		if(windowW <= 0 || windowH <= 0){
			//getting current display-resolution
			SDL_DisplayMode current;
			if (SDL_GetDesktopDisplayMode(0, &current) != 0){
				WARNING_F("Could not retrieve current display resolution: %s", SDL_GetError());
			}
			else{
				windowW = current.w;
				windowH = current.h;
			}
		}
	}
	else if(flags & Z_RESIZABLE)
	{
		sdl_flags |= SDL_WINDOW_RESIZABLE;
	}

	if(flags & Z_MAXIMIZED)
	{
		sdl_flags |= SDL_WINDOW_MAXIMIZED;
	}

	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	_multisamples = multisamples;
	if (multisamples > 1)
	{
		SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
		SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, multisamples);
	}
	else
		_multisamples = 0;

	if(verbose)
		INFO_F("-> Creating window (%dx%d)", windowW, windowH);

	//creating window
	if(windowX < 0)
		windowX = SDL_WINDOWPOS_CENTERED;
	if(windowY < 0)
		windowY = SDL_WINDOWPOS_CENTERED;

	if(windowW <= 0 || windowH <= 0){
		windowW = 800;
		windowH = 600;
	}

	_mainWindow = SDL_CreateWindow(APPLICATION_NAME, windowX, windowY, windowW, windowH, sdl_flags);

	if (_mainWindow == NULL)//failed to create a window
	{
		ERROR_F("While creating window: %s", SDL_GetError());
		return -1;
	}

	if(verbose)
		INFO("-> Creating OpenGL context");
	//create gl-context
	_glContext = SDL_GL_CreateContext(_mainWindow);
	if (_glContext == 0)
	{
		ERROR_F("While creating OpenGL Context: %s", SDL_GetError());
		return -1;
	}

	// activate/deactivate vsync
	setVSync((flags&Z_V_SYNC) != 0);

	//init glew
	if(verbose)
		INFO("-> Initializing GLEW");
	glewExperimental = GL_TRUE;// support experimental drivers
	GLenum glew_res = glewInit();
	if (glew_res != GLEW_OK)
	{
		ERROR_F("While initializing GLEW: %s", (const char *)glewGetErrorString(glew_res));
		return -1;
	}
	if(verbose)
		INFO_F("-> Running on OpenGL-Version: %s", (const char *)glGetString(GL_VERSION));

	//init GL parameters
	glClearColor(1.f, 1.f, 1.f, 1.f);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	if (_multisamples){
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_SAMPLE_SHADING);
		glMinSampleShading(1.f);
	}

	// opening joysticks
	_numJoysticks = SDL_NumJoysticks();
	_gamepads = new GamePad[_numJoysticks];
	if(verbose)
		INFO_F("-> Opening Joysticks: %d found", _numJoysticks);
	for(int i = 0; i < _numJoysticks; i++){
		_gamepads[i]._pad = SDL_JoystickOpen(i);
		if(_gamepads[i]._pad == NULL){
			ERROR_F("Unable to open Joystick %d: %s", i, SDL_GetError());
		}
		else{
			_gamepads[i]._haptic = SDL_HapticOpenFromJoystick(_gamepads[i]._pad);
			if(_gamepads[i]._haptic == NULL){
				WARNING_F("No haptic device found for joystick %d", i);
			}
			else{
				SDL_HapticRumbleInit(_gamepads[i]._haptic);
			}
		}
	}
	

	//setting viewport
	_resizeWindow(windowW, windowH);
	return 0;
}

GamePad* zFramework::getGamePadFromID(SDL_JoystickID id)
{
	for(int i = 0; i < _numJoysticks; i++){
		if(id == _gamepads[i].getInstanceID()){
			return &_gamepads[i];
		}
	}
	return NULL;
}

int zFramework::recreateWindow(int windowW, int windowH, unsigned int flags)
{
	_isFullscreen = false;
	Uint32 sdl_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_ALLOW_HIGHDPI;
	_windowW = windowW;
	_windowH = windowH;
	if(flags & Z_FULLSCREEN){
		sdl_flags |= SDL_WINDOW_FULLSCREEN;
		_isFullscreen = true;
		// getting current display resolution
		if(windowW <= 0 || windowH <= 0){
			SDL_DisplayMode current;
			if (SDL_GetDesktopDisplayMode(0, &current) != 0){
				WARNING_F("Could not retrieve current display resolution: %s", SDL_GetError());
			}
			else{
				_windowW = current.w;
				_windowH = current.h;
			}
		}
	}
	else if(flags & Z_RESIZABLE){
		sdl_flags |= SDL_WINDOW_RESIZABLE;
		if(_windowW <= 0 || windowH <= 0){
			_windowW = 800;
			_windowH = 600;
		}
	}

	// destroy old window
	SDL_DestroyWindow(_mainWindow);
	
	// creating new window
	_mainWindow = SDL_CreateWindow(APPLICATION_NAME, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _windowW, _windowH, sdl_flags);
	if(_mainWindow == NULL){
		INFO_F("Error while creating new window: %s", SDL_GetError());
		return -1;
	}

	// setting openGL context
	if(SDL_GL_MakeCurrent(_mainWindow, _glContext)) {
		INFO_F("Error while setting current OpenGL Context: %s", SDL_GetError());
		return -1;
	}

	// restore vsync setting
	setVSync(_vsync);

	// send resize event
	SDL_Event evt;
	evt.type = SDL_WINDOWEVENT;
	evt.window.event = SDL_WINDOWEVENT_RESIZED;
	evt.window.data1 = _windowW;
	evt.window.data2 = _windowH;
	SDL_PushEvent(&evt);
	_resizeWindow(_windowW, _windowH);

	return 0;
}

void zFramework::clearScreen()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


int zFramework::update(std::list<SDL_Event> * event_list)
{
	//first time fetch
	Uint32 lastTicks = _currentTicks;

	/// timing is now done in separate class 'Timer'
	/*
	//FPS-Lock
	Uint32 t = SDL_GetTicks();
	if(_desiredFPS > 0){
		int sec = (1000+_delayRemainder);
		int frame_time = 1000/_desiredFPS;
		_delayRemainder = sec%_desiredFPS;
		int delay = frame_time -(t-lastTicks);
		if(delay > 0)
			SDL_Delay(delay);
	}
	*/

	//updating time
	_currentTicks = SDL_GetTicks();
	_lastTime = _currentTime;
	_currentTime = _currentTicks/1000.f;


	//tick events
	if(lastTicks/500 != _currentTicks/500)//half second has past, generate tick-event
	{
		Uint32 now = SDL_GetTicks();
		_currentFPS = _frameCount/((now-_lastMeasureTime)/1000.0f);
		_frameCount = 0;
		_lastMeasureTime = now;
		SDL_Event user_event;
		user_event.type = SDL_USEREVENT;
		user_event.user.code = Z_EVENT_TIMER_HALF;
		SDL_PushEvent(&user_event);
		if(lastTicks/1000 != _currentTicks/1000)//full second has past
		{
			user_event.user.code = Z_EVENT_TIMER_FULL;
			SDL_PushEvent(&user_event);
		}
	}

	SDL_Event e;
	int res = 1;
	//polling events
	while(SDL_PollEvent(&e))
	{
		bool forward_event = true;
		switch(e.type)
		{
		case SDL_QUIT:
		{
			forward_event = false;
			res = 0;
		}break;
		case SDL_WINDOWEVENT:
		{
			if(e.window.event == SDL_WINDOWEVENT_RESIZED)
			{
				forward_event = true;//forwarding event so user can react to window resizing (e.g. update perspective matrix)
				_resizeWindow(e.window.data1, e.window.data2);
			}
			else if(e.window.event == SDL_WINDOWEVENT_ENTER){
				_mouseOverWindow = true;
			}
			else if(e.window.event == SDL_WINDOWEVENT_LEAVE){
				_mouseOverWindow = false;
			}
		}break;
		default:
		{
			forward_event = true;
		}
		}

		if(forward_event && event_list != NULL)//event shall be forwarded to user
		{
			event_list->push_back(e);	
		}
	}

	//getting current mouse position
	SDL_GetMouseState(_mousePosition+0, _mousePosition+1);
	SDL_GetRelativeMouseState(_relativeMousePosition+0, _relativeMousePosition+1);
	return res;
}

void zFramework::flipScreen()
{
	SDL_GL_SwapWindow(_mainWindow);
	_frameCount++;
}

void zFramework::getMousePos(int * x, int * y)
{
	if(x != NULL)
		*x = _mousePosition[0];
	if(y != NULL)
		*y = _mousePosition[1];
}

void zFramework::getRelMousePos(int * dx, int * dy)
{
	if(dx != NULL)
		*dx = _relativeMousePosition[0];
	if(dy != NULL)
		*dy = _relativeMousePosition[1];
}

void zFramework::_resizeWindow(int new_w, int new_h)
{
	_windowW = new_w;
	_windowH = new_h;
	glViewport(0, 0, new_w, new_h); 
}

void zFramework::setDisplayMode(SDL_DisplayMode * mode)
{
	SDL_SetWindowDisplayMode(_mainWindow, mode);
	_resizeWindow(mode->w, mode->h);
}
