/*
created on: 24th March 2018
author: Cornelius Marx
description: 
	- framework is a singleton class that handles the main window events
*/

#ifndef Z_FRAMEWORK_H
#define Z_FRAMEWORK_H

#include <SDL2/SDL.h>
#include "glew/glew.h"
#include <stdio.h>
#include <list>
#include "zLogfile.hpp"
#include <assert.h>
#include "f_math.h"


//quick access
#define Z_FW zFramework::get()

//flags for window creation
#define Z_FULLSCREEN	0x00000001//window in fullsreen mode
#define Z_RESIZABLE		0x00000002//window can be resized
#define Z_V_SYNC		0x00000004//vertical synchronisation
#define Z_MAXIMIZED		0x00000008// starting window maximized

#define Z_EVENT_TIMER_HALF 1 //half second tick
#define Z_EVENT_TIMER_FULL 2 //full second tick

#define APPLICATION_NAME "Arena2D"

struct GamePad{
	GamePad(): _pad(NULL), _haptic(NULL){}
	SDL_Joystick * _pad;
	SDL_Haptic * _haptic;
	bool rumble(float strength, Uint32 duration){
		if(_haptic == NULL)
			return false;
		return SDL_HapticRumblePlay(_haptic, strength, duration) == 0;
	}
	~GamePad(){
		if(SDL_JoystickGetAttached(_pad))// close joystick
			SDL_JoystickClose(_pad);

		if(_haptic != NULL)
			SDL_HapticClose(_haptic);
	}

	SDL_JoystickID getInstanceID(){
		if(_pad != NULL){
			return SDL_JoystickInstanceID(_pad);
		}
		return -1;
	}
};

typedef std::list<SDL_Event> zEventList;
class zFramework : public zTSingleton<zFramework>
{
public:
	//constructor/destructor
	zFramework();
	~zFramework(){shutDown();}

	//shut down framework by quitting sdl and deleting the glcontext
	void shutDown();

	//initialize framework with given window size and flags
	// if windowX/windowY < 0 -> window is positioned at center
	//@return -1 on error, 0 on success
	int init(int windowW, int windowH, int windowX, int windowY, int multisamples, unsigned int flags, bool verbose = true);

	// recreating window with new attributes
	// if windowW/windowH <= 0 a default resolution will be set (800x600 for non-fullscreen, and current desktor resolution for fullscreen)
	//return: -1 on error, 0 on success
	int recreateWindow(int windowW, int windowH, unsigned int flags);

	//clearing color and depth buffer
	//call this function before next frame is rendered
	void clearScreen();

	//updating timer and events
	//events that are not handled by the framework itself will be put in @event_list (if not NULL)
	//@return 0 if user closed the application
	int update(std::list<SDL_Event> * event_list = NULL);

	//switching buffers so the next rendered image is shown on the screen
	//call this function after every draw-command
	void flipScreen();

	//get times
	float getCurrentTime(){return _currentTime;}
	float getDeltaTime(){
		return _currentTime-_lastTime;
	}
	//get an average fps-rate of the last 0.5 seconds
	float getCurrentFPS(){return _currentFPS;}

	//get window dimensions
	int getWindowW(){return _windowW;}
	int getWindowH(){return _windowH;}
	float getAspectWH(){return _windowW/static_cast<float>(_windowH);}
	float getAspectHW(){return _windowH/static_cast<float>(_windowW);}
	int getMinWindowDim(){return _windowH < _windowW ? _windowH : _windowW;}
	bool isFullscreen(){return _isFullscreen;}

	//get mouse positions
	void getMousePos(int * x, int * y);
	int getMouseX(){return _mousePosition[0];}
	int getMouseY(){return _mousePosition[1];}
	float getHomogeneMouseX(){return 2*_mousePosition[0]/static_cast<float>(_windowW) -1.f;}
	float getHomogeneMouseY(){return -2*_mousePosition[1]/static_cast<float>(_windowH) +1.f;}
	bool isMouseOverWindow(){return _mouseOverWindow;}
	//get relative mouse positions
	void getRelMousePos(int * dx, int * dy);
	int getRelMouseX(){return _relativeMousePosition[0];}
	int getRelMouseY(){return _relativeMousePosition[1];}

	//Lock framerate to given value (0 -> no lock)
	void setFPSLock(int desiredFPS){_desiredFPS = desiredFPS;}
	int getDesiredFPS(){return _desiredFPS;}

	void setVSync(bool on){SDL_GL_SetSwapInterval(on ? 1 : 0); _vsync = on;}

	int getNumGamePads(){return _numJoysticks;}
	GamePad* getGamePad(int index){assert(index >= 0 && index < _numJoysticks); return &_gamepads[index];}
	GamePad* getGamePadFromID(SDL_JoystickID id);
	// returns != NULL if there is a gamepad at given index, that matches the system id
	GamePad* getGamePadFromID(int index, SDL_JoystickID id){GamePad * g = getGamePadFromID(id);return g == &_gamepads[index] ? g : NULL;}
	void setDisplayMode(SDL_DisplayMode * mode);
	SDL_Window* getMainWindow(){return _mainWindow;}
private:
	///private functions
	//called when window was resized (SDL_WINDOWEVENT_RESIZED)
	void _resizeWindow(int new_w, int new_h);


	//current dimensions of main window
	int _windowW;
	int _windowH;
	//SDL structs for handling main window
	SDL_Window * _mainWindow;
	SDL_GLContext _glContext;
	bool _isFullscreen;//window is in fullscreen
	bool _vsync;
	int _multisamples;//number of multisamples (0 -> multisampling deactivated)

	//timing
	Uint32 _currentTicks;
	float _currentTime;
	float _lastTime;
	float _currentFPS;
	Uint32 _lastMeasureTime; // last time the fps was measured
	int _desiredFPS;//FPS that we want to have (default 60), can be set to 0 for no FPS-lock
	int _frameCount;
	int _delayRemainder;

	//mouse position
	int _mousePosition[2];
	int _relativeMousePosition[2];
	bool _mouseOverWindow;// is mouse over window (aka mouse position valid)?

	// gamepads
	int _numJoysticks;
	GamePad * _gamepads;
};


#endif
