/* author: Cornelius Marx */
#ifndef TIMER_H
#define TIMER_H

#include <SDL2/SDL.h>

class Timer{
public:
	Timer():Timer(60){}
	Timer(int target_fps);

	~Timer(){
		delete[](_frameTimeBuffer);
	}

	// reset time measurement
	void reset();

	// set desired fps
	void setTargetFPS(int target_fps);

	// updating timer and set delays so a constant fps is reached
	int update(bool apply_delay = false);

	// check whether last timer update was so long ago that fps is below 1
	void checkLastUpdate();

	// set timer to none-measurable
	void setZeroFPS();

	// set time to 0
	void setZeroTime();

	// getter
	float getCurrentFPS(){return _currentFPS;}
	float getCurrentFrameTime(){return _currentFrameTime;}

	// create a string representing the current time (days, hours, minutes, seconds)
	static void getTimeString(Uint32 millis, char * buffer, int buffer_size);
private:
	int _targetFPS;
	//int _remainder;
	int _offset;
	float _currentFrameTime;
	float _currentFPS;
	Uint32 _lastTicks;
	int *_frameTimeBuffer; // storing frametimes of 1 second
	int _bufferIndex;
	int _accumFrameTime;
};

#endif
