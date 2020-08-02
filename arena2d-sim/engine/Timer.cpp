/* author: Cornelius Marx */
#include "Timer.hpp"

Timer::Timer(int target_fps){
	_targetFPS = 0;
	_currentFrameTime = 0.f;
	_frameTimeBuffer = NULL;
	_lastTicks = SDL_GetTicks();
	_accumFrameTime = 0;
	setTargetFPS(target_fps);
}

void Timer::reset(){
	_lastTicks = SDL_GetTicks();
}

void Timer::setTargetFPS(int target_fps){
	if(target_fps != _targetFPS){// fps has changed
		_targetFPS = target_fps;
		if(_targetFPS < 1){
			_targetFPS = 1;
		}
		delete[]_frameTimeBuffer;
		_frameTimeBuffer = new int[_targetFPS];
		setZeroFPS();
	}
}

void Timer::setZeroTime()
{
	_accumFrameTime = 0;
	_currentFPS = 0.f;
	for(int i = 0; i < _targetFPS; i++){
		_frameTimeBuffer[i] = 0;
	}
	_bufferIndex = 0;
	_currentFrameTime = 0;
	_lastTicks = SDL_GetTicks();
	_offset = 0;
}

// updating timer and set delays so a constant fps is reached
int Timer::update(bool apply_delay){
	Uint32 delta = (SDL_GetTicks()-_lastTicks);
	int delay = (1000/_targetFPS) - delta + _offset;
	if(apply_delay && delay > 0){
		SDL_Delay(delay);	
	}
	Uint32 now = SDL_GetTicks();
	int measured_frame_time = now -_lastTicks;
	_lastTicks = now;

	// remove old frame time
	_accumFrameTime -= _frameTimeBuffer[_bufferIndex];

	// add new frame time
	_frameTimeBuffer[_bufferIndex] = measured_frame_time;
	_accumFrameTime += measured_frame_time;
	_bufferIndex = (_bufferIndex+1)%_targetFPS;

	// adjusting offset
	if(_accumFrameTime > 1000){// too much time needed
		_offset = -1;
	}else{
		_offset = 0;
	}

	// calculating frame time
	_currentFrameTime = _accumFrameTime/static_cast<float>(_targetFPS);
	_currentFPS = 1000.f/_currentFrameTime;// FPS can get infinitely large

	return delay;
}

// check whether last timer update was so long ago that fps is below 1
void Timer::checkLastUpdate(){
	Uint32 now = SDL_GetTicks();
	if(now - _lastTicks >= 1000){
		setZeroFPS();
	}
}

void Timer::setZeroFPS()
{
	_accumFrameTime = 1000*_targetFPS;
	_currentFPS = 0.f;
	for(int i = 0; i < _targetFPS; i++){
		_frameTimeBuffer[i] = 1000;
	}
	_bufferIndex = 0;
	_currentFrameTime = 1000;
	_lastTicks = SDL_GetTicks();
	_offset = 0;
}

void Timer::getTimeString(Uint32 millis, char * buffer, int buffer_size){
	int seconds = millis/1000;
	int minutes = seconds/60;
	int hours = minutes/60;
	int days = hours/24;
	buffer[0] = '\0';
	if(minutes == 0){
		snprintf(buffer, buffer_size, "%ds", seconds);
	}
	else if(hours == 0){
		snprintf(buffer, buffer_size, "%dm %ds", minutes, seconds%60);
	}
	else if(days == 0){
		snprintf(buffer, buffer_size, "%dh %dm %ds", hours, minutes%60, seconds%60);
	}
	else{
		snprintf(buffer, buffer_size, "%dd %dh %dm %ds", days, hours%24, minutes%60, seconds%60);
	}
}
