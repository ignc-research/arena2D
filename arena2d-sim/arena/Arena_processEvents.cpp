/* Author: Cornelius Marx */
#include "Arena.hpp"

void Arena::processEvents(zEventList & evtList)
{
	for(zEventList::iterator it = evtList.begin(); it != evtList.end(); it++)
	{
		switch(it->type)
		{
		/* window resize */
		case SDL_WINDOWEVENT: {
			if(it->window.event == SDL_WINDOWEVENT_RESIZED){
				_SETTINGS->video.resolution_w = it->window.data1;
				_SETTINGS->video.resolution_h = it->window.data2;
				resize();
			}
			else if(it->window.event == SDL_WINDOWEVENT_MOVED){
				_SETTINGS->video.window_x = it->window.data1;
				_SETTINGS->video.window_y = it->window.data2;
				_SETTINGS->video.maximized = 0;
			}
			else if(it->window.event == SDL_WINDOWEVENT_MAXIMIZED){
				_SETTINGS->video.maximized = 1;
				_SETTINGS->video.window_x = it->window.data1;
				_SETTINGS->video.window_y = it->window.data2;
			}
			else if(it->window.event == SDL_WINDOWEVENT_RESTORED){
				_SETTINGS->video.maximized = 0;
			}
			else if(it->window.event == SDL_WINDOWEVENT_EXPOSED){
				if(_videoDisabled) {// redraw screen
					renderVideoDisabledScreen();
				}
			}
		}break;
		case SDL_USEREVENT:{
			if(it->user.code == Z_EVENT_TIMER_HALF){
				refreshFPSCounter();
			}
		}break;
		case SDL_MOUSEMOTION:{
			zVector2D t(it->motion.xrel, it->motion.yrel);
			_RENDERER->toGLCoordinates(t, true);
			if(_translateCamera){
				t.x *= _envsX;
				t.y *= _envsY;
				_camera.translateScaled(-t);
				_camera.refresh();

				//updating settings
				_SETTINGS->gui.camera_x = _camera.getPos().x;
				_SETTINGS->gui.camera_y = _camera.getPos().y;
			}
			if(_rotateCamera){
				_camera.rotate(t.x);
				_camera.refresh();

				//updating settings
				_SETTINGS->gui.camera_rotation = f_deg(_camera.getRotation());
			}
		}break;
		case SDL_MOUSEBUTTONUP:
		case SDL_MOUSEBUTTONDOWN:{
			if(it->button.button == SDL_BUTTON_RIGHT){
				_translateCamera = (it->button.type == SDL_MOUSEBUTTONDOWN);
			}
			else if(it->button.button == SDL_BUTTON_MIDDLE){
				_rotateCamera = (it->button.type == SDL_MOUSEBUTTONDOWN);
			}
		}break;
		case SDL_MOUSEWHEEL:{
			_camera.zoomExp(it->wheel.y);
			_camera.refresh();

			// updating settings
			_SETTINGS->gui.camera_zoom = _camera.getZoom();
		}break;
		case SDL_KEYUP:
		case SDL_KEYDOWN:{
			SDL_Keycode sym = it->key.keysym.sym;
			Uint16 mod = it->key.keysym.mod;
			bool down = (it->key.type == SDL_KEYDOWN);
			bool event = (down && it->key.repeat == 0);
			if(sym == ARENA_HOTKEY_CONSOLE)//toggle console
			{
				if(event){
					_consoleEnabled = !_consoleEnabled;
				}
			}
			else if(sym == ARENA_HOTKEY_SHOW_STATS)//toggle video
			{
				if(event){
					if(_SETTINGS->gui.show_stats == 0)
						_SETTINGS->gui.show_stats = 1;
					else		
						_SETTINGS->gui.show_stats = 0;
				}
			}
			else if(sym == ARENA_HOTKEY_DISABLE_VIDEO)//toggle video
			{
				if(event){
					_videoDisabled = !_videoDisabled;
					if(_videoDisabled){// video is now disabled -> render overlay
						renderVideoDisabledScreen();
					}
				}
			}
			else if(sym == ARENA_HOTKEY_SCREENSHOT)// screenshot
			{
				if(event){
					INFO("Saving screenshot to 'arena2d.bmp'");
					screenshot("arena2d.bmp");
				}
			}
			else if(sym == _SETTINGS->keys.up){
				if(_trainingMode){
					if(event)
						showTrainingModeWarning();
				}
				else
					_keysPressed[UP] = down;
			}
			else if(sym == _SETTINGS->keys.left){
				if(_trainingMode){
					if(event)
						showTrainingModeWarning();
				}
				else
					_keysPressed[LEFT] = down;
			}
			else if(sym == _SETTINGS->keys.down){
				if(_trainingMode){
					if(event)
						showTrainingModeWarning();
				}
				else
					_keysPressed[DOWN] = down;
			}
			else if(sym == _SETTINGS->keys.right){
				if(_trainingMode){
					if(event)
						showTrainingModeWarning();
				}
				else
					_keysPressed[RIGHT] = down;
			}
			else if(sym == _SETTINGS->keys.reset){
				if(event){
					if(_trainingMode){
						showTrainingModeWarning();
					}
					else{
						reset(true);
					}
				}
			}
			else if(event && sym == SDLK_s && (mod & KMOD_CTRL)){
				command("save_settings");
			}
			else if(event && sym == SDLK_l && (mod & KMOD_CTRL)){
				command("load_settings");
			}
			else if(sym == _SETTINGS->keys.play_pause_simulation)
			{
				if(!_trainingMode && event){
					_playSimulation = !_playSimulation;
					INFO(_playSimulation ? "Simulation play!" : "Simulation stop!");
				}
			}
		}break;
		}
	}
}
