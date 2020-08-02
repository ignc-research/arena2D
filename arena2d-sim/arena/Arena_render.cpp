/* Author: Cornelius Marx */

#include "Arena.hpp"

void Arena::render()
{
	Quadrangle q;
	_RENDERER->getTransformedGLRect(_camera.getInverseMatrix(), &q);
	zRect r;
	q.getAABB(&r);
	int screen_w = Z_FW->getWindowW();
	int screen_h = Z_FW->getWindowH();
	int w_per_env = screen_w/_envsX;
	int h_per_env = screen_h/_envsY;
	int w_last_env = w_per_env + screen_w%_envsX;
	int h_last_env = h_per_env + screen_h%_envsY;
	for(int i = 0; i < _numEnvs; i++){
		int x_index = i%_envsX;
		int y_index = _envsY - 1 -(i/_envsY);
		int w =  x_index == (_envsX-1) ? w_last_env : w_per_env;
		int h =  y_index == (_envsY-1) ? h_last_env : h_per_env;
		int x = x_index*w_per_env;
		int y = y_index*h_per_env;
		_RENDERER->refreshProjectionMatrizes(w, h);
		glViewport(x, y, w, h);
		_envs[i].render(_camera, r);
	}
	glViewport(0, 0, screen_w, screen_h);
	_RENDERER->refreshProjectionMatrizes(screen_w, screen_h);

	// render environment grid
	if(_envGridBuffer)
	{
		_RENDERER->useColorShader();
		Z_SHADER->setColor(zColor::BLACK);
		_RENDERER->resetCameraMatrix();
		_RENDERER->resetProjectionMatrix();
		_RENDERER->resetModelviewMatrix();
		Z_SHADER->enableVertexArray(true);
		glBindBuffer(GL_ARRAY_BUFFER, _envGridBuffer);
		Z_SHADER->setVertexAttribPointer2D(0, 0);
		glDrawArrays(GL_LINES, 0, _envGridBufferCount);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		Z_SHADER->enableVertexArray(false);
	}
}

void Arena::renderVideoDisabledScreen()
{
	Z_FW->clearScreen();
	render();
	renderGUI();
	// render overlay
	int ms = glIsEnabled(GL_MULTISAMPLE);
	if(ms)
		glDisable(GL_MULTISAMPLE);

	// render background
	_RENDERER->useColorShader();
	_RENDERER->setScreenMatrix();
	_RENDERER->resetCameraMatrix();
	Z_SHADER->setColor(0.2f, 0.2f, 0.2f, 0.8f);
	zRect r;
	_RENDERER->getScreenRect(&r);
	_RENDERER->set2DTransform(zVector2D(r.x, r.y), zVector2D(r.w*2, r.h*2));
	_RENDERER->bindGO(GO_CENTERED_QUAD);
	_RENDERER->drawGO();

	// render text
	_RENDERER->useTextShader();
	_RENDERER->setScreenMatrix();
	_RENDERER->resetCameraMatrix();
	_videoDisabledText->render();

	if(ms)
		glEnable(GL_MULTISAMPLE);
	Z_FW->flipScreen();
}

void Arena::renderGUI()
{
	// disable multisampling for gui
	int ms = glIsEnabled(GL_MULTISAMPLE);
	if(ms)
		glDisable(GL_MULTISAMPLE);

	/*** render stats ***/
	if(_SETTINGS->gui.show_stats){
		_RENDERER->useColorShader();
		_RENDERER->resetCameraMatrix();
		_RENDERER->setScreenMatrix();
		_statsDisplay->renderBackground();

		_RENDERER->useTextShader();
		_RENDERER->setScreenMatrix();
		_RENDERER->resetCameraMatrix();
		_statsDisplay->renderText();
	}
	
	/*** render console ***/
	if(_consoleEnabled){
		// render background
		_RENDERER->useColorShader();
		_RENDERER->setScreenMatrix();
		_RENDERER->resetCameraMatrix();
		_console->renderBox();

		// render text
		_RENDERER->useTextShader();
		_RENDERER->setScreenMatrix();
		_RENDERER->resetCameraMatrix();
		_console->renderText();
	}
	/************************/

	// restore multisampling
	if(ms)
		glEnable(GL_MULTISAMPLE);
}
