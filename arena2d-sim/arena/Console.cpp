/*
 * Console.cpp
 *
 *  Created on: May 21, 2018
 *      Author: zer0divider
 */
#include "Console.hpp"

Console::Console(zFont * font, zFontSize size, const char * path_command_history)
{
	//pre text
	_preTextLength = 2;
	_preText = new char[_preTextLength + 1];
	_preText[0] = '>';
	_preText[1] = ' ';
	_preText[2] = '\0';

	//initialize text views
	zColor text_color(CONSOLE_TEXT_COLOR);
	_commandLine = new zTextView(font, size, CONSOLE_MAX_CHARS +_preTextLength +1);
	_commandLine->setColor(text_color);
	_cursor = new zTextView(font, size, "_");
	_cursor->setColor(text_color);
	_currentCommand[0] = '\0';
	_currentCommandLength = 0;
	_cursorPos = 0;

	_isEnabled = false;
	_cursorBlink = false;
	_blinkTimer = 0.f;

	//creating buffer for background
	_boxMargin = 2;

	_historyBrowsing = false;
	_currentHistoryCommand = _commandHistory.end();

	refreshTextAlignment();
	refreshText();
	refreshCursorPosition();
	_ctrlPressed = false;

	// loading commands from history file
	_historyFile = NULL;
	if(path_command_history != NULL){
		INFO_F("Loading command history from '%s'...", path_command_history);
		FILE * f = fopen(path_command_history, "r");
		if(f){
			char line[CONSOLE_MAX_CHARS+1];
			while(fgets(line, CONSOLE_MAX_CHARS+1, f)){
				int line_len = strlen(line);
				if(line[line_len-1] == '\n'){
					line[line_len-1] = '\0';
					line_len--;
				}
				char * cmd = new char[line_len+1];
				memcpy(cmd, line, sizeof(char)*line_len+1);
				_commandHistory.push_front(cmd);
			}
			fclose(f);
		}
		// open same path for writing (appending mode)
		_historyFile = fopen(path_command_history, "a");
		if(_historyFile == NULL)
			WARNING_F("Unable to open command history '%s'in writing mode: %s", path_command_history, strerror(errno));
	}
}

Console::~Console()
{
	// closing history file
	if(_historyFile != NULL)
		fclose(_historyFile);
	delete _commandLine;
	delete _cursor;
	delete []_preText;
	clearHistory();
}

bool Console::update(std::list<SDL_Event>* events)
{
	_blinkTimer -= Z_FW->getDeltaTime();
	if(_blinkTimer < 0.f)
		_blinkTimer = 0.f;
	std::list<SDL_Event>::iterator it = events->begin();
	bool new_command = false;
	while(it != events->end())
	{
		bool fetched = false;//event is catched by console
		switch(it->type)
		{
		case SDL_WINDOWEVENT:
		{
			if(it->window.event == SDL_WINDOWEVENT_RESIZED)
			{
			}
		}break;
		case SDL_TEXTINPUT:
		{
			if(_isEnabled && !_ctrlPressed)
			{
				_historyBrowsing = false;
				insertText(it->text.text);
				fetched = true;
			}
		}break;
		case SDL_KEYDOWN:
		{
			if(_isEnabled)
			{
				SDL_Keycode sym = it->key.keysym.sym;
				Uint16 mod = it->key.keysym.mod;

				fetched = !_ctrlPressed;
				switch(sym)
				{
				case SDLK_LCTRL:
				case SDLK_RCTRL:
				{
					fetched = false;
					_ctrlPressed = true;
				}break;
				case SDLK_BACKSPACE:
				{
					_historyBrowsing = false;
					if(mod & KMOD_CTRL)//delete all
					{
						removeTextBack(_cursorPos);
					}
					else
						removeTextBack(1);
				}break;
				case SDLK_DELETE:
				{
					_historyBrowsing = false;
					removeTextFront(1);
				}break;
				case SDLK_RETURN:
				{
					_historyBrowsing = false;
					new_command = applyCommand();
				}break;
				case SDLK_LEFT:
				{
					if(mod & KMOD_CTRL)//moving to very beginnning with ctrl
					{
						setCursor(0);
					}
					else
						moveCursor(-1);
				}break;
				case SDLK_RIGHT:
				{
					if(mod & KMOD_CTRL)//moving to very end with ctrl
					{
						setCursor(_currentCommandLength);
					}
					else
						moveCursor(1);
					
				}break;
				/* traversing history */
				case SDLK_UP:
				{
					if(_commandHistory.size() == 0)
						break;
					if(!_historyBrowsing){// start browsing
						_historyBrowsing = true;
						_currentHistoryCommand = _commandHistory.begin();
						// setting command
						setToHistoryCommand();
					}
					else if(std::next(_currentHistoryCommand, 1) != _commandHistory.end()){
							_currentHistoryCommand++;
							// setting command
							setToHistoryCommand();
					}
				}break;
				case SDLK_DOWN:
				{
					if(_commandHistory.size() == 0)
						break;
					if(_historyBrowsing){
						if(_currentHistoryCommand != _commandHistory.begin()){
							_currentHistoryCommand--;
							setToHistoryCommand();
						}
					}
				}break;
				case SDLK_v://paste with ctrl+v
				{
					_historyBrowsing = false;
					if(mod & KMOD_CTRL)
					{
						const char * c = SDL_GetClipboardText();
						if(c != NULL)
						{
							insertText(c);
							SDL_free((void*)c);
						}
					}
				}break;
				/* ignore function-keys */
				case SDLK_F1:
				case SDLK_F2:
				case SDLK_F3:
				case SDLK_F4:
				case SDLK_F5:
				case SDLK_F6:
				case SDLK_F7:
				case SDLK_F8:
				case SDLK_F9:
				case SDLK_F10:
				case SDLK_F11:
				case SDLK_F12:
				{
					fetched = false;
				}break;
				}
			}
		}break;
		case SDL_KEYUP:
		{
			SDL_Keycode sym = it->key.keysym.sym;
			if(sym == SDLK_LCTRL || sym == SDLK_RCTRL)
			{
				fetched = false;
				_ctrlPressed = false;
			}
		}break;
		case SDL_USEREVENT:
		{
			if(it->user.code == Z_EVENT_TIMER_HALF && _blinkTimer <= 0.f)
			{
				_cursorBlink = !_cursorBlink;
			}
		}break;
		}
		if(fetched)//remove event from list
			it = events->erase(it);
		else
			it++;
	}
	return new_command;
}

void Console::setToHistoryCommand(){
	assert(_currentHistoryCommand != _commandHistory.end());
	
	setText(*_currentHistoryCommand);
}

void Console::setText(const char * t){
	_currentCommandLength = 0;
	while(*t != '\0' && _currentCommandLength < CONSOLE_MAX_CHARS){
		_currentCommand[_currentCommandLength] = *t;
		t++;
		_currentCommandLength++;
	}
	_currentCommand[_currentCommandLength] = '\0';
	refreshText();
	setCursor(_currentCommandLength);
}

void Console::refreshText()
{
	//concatenating pretext with commandtext
	int t_len = _preTextLength + _currentCommandLength;
	char * t = new char[t_len+1];
	memcpy(t, _preText, _preTextLength);
	memcpy(t+_preTextLength, _currentCommand, _currentCommandLength);
	t[t_len] = '\0';
	//text has changed
	_commandLine->setText(t);

	delete[]t;
}

void Console::insertText(const char * t)
{
	int t_len = strlen(t);
	if(_currentCommandLength+t_len > CONSOLE_MAX_CHARS)//too many characters, adjusting text added
		t_len = CONSOLE_MAX_CHARS -_currentCommandLength;

	if(t_len > 0)
	{
		//creating space by shifting characters
		for(int i = _currentCommandLength-1;  i >= _cursorPos; i--)
		{
			_currentCommand[i+t_len] = _currentCommand[i];
		}
		//insert text
		memcpy(_currentCommand+ _cursorPos, t, t_len);
		_cursorPos+= t_len;
		_currentCommandLength+=t_len;
		_currentCommand[_currentCommandLength] = '\0';
		refreshText();
		refreshCursorPosition();
		_cursorBlink = true;
		_blinkTimer = 1.f;
	}

}

void Console::removeTextBack(int num_chars)
{
	if(num_chars > _cursorPos)
	{
		num_chars = _cursorPos;
	}
	if(num_chars > 0)
	{
		for(int i = _cursorPos-1; i < _currentCommandLength; i++)
		{
			_currentCommand[i] = _currentCommand[i+num_chars];
		}
		_cursorPos -= num_chars;
		_currentCommandLength -= num_chars;
		_currentCommand[_currentCommandLength] = '\0';
		//text has changed
		refreshText();
		refreshCursorPosition();
		_cursorBlink = true;
		_blinkTimer = 1.f;
	}
}

void Console::removeTextFront(int num_chars)
{
	if(num_chars > _currentCommandLength-_cursorPos)
	{
		num_chars = _currentCommandLength-_cursorPos;
	}
	if(num_chars > 0)
	{
		for(int i = _cursorPos; i < _currentCommandLength-num_chars; i++)
		{
			_currentCommand[i] = _currentCommand[i+num_chars];
		}
		_currentCommandLength -= num_chars;
		_currentCommand[_currentCommandLength] = '\0';
		//text has changed
		refreshText();
		refreshCursorPosition();
		_cursorBlink = true;
		_blinkTimer = 1.f;
	}
}

void Console::moveCursor(int dx)
{
	int prev_pos = _cursorPos;
	_cursorPos += dx;
	if(_cursorPos > _currentCommandLength)
		_cursorPos = _currentCommandLength;
	else if(_cursorPos < 0)
		_cursorPos = 0;

	if(prev_pos != _cursorPos)
	{
		refreshCursorPosition();
		_cursorBlink = true;
		_blinkTimer = 1.f;
	}
}

void Console::setCursor(int x)
{
	int prev_pos = _cursorPos;
	_cursorPos = x;
	if(_cursorPos > _currentCommandLength)
		_cursorPos = _currentCommandLength;
	else if(_cursorPos < 0)
		_cursorPos = 0;

	if(prev_pos != _cursorPos)
	{
		refreshCursorPosition();
		_cursorBlink = true;
		_blinkTimer = 1.f;
	}
}

bool Console::applyCommand()
{
	if(_currentCommandLength == 0)
		return false;
	char * new_cmd = new char[_currentCommandLength+1];
	strcpy(new_cmd, _currentCommand);
	_commandHistory.push_front(new_cmd);
	//reset command
	_currentCommandLength = 0;
	_currentCommand[0] = '\0';
	_cursorPos = 0;
	refreshText();
	refreshCursorPosition();
	// writing to history file
	if(_historyFile){
		fputs(new_cmd, _historyFile);
		fputs("\n", _historyFile);
	}
	return true;
}

void Console::clearHistory()
{
	for(std::list<char*>::iterator it = _commandHistory.begin(); it != _commandHistory.end(); it++){
		delete[](*it);
	}
	_commandHistory.clear();
}

void Console::enable()
{
	if(!_isEnabled)
	{
		SDL_StartTextInput();
		_isEnabled = true;

	}
}

void Console::disable()
{
	if(_isEnabled)
	{
		SDL_StopTextInput();
		_isEnabled = false;
	}
}

void Console::refreshTextAlignment()
{
	zGlyphMap * g = _commandLine->getFont()->getGlyphMap(_commandLine->getSize());
	_commandLine->setPosition(static_cast<float>(_boxMargin), static_cast<float>(g->getMaxGlyphTop() + _boxMargin));
	_charHeight = g->getMaxGlyphTop() - g->getMinGlyphBot();
}

void Console::refreshCursorPosition()
{
	zGlyphMap * g = _commandLine->getFont()->getGlyphMap(_commandLine->getSize());
	const char * text = _commandLine->getText();
	zGlyph glyph;
	float x = 0;
	for(int i = 0; i < _cursorPos+2;i++)
	{
		g->getGlyph(static_cast<unsigned char>(text[i]), &glyph);
		x+= glyph.advanceX/64.f;
	}
	_cursor->setPosition(_commandLine->getXPos()+x, _commandLine->getYPos());
}

void Console::refreshBox()
{

}

void Console::renderText()
{
	_commandLine->render();
	if(_cursorBlink)
		_cursor->render();
}

void Console::renderBox()
{
	zMatrix4x4 m;
	//zMatrix4x4 t;
	m.setScale(zVector3D(static_cast<float>(Z_FW->getWindowW()), static_cast<float>(_charHeight+_boxMargin*2), 0.f));
	//m = t * m;
	Z_SHADER->setModelviewMatrix(m);
	Z_SHADER->setColor(zColor(CONSOLE_BACKGROUND_COLOR));
	_RENDERER->bindGO(GO_QUAD);
	_RENDERER->drawGO();
	_RENDERER->resetModelviewMatrix();
}
