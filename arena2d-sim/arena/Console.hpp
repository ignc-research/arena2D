/*
 * Console.h
 *
 *  Created on: May 21, 2018
 *      Author: zer0divider
 */

#ifndef CONSOLE_H_
#define CONSOLE_H_

#include <engine/zTextView.hpp>
#include <engine/zFramework.hpp>
#include <engine/Renderer.hpp>
#include <list>

#define CONSOLE_TEXT_COLOR			0xFFFFFFFF
#define CONSOLE_BACKGROUND_COLOR	0x404040D0
#define CONSOLE_MAX_CHARS 256
class Console
{
public:
	//constructor
	Console(zFont * font, zFontSize size, const char * path_command_history = NULL);
	~Console();

	//updating text input
	//@return true if a new command was triggered
	bool update(std::list<SDL_Event> *events);

	void enable();//enable console
	void disable();//disable console
	bool isEnabled(){return _isEnabled;}

	//refreshing visual text if e.g. font-size has changed
	void refreshFont();
	void refreshFont(zFontSize new_size);

	void refreshText();//called whenever the textview needs to be updated
	void refreshTextAlignment();
	void refreshCursorPosition();//call after text has been refreshed
	void refreshBox();

	void renderBox();
	void renderText();

	//modifying input at current position
	void insertText(const char * t);
	void removeTextBack(int num_chars);//removing @num_chars characters before cursor
	void removeTextFront(int num_chars);//removing @num_chars characters behind cursor
	bool applyCommand();//triggered when enter is pressed, current command, returns true, if command was not empty
	void moveCursor(int dx);//move cursor left/right about the given amount
	void setCursor(int x);//set cursor to given position
	void setText(const char * t);// set commandline to given text and put cursor at the end

	void clearHistory();
	//getter
	//getting the latest command
	const char * getCommand(){if(_commandHistory.size() > 0)return _commandHistory.front();else return NULL;}
private:
	// set text to currently selected history command (puts cursor at the end)
	void setToHistoryCommand();
	bool _isEnabled;
	zTextView * _commandLine;//visual command text
	zTextView * _cursor;
	std::list<char *> _commandHistory;//first element is lastly triggered command
	std::list<char *>::iterator _currentHistoryCommand;
	char * _preText;// "> "
	int _preTextLength;
	FILE * _historyFile;
	char _currentCommand[CONSOLE_MAX_CHARS+1];
	int _currentCommandLength;
	int _charWidth;//character width in pixels
	int _charHeight;//character height in pixels
	int _cursorPos;//current cursor position (0 is before first character)
	bool _cursorBlink;
	bool _historyBrowsing;
	float _blinkTimer;
	int _boxMargin;
	bool _ctrlPressed;
};

#endif /* CONSOLE_H_ */
