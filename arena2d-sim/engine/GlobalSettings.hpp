/* author: Cornelius Marx */
#ifndef GLOBAL_SETTINGS_H
#define GLOBAL_SETTINGS_H

#include <SDL2/SDL.h>

// including settings structs
using namespace std;
#include <string>
#include <settings/SettingsStructs.h>

#include "hashTable.h"
#include <errno.h>
#include "zSingleton.hpp"
#include "zLogfile.hpp"
#include "GamePadButtonCodes.hpp"
#include "zStringTools.hpp"

#define DEFAULT_SETTINGS_FILE_NAME "settings.st"

#define _SETTINGS_OBJ GlobalSettings::get()
#define _SETTINGS GlobalSettings::get()->getSettings()

#define SETTINGS_KEY_NUM 236
//all possible key names in settings file
extern const char * SETTINGS_KEY_NAMES[SETTINGS_KEY_NUM];
#define SETTINGS_GAMEPAD_BUTTON_NUM 14
#define SETTINGS_GAMEPAD_AXIS_NUM 12
#define SETTINGS_GAMEPAD_HAT_NUM 4
#define SETTINGS_GAMEPAD_TOTAL_BUTTON_NUM	SETTINGS_GAMEPAD_BUTTON_NUM+SETTINGS_GAMEPAD_AXIS_NUM+SETTINGS_GAMEPAD_HAT_NUM 
#define SETTINGS_GAMEPAD_KEYCODE_START (SDLK_DELETE+1)
extern const char * SETTINGS_GAMEPAD_BUTTON_NAMES[SETTINGS_GAMEPAD_TOTAL_BUTTON_NUM];// button names in settings file
extern const char * GAMEPAD_BUTTON_NAMES[SETTINGS_GAMEPAD_TOTAL_BUTTON_NUM];
extern const char* GAMEPAD_BUTTON_NAME_UNKNOWN;


#define SETTINGS_TYPE_INT 		0
#define SETTINGS_TYPE_FLOAT 	1
#define SETTINGS_TYPE_STRING	2
#define SETTINGS_TYPE_KEY 		3

struct sSettingsOption{
	int type;	
	void * data;
	void printType(){
		switch(type) {
			case SETTINGS_TYPE_INT:
				printf("INT");break;
			case SETTINGS_TYPE_FLOAT:
				printf("FLOAT");break;
			case SETTINGS_TYPE_STRING:
				printf("STRING");break;
			case SETTINGS_TYPE_KEY:
				printf("KEY");break;
			default:
				printf("UNKNOWN");
		}
	}

};

struct sGamePadButton{
    enum Type{NONE, BUTTON, AXIS, HAT};
	enum State{FALSE=0, PRESSED, RELEASED};
    void setButton(int btn_id){
        t = BUTTON;
        value = btn_id;
    }
    void setHat(Uint8 v){
        t = HAT;
        value = v;
    }
    void setAxis(int axis_id, bool positive){
        t = AXIS;
        value = axis_id*2;
        if(positive) value++;
    }
	static SDL_Keycode getKeycode(const SDL_Event & e);
	static State checkEvent(SDL_Keycode gamepad_button_code, const SDL_Event & e, Uint32 & timestamp);// check whether given event matches this button, FALSE means, this button does not match the event
	static bool isGamepadID(const SDL_Event & e, SDL_JoystickID gamepad_id);
    Type t;
    int value;
};

class GlobalSettings : public zTSingleton<GlobalSettings>
{
public:
    GlobalSettings(): _lastSettingsPath(NULL){setLastSettingsPath(DEFAULT_SETTINGS_FILE_NAME);}
    
    // initialize global settings by loading settings from file, settings file will be created automatically, if no such file can be found
	// @return -1 on error parsing file, 0 on success
    int init(const char * file_name = DEFAULT_SETTINGS_FILE_NAME);
    
    // destructor
    ~GlobalSettings(){delete[]_lastSettingsPath; clear();}
    
    //clear hash tables etc.
    void clear();

    //set to default
	//NOTE: this function is defined in config/SettingsDefault.cpp
    void setToDefault();

    //write current settings to file
	//@param set_last_path if set to true, the last settings path is set to given path
    //returns 1 on error, else 0
    int saveToFile(const char * path = DEFAULT_SETTINGS_FILE_NAME, bool set_last_path = true);
	int save(){return saveToFile(((_lastSettingsPath == NULL) ?DEFAULT_SETTINGS_FILE_NAME : _lastSettingsPath));};

    //load settings from file
    //if @silence set to 1 no error messages are printed via log
    //returns 1 on parse error, -1 on file error, else 0
    int loadFromFile(const char * path, int silence);
	int load(){return loadFromFile(((_lastSettingsPath == NULL) ?DEFAULT_SETTINGS_FILE_NAME : _lastSettingsPath), 0);};

    //loading settings from given string
    //returns 1 on parse error, else 0
    int  loadFromString(const char * c);

    //parsing one line of settings with given context (e.g. variable= "msaa", value = "1" context = "video" -> video.msaa = 1)
    //var_len, value_len, context_len can be set to -1 if length is unknown
    //@context can be set to NULL if no context is given
    //returns 1 on parse error, else 0
    int parseSettingsExpression(const char * variable, int var_len, const char * value, int value_len, const char * context, int context_len);

    // string hash function for quick symbol lookup
    static unsigned int stringHash(const void * key, uint key_len, void * info);

	// get name of key as its called in the setttings.st file
    static const char * getSettingsKeyName(SDL_Keycode code);
    static const char * getSettingsGamePadButtonName(const sGamePadButton & b);

	static const char * getKeyName(SDL_Keycode code);// same as SDL_GetKeyName() but with extended Keycodes (for gamepads)

	f_settings* getSettings(){return &_settings;}

	const char * getLastSettingsPath(){return _lastSettingsPath;}
    
private:
	void setLastSettingsPath(const char * p);
	// filling hash tables
	void initSymbolTable();
    void initKeyNamesTable();

    //writing current settings
    void writeToFile(FILE * f);


    //printing hash (for testing distribution)
    void printHashTable(void * data, void * info);

    // settings instance
    f_settings _settings;
    
    sHashTable * _hashTable; // storing settings symbols
    sHashTable * _keyNamesTable;
	char * _lastSettingsPath;
    
};

#endif
