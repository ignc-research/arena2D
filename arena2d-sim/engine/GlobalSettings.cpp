/* author: Cornelius Marx */
#include "GlobalSettings.hpp"

SDL_Keycode sGamePadButton::getKeycode(const SDL_Event & e)
{
	switch(e.type){
	case SDL_JOYBUTTONUP:
	case SDL_JOYBUTTONDOWN:{
		if(e.jbutton.button >= SETTINGS_GAMEPAD_BUTTON_NUM)
			return SDLK_UNKNOWN;
		return SETTINGS_GAMEPAD_KEYCODE_START + e.jbutton.button;
	}break;
	case SDL_JOYAXISMOTION:{
		if(e.jaxis.axis >= SETTINGS_GAMEPAD_AXIS_NUM)
			return SDLK_UNKNOWN;
		if(e.jaxis.value > 0)
			return SETTINGS_GAMEPAD_KEYCODE_START + e.jaxis.axis*2+1 + SETTINGS_GAMEPAD_BUTTON_NUM;
		else if(e.jaxis.value < 0)
			return SETTINGS_GAMEPAD_KEYCODE_START + e.jaxis.axis*2 + SETTINGS_GAMEPAD_BUTTON_NUM;
	}break;
	case SDL_JOYHATMOTION:{
		if(e.jhat.hat > 0)
			return SDLK_UNKNOWN;
		int s = 0;
		switch(e.jhat.value){
		case SDL_HAT_UP: s = 0;break;
		case SDL_HAT_LEFT: s = 1;break;
		case SDL_HAT_DOWN:s = 2;break;
		case SDL_HAT_RIGHT:s = 3;break;
		default:
			return SDLK_UNKNOWN;
		}
		return SETTINGS_GAMEPAD_KEYCODE_START + s+ SETTINGS_GAMEPAD_BUTTON_NUM + SETTINGS_GAMEPAD_AXIS_NUM;
	}break;
	}

	return SDLK_UNKNOWN;
}

bool sGamePadButton::isGamepadID(const SDL_Event & e, SDL_JoystickID gamepad_id){
	switch(e.type){
		case SDL_JOYBUTTONUP:
		case SDL_JOYBUTTONDOWN: return (gamepad_id == e.jbutton.which);
		case SDL_JOYHATMOTION: return (gamepad_id == e.jhat.which);
		case SDL_JOYAXISMOTION: return (gamepad_id == e.jaxis.which);
	}
	return false;
}

sGamePadButton::State sGamePadButton::checkEvent(SDL_Keycode gamepad_button_code, const SDL_Event & e, Uint32 & timestamp)
{
	switch(e.type){
	case SDL_JOYBUTTONUP:
	case SDL_JOYBUTTONDOWN:{
		// check whether gamepad-btn-code is a button
		if(gamepad_button_code < SETTINGS_GAMEPAD_KEYCODE_START || 
			gamepad_button_code >= SETTINGS_GAMEPAD_KEYCODE_START+SETTINGS_GAMEPAD_BUTTON_NUM)
			return sGamePadButton::FALSE;
		if((gamepad_button_code-SETTINGS_GAMEPAD_KEYCODE_START) == e.jbutton.button){
			timestamp = e.jbutton.timestamp;
			return (e.jbutton.state == SDL_PRESSED) ? sGamePadButton::PRESSED : sGamePadButton::RELEASED;
		}
		return sGamePadButton::FALSE;
	}break;
	case SDL_JOYAXISMOTION:{
		// check whether gamepad-btn-code is an axis
		if(gamepad_button_code < SETTINGS_GAMEPAD_KEYCODE_START+SETTINGS_GAMEPAD_BUTTON_NUM || 
			gamepad_button_code >= SETTINGS_GAMEPAD_KEYCODE_START+SETTINGS_GAMEPAD_BUTTON_NUM+SETTINGS_GAMEPAD_AXIS_NUM)
			return sGamePadButton::FALSE;
		int a = (gamepad_button_code-(SETTINGS_GAMEPAD_KEYCODE_START+SETTINGS_GAMEPAD_BUTTON_NUM));
		if(a/2 == e.jaxis.axis){
			timestamp = e.jaxis.timestamp;
			if(e.jaxis.value == 0){
				return sGamePadButton::RELEASED;
			}
			if(a%2 == 0){// negative axis
				if(e.jaxis.value < 0)
					return  sGamePadButton::PRESSED;
			}else{
				if(e.jaxis.value > 0)
					return  sGamePadButton::PRESSED;
			}
		}
		return sGamePadButton::FALSE;
	}break;
	case SDL_JOYHATMOTION:{
		// check whether gamepad-btn-code is a hat
		if(gamepad_button_code < SETTINGS_GAMEPAD_KEYCODE_START+SETTINGS_GAMEPAD_BUTTON_NUM+SETTINGS_GAMEPAD_AXIS_NUM || 
			gamepad_button_code >= SETTINGS_GAMEPAD_KEYCODE_START+SETTINGS_GAMEPAD_BUTTON_NUM+SETTINGS_GAMEPAD_AXIS_NUM
									+ SETTINGS_GAMEPAD_HAT_NUM)
			return sGamePadButton::FALSE;
		int s = 0;
		switch(e.jhat.value){
		case SDL_HAT_UP: s= 0;break;
		case SDL_HAT_LEFT: s = 1;break;
		case SDL_HAT_DOWN:s = 2;break;
		case SDL_HAT_RIGHT:s = 3;break;
		case SDL_HAT_CENTERED:return sGamePadButton::RELEASED;
		default:
			return sGamePadButton::FALSE;
		}
		timestamp = e.jhat.timestamp;
		if(gamepad_button_code == (s+SETTINGS_GAMEPAD_KEYCODE_START+SETTINGS_GAMEPAD_BUTTON_NUM+SETTINGS_GAMEPAD_AXIS_NUM)){
			return sGamePadButton::PRESSED;
		}
		else return sGamePadButton::RELEASED;
	}break;
	}
	return sGamePadButton::FALSE;
}

unsigned int GlobalSettings::stringHash(const void * key, uint key_len, void * info)
{
	uint hash = 5381;
	uint i = 0;

	const char * c_key = (const char*)key;
	for (i = 0; i < key_len; c_key++, i++)
		hash = ((hash << 5) + hash) + *(c_key);

	return hash;
}

const char * GlobalSettings::getSettingsKeyName(SDL_Keycode code)
{
	if(code >= SDLK_LEFTBRACKET && code <= SDLK_z)//most likely case
		return SETTINGS_KEY_NAMES[38 + (int)code - SDLK_LEFTBRACKET];	

	if(code >= SDLK_SPACE && code <= SDLK_AT)
		return SETTINGS_KEY_NAMES[5 + (int)code - SDLK_SPACE];

	if(code >= SDLK_CAPSLOCK && code <= SDLK_PAGEUP)
		return SETTINGS_KEY_NAMES[71 + (int)code - SDLK_CAPSLOCK];

	if(code >= SDLK_LCTRL && code <= SDLK_RGUI)
		return SETTINGS_KEY_NAMES[202 + (int)code - SDLK_LCTRL];

	if(code == SDLK_BACKSPACE)
		return SETTINGS_KEY_NAMES[1]; 

	if(code == SDLK_TAB)
		return SETTINGS_KEY_NAMES[2]; 

	if(code == SDLK_RETURN)
		return SETTINGS_KEY_NAMES[3]; 
	if(code == SDLK_ESCAPE)
		return SETTINGS_KEY_NAMES[4]; 

	if(code == SDLK_DELETE)
		return SETTINGS_KEY_NAMES[70];

	if(code >= SDLK_END && code <= SDLK_KP_PERIOD)
		return SETTINGS_KEY_NAMES[90 + (int)code - SDLK_END];

	if(code >= SDLK_APPLICATION && code <= SDLK_VOLUMEDOWN)
		return SETTINGS_KEY_NAMES[113 + (int)code - SDLK_APPLICATION];

	if(code >= SDLK_ALTERASE && code <= SDLK_EXSEL)
		return SETTINGS_KEY_NAMES[144 + (int)code - SDLK_ALTERASE];

	if(code >= SDLK_KP_00 && code <= SDLK_KP_HEXADECIMAL)
		return SETTINGS_KEY_NAMES[156 + (int)code - SDLK_KP_00];

	if(code >= SDLK_MODE && code <= SDLK_SLEEP)
		return SETTINGS_KEY_NAMES[210 + (int)code - SDLK_MODE]; 

	if(code == SDLK_KP_COMMA)
		return SETTINGS_KEY_NAMES[142];

	if(code == SDLK_KP_EQUALSAS400)
		return SETTINGS_KEY_NAMES[143];

	// GAMEPAD - code
	if(code >= SETTINGS_GAMEPAD_KEYCODE_START && code < SETTINGS_GAMEPAD_KEYCODE_START+SETTINGS_GAMEPAD_TOTAL_BUTTON_NUM)
		return SETTINGS_GAMEPAD_BUTTON_NAMES[code - SETTINGS_GAMEPAD_KEYCODE_START];

	//unknown key
	return SETTINGS_KEY_NAMES[0];
}

const char * GlobalSettings::getKeyName(SDL_Keycode code)
{
	if(code >= SETTINGS_GAMEPAD_KEYCODE_START && code < SETTINGS_GAMEPAD_KEYCODE_START+SETTINGS_GAMEPAD_TOTAL_BUTTON_NUM){
		return GAMEPAD_BUTTON_NAMES[code-SETTINGS_GAMEPAD_KEYCODE_START];
	}
	else if(code == SDLK_UNKNOWN){
		return "UNKNOWN";
	}
	else
		return SDL_GetKeyName(code);
}

const char * GlobalSettings::getSettingsGamePadButtonName(const sGamePadButton & b)
{
    switch(b.t){
    case sGamePadButton::BUTTON:
        return SETTINGS_GAMEPAD_BUTTON_NAMES[b.value];
    case sGamePadButton::AXIS:
        return SETTINGS_GAMEPAD_BUTTON_NAMES[b.value + SETTINGS_GAMEPAD_BUTTON_NUM];
    case sGamePadButton::HAT:
        return SETTINGS_GAMEPAD_BUTTON_NAMES[b.value+ SETTINGS_GAMEPAD_BUTTON_NUM+SETTINGS_GAMEPAD_AXIS_NUM];
    default:
        return NULL;
    }
}

void GlobalSettings::printHashTable(void * data, void * info)
{
	sSettingsOption * o = (sSettingsOption*)data;
	o->printType();
}

void GlobalSettings::clear()
{
	unsigned int num;
	void ** d = h_free(_hashTable, &num);
	for(unsigned int i = 0; i < num; i++){
		free(d[i]);
	}
	free(d);

	//removing key hash
	d = h_free(_keyNamesTable, &num);
	for(unsigned int i = 0; i < num; i++)
		free(d[i]);
	free(d);
}

int GlobalSettings::init(const char * path)
{
	//set to default settings
	setToDefault();

	//init hash table
	initSymbolTable();

	//initializing key table
	initKeyNamesTable();

	//trying to load settings file
	INFO_F("Loading settings file from '%s'...", path);
	int res = loadFromFile(path, 1);
	if(res < 0) {//error loading
		WARNING_F("Couldn't find settings file, creating default '%s'...", path);
		saveToFile(path);
	}
	setLastSettingsPath(path);
	if(res > 0)// parsing error
	{
		return -1;
	}
	return 0;
}

void GlobalSettings::setLastSettingsPath(const char * p)
{
	delete[]_lastSettingsPath;
	_lastSettingsPath = new char[strlen(p)+1];
	strcpy(_lastSettingsPath, p);
}


int GlobalSettings::loadFromFile(const char * path, int silence)
{
	FILE * f = fopen(path, "rb");//loading binary because of trailing characters in file when reading non-binary
	if(f == NULL)
	{
		if(silence == 0)//print errors?
			WARNING_F("Could not load settings file %s: %s!", path, strerror(errno));	

		return -1;
	}
	//determine file size
	fseek(f, 0L, SEEK_END);
	long int file_size = ftell(f);
	fseek(f, 0L, SEEK_SET);

	char * content = (char*)malloc(file_size+1);
	fread(content, 1, file_size, f);
	content[file_size] = '\0';

	//loading settings
	int res = loadFromString(content);

	free(content);
	setLastSettingsPath(path);

	return res;
}

int GlobalSettings::loadFromString(const char * c)
{
	char buffer[512];
	buffer[0] = '\0';
	int buffer_len = 0;
	static const char * addSymbols = "[]_.";//additional symbols allowed in name
	int num_addSymbols = strlen(addSymbols);
	int mode = 0;//current mode:
					//0:searching for next symbol
					//1:searching for operator
					//2:operator '=' found, searching for value
					//3:value set, waiting for end of line or ';'
	
	sList * stack = list_init();
	int current_line = 1;
	int last_line = 1;
	int error = 0;
	while(!error)
	{
		current_line += zStringTools::skipWhiteLine(&c);
		// check for comment
		if(mode != 2 && *c == '#'){
			while(*c != '\n' && *c != '\0')
				c++;

			current_line += zStringTools::skipWhiteLine(&c);
		}
		if(*c == '\0')
			break;

		switch(mode)
		{
		case 0:
		{
			int count = 0;
			int prev_buffer_len = buffer_len;
			while(zStringTools::isAlphanum(*c) || zStringTools::charIsElementOf(*c, addSymbols, num_addSymbols))
			{
				if(count == 0 && buffer_len > 0)//adding reference operator '.'
					buffer[buffer_len++] = '.';
				buffer[buffer_len++] = *c;
				count++;
				c++;
			}
			buffer[buffer_len] = '\0';
			if(*c == '}')//end of context
			{
				//end of context
				if(stack->last == NULL)//something went wrong, no more elements to pop
				{
					ERROR_F("Line %d: No more contexts to be closed with '}'!", current_line);
					error = 1;
					break;
				}
				//go back to previous context
				int * len = (int*)list_popBack(stack);
				buffer_len = *len;
				buffer[buffer_len] = '\0';
				free(len);
				c++;
				mode = 0;
			}
			else if(count == 0)//nothing read
			{
				ERROR_F("Line %d: Unexpected character '%c' found!", current_line, *c);	
				error = 1;
				break;
			}
			else
			{
				//push on stack
				int * b_len = (int*)malloc(sizeof(int));
				*b_len = prev_buffer_len;
				list_pushBack(stack, b_len);
				mode = 1;
			}

		}break;
		case 1:
		{
			if(*c == '=')//searching for next value
			{
				mode = 2;
				c++;
			}
			else if(*c == '{')
			{
				mode = 0;//back to search for symbols
                c++;
			}
			else
			{
				ERROR_F("Line %d: Expected operator!", current_line);
				error = 1;
				break;
			}
		}break;
		case 2:
		{
			char value_buffer[256];
			int value_buffer_len = 0;
			if(*c == '"'){
				c++;
				while(value_buffer_len < 255 && *c != '"' && *c != '\0' && *c != '\n' && *c != '\r'){
					value_buffer[value_buffer_len++] = *c;
					c++;
				}
				if(*c != '"'){
					WARNING_F("Line %d: String must start and end with '\"'!", current_line);
				}
				else{
					c++;
				}

			}
			else{
				while(value_buffer_len < 255 &&
					(zStringTools::isAlphanum(*c) || zStringTools::charIsElementOf(*c, addSymbols, num_addSymbols)))
				{
					value_buffer[value_buffer_len++] = *c;
					c++;
				}
			}
			if(value_buffer_len >= 255)//buffer exceeded
			{
				ERROR_F("Line %d: Exceeded maximum value length of 255!", current_line);
				error = 1;
				break;
			}
            value_buffer[value_buffer_len] = '\0';
			if(parseSettingsExpression(buffer, buffer_len, value_buffer, value_buffer_len, NULL, 0))
			{
				ERROR_F("Line %d: Parsing failed!", current_line);

				error = 1;
				break;
			}
			mode = 3;
		}break;
		case 3:
		{
			current_line += zStringTools::skipWhiteSpace(&c);
			if(*c == ';')
			{
				mode = 0;
				c++;
			}
			else if(last_line < current_line || *c == '#'){
				mode = 0;
			}else
			{
				ERROR_F("Line %d: Trailing characters found after assigning value!", current_line);
				error = 1;
				break;
			}

			if(stack->last == NULL)
			{
				ERROR_F("Line %d: Context stack error!", current_line);
				error = 1;
				break;
			}
			//go back to previous context
			int * len = (int*)list_popBack(stack);
			buffer_len = *len;
			buffer[buffer_len] = '\0';
			free(len);
		}break;
		default:{
		}break;
		}
		last_line = current_line;
	}

	list_freeAll(stack);
	return error;
}

int GlobalSettings::parseSettingsExpression(const char * variable, int var_len, const char * value, int value_len, const char * context, int context_len)
{
	//checking paramters
	if(var_len < 0)
		var_len = strlen(variable);
	if(value_len < 0)
		value_len = strlen(value);
	
	if(context == NULL)
		context_len = 0;	
	else if(context_len < 0)
		context_len = strlen(context);

	//no value or variable name given?
	if(var_len == 0)
	{
		ERROR("No variable given");
		return 1;
	}
	if(value_len == 0)
	{
		ERROR("No value given");
		return 1;
	}

	//context given?
	int offset = 0;
	if(context_len > 0)
		offset = 1;
	//concatinating context and variable
	char * expression = (char*)malloc(var_len + context_len + 1 + offset);
	if(context_len > 0)
	{
		memcpy(expression, context, context_len);
		expression[context_len] = '.';
	}
	memcpy(expression+context_len+offset, variable, var_len);
	expression[var_len+context_len+offset] = '\0';

	//retrieving variable
	sSettingsOption * o = (sSettingsOption*)h_get(_hashTable, expression,
														var_len + context_len + 1 + offset, NULL);
	if(o == NULL)
	{
		ERROR_F("Unknown variable '%s'", expression);
		return 1;
	}

	int error = 0;
	switch(o->type)
	{
	case SETTINGS_TYPE_INT:
	{
		int d = zStringTools::toInt(value, &error);
		if(error == 0)
			*((int*)o->data) = d;
        else{
            ERROR_F("Unrecognized integral value '%s'!", value);
        }
	}break;
	case SETTINGS_TYPE_FLOAT:
	{
		float d = zStringTools::toFloat(value, &error);
		if(error == 0)
			*((float*)o->data) = d;
        else{
            ERROR_F("Unrecognized floating point value '%s'!", value);
        }
	}break;
	case SETTINGS_TYPE_STRING: {
		*((string*)o->data) = value;
	}break;
	case SETTINGS_TYPE_KEY: {
        SDL_Keycode *k = (SDL_Keycode *) h_get(_keyNamesTable, value, value_len, NULL);
        if (k == NULL) {
            error = 1;
            ERROR_F("Unrecognized key value '%s'!", value);
        }
		else
			*((SDL_Keycode*)o->data) = *k;
	}break;
	default:
	{
		WARNING_F("Type %d is not supported", o->type);
	}
	}

	free(expression);
	return error;
}

int GlobalSettings::saveToFile(const char * path, bool set_last_path)
{
	FILE * f = fopen(path, "w");
	if(f == NULL)
	{
		ERROR_F("Could not write settings file to %s: %s!", path, strerror(errno));	
		return 1;
	}

	//writing out
	writeToFile(f);

	fflush(f);
	fclose(f);
	if(set_last_path)
		setLastSettingsPath(path);
	return 0;
}

void GlobalSettings::initKeyNamesTable()
{
	_keyNamesTable = h_init(97 ,GlobalSettings::stringHash, NULL);
	const char * c;
	int c_len;
	int *key;
	sHashTable * h = _keyNamesTable;

	static const int num_single = 8;
	int single_keys[8] = {SDLK_BACKSPACE, SDLK_TAB, SDLK_RETURN, SDLK_ESCAPE, SDLK_DELETE,
									SDLK_KP_COMMA, SDLK_KP_EQUALSAS400, SDLK_UNKNOWN};
	static const int num_paired = 18;
	int paired_keys[18] = { SDLK_LEFTBRACKET, SDLK_z,
							SDLK_SPACE, SDLK_AT,
							SDLK_CAPSLOCK, SDLK_PAGEUP,
							SDLK_LCTRL, SDLK_RGUI,
							SDLK_END, SDLK_KP_PERIOD,
							SDLK_APPLICATION, SDLK_VOLUMEDOWN,
							SDLK_ALTERASE, SDLK_EXSEL,
							SDLK_KP_00, SDLK_KP_HEXADECIMAL,
							SDLK_MODE, SDLK_SLEEP		
									};

	for(int i = 0; i < num_single; i++)
	{
		key = (int*)malloc(sizeof(int));
		*key = single_keys[i];
		c =  getSettingsKeyName((SDL_Keycode)*key);
		c_len = strlen(c);
		h_add(h, c, c_len, key, sizeof(int));
	}
	for(int i = 0; i < num_paired; i+=2)
	{
		for(SDL_Keycode k = (SDL_Keycode)paired_keys[i]; k <= (SDL_Keycode)paired_keys[i+1]; k++)
		{
			key = (int*)malloc(sizeof(int));
			*key = k;
			c =  getSettingsKeyName((SDL_Keycode)*key);
			c_len = strlen(c);
			h_add(h, c, c_len, key, sizeof(int));
		}
	}
	// adding gamepad button names
	for(int i = 0; i < SETTINGS_GAMEPAD_TOTAL_BUTTON_NUM; i++)
    {
        key = (int*)malloc(sizeof(int));
        *key = i+SETTINGS_GAMEPAD_KEYCODE_START;
        c = SETTINGS_GAMEPAD_BUTTON_NAMES[i];
        c_len = strlen(c);
        h_add(h, c, c_len, key, sizeof(int));
    }
}
