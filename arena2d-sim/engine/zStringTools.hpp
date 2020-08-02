//Created on: 21st Jan 2017
//Author: Cornelius Marx

//describes functions to modify strings

#ifndef ZER0_STRING_TOOLS_H
#define ZER0_STRING_TOOLS_H

#include <string>
#include "zLogfile.hpp"

namespace zStringTools
{

	//loads a text-file (filename) and puts its content into the given string (text)
	//returns 1 on success; -1 on failure (could'nt open filename)
	int loadFromFile(const char * filename, std::string * text);

	//stores a string in a file
	//returns 1 on success; -1 on failure (could'nt open filename)
	int storeToFile(const char * filename, const char * text);
    
    // convert integer to (0-terminated) string-sequence, returns number of characters written to buffer (without 0-terminator)
    int fromInt(int a, char * str);
    
    //convert string literal to integer
    //@return number parsed
    //@error is set to 1 if c doesnt represent a valid number, can be set to NULL (if no error handling needed)
    int toInt(const char * c, int * error);

    //convert string literal to float
    //@return number parsed
    //@error is set to 1 if c doesnt represent a valid number
    float toFloat(const char * c, int * error);

	// convert float to null-terminated string literal
	// trailing zeros are cut off
	// @returns number of characters written into buffer, without the null terminator
	// buffer's size should be at least 16
	int fromFloat(float value, char * buffer);

    //is given character a white or tab
    int isWhiteSpace(char c);

    //skipping white space characters
    int skipWhiteSpace(const char ** c);

    //like f_string_skipWhiteSpace, but also skips endline-characters
    //returns number of lines skipped
    int skipWhiteLine(const char ** c);

    //returns 1 if c is one of the characters in str
    int charIsElementOf(char c, const char *str, int str_len);

    //returns 1 if the given char has an alphanumeric value (a-z, A-Z, 0-9)
    int isAlphanum(char c);

	//returns 1 if c is 0-9
	int isNumber(char c);

	// return 1 if c is a-z or A-Z
	int isLetter(char c);

	// move text until pointing to c
	// returns c if found or \0 if not
	char goTo(char c, const char ** text);

	// convert every character to lower case
	void toLower(std::string & s);

	// checks whether a starts with b or b starts with a
	int startsWith(const char * a, const char * b);

	// creating a c string by allocating dynamic memory
	// memory can be freed by calling free() (in stdlib.h)
	char* createCString(const std::string & s);
	char* createCString(const char * s);
};

#endif
