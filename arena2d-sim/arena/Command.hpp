/*
 * Command.h
 *
 *  Created on: May 20, 2018
 *      Author: zer0divider
 */

#ifndef COMMAND_H_
#define COMMAND_H_

#include <string>
#include <string.h>
#include <assert.h>
#include <list>
#include <engine/zLogfile.hpp>
#include "CommandStatus.hpp"

using namespace std;

/* helper functions */
namespace CommandTools {

	/* splitting given text into arguments separated by ' ' until ';' is read
		number of tokens is stored in given ptr <num_tokens>
		@tokens contains an array of strings with <num_tokens> elements containing separate tokens
		return: pointer to the next command inside given string (commands are separated by ';')
	*/
	const char * splitCommand(const char * t, int * num_tokens, char *** tokens);
	 
	 /*split command helper */
	char* splitCommand_add(const char * s, int start, int end);

	/* return value of splitCommand can be passed to free the array
		<size> is the number of elements in that array
	*/
	void splitCommand_free(char ** c, int size);
};

/* command argument represents an console argument that is parsed according to its type */

/* name, description and hints of a command */
struct CommandDescription
{
	CommandDescription(){}
	CommandDescription(const string & _name, const string & _hint, const string & _descr):
	name(_name), hint(_hint), descr(_descr){}

	string name;  /* command name */
	string hint;  /* required (and optional) parameters */
	string descr; /* what does the command do? */
};

/* providing functions for creating custom commands that are executed during application runtime
	T: class of which to call the member function
	R: return type of member function
	P: parameters of member function
*/
template<class T, typename R, typename ... P>
class DevCommand
{
public:
	/* pointer to a member function of T */
	typedef R (T::*cmd_p)(P... args);

	/* constructor: create a new command with given attributes
		name: name of the command
		hint: list of the parameters used in this command
		descr: description of what the command does */
	DevCommand(cmd_p command, const string& name, const string& hint = "", const string& descr = ""):
	_descriptor(name, hint, descr){
		_command =  command;
		}
	
	/* destructor */
	virtual ~DevCommand(){}
	
	/* execute the command with given arguments (abstract, needs to be overwritten by child-class)
		argc: argument-count, number of given arguments in <argv>
		argv: argument-values, string array containing <argc> elements 
		return: 0 on success */
	R exec(T * t, P... args){return (t->*_command)(args...);}

	/* getting string members */
	const CommandDescription& getDescription(){return _descriptor;} 
private:
	CommandDescription _descriptor;
	cmd_p _command;
};

#endif /* COMMAND_H_ */
