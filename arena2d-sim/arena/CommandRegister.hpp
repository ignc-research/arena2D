#ifndef COMMAND_REGISTER_H
#define COMMAND_REGISTER_H

#include "Command.hpp"
#include <engine/zLogfile.hpp>
#include <list>
#include <unordered_map>

/* controlling many commands belonging to one context
	T: class of which to call the member function
	R: return type of member function
	P: parameters of member function
*/
template<class T, typename R, typename ... P>
class CommandRegister
{
public:
	/* useful type definitions */
	typedef DevCommand<T, R, P...> CommandType;
	typedef unordered_map<string, CommandType*> HashType;
	typedef typename HashType::value_type ValuePair;

	/* constructor */
	CommandRegister():_lastCommand(NULL){}

	/* destructor */
	~CommandRegister();


	/* register a new command */
	void registerCommand(typename DevCommand<T, R, P...>::cmd_p func, const string& name,
													const string& hint,
													const string& descr);

	/* execute a command by given name and parameters */
	R execCommand(T * t, const string & name, P...); 
	
	/* get command that corresponds to the given name
		return NULL if command was not found */
	DevCommand<T, R, P...>* getCommand(const string & name);

	/* get last error that occured */
	const char* getError();

	/* getting all command descriptions that are currently in the register */
	void getAllCommandDescriptions(list<const CommandDescription*> & cmd_list);
private:
	DevCommand<T, R, P...> * _lastCommand; /* last command that was executed */
	HashType _hashTable;

};


template<class T, typename R, typename ... P>
CommandRegister<T, R, P...>::~CommandRegister()
{
	/* remove all instanciated dev commands */
	typename HashType::iterator it = _hashTable.begin();	
	while(it != _hashTable.end())
	{
		delete(it->second);
		it++;
	}
}

template<class T, typename R, typename ... P>
void CommandRegister<T, R, P...>::registerCommand(typename DevCommand<T, R, P...>::cmd_p func, const string& name,
																 const string& hint,
																 const string& descr)
{
	pair<typename HashType::iterator, bool> res = _hashTable.insert(ValuePair(name, new DevCommand<T, R, P...>(func, name, hint, descr)));
	if(!res.second)/* command already exists */
	{
		WARNING_F("Command has already been added: %s", name.c_str());
	}
}


template<class T, typename R, typename ... P>
R CommandRegister<T, R, P...>::execCommand(T * t, const string & name, P... args)
{
	/* trying to find command */
	DevCommand<T, R, P...> * cmd = getCommand(name);
	if(cmd == NULL)/* command not found */
		return CommandStatus::UNKNOWN_COMMAND;

	/* command has been found -> executing */
	_lastCommand = cmd;
	return cmd->exec(t, args...);
}

template<class T, typename R, typename ... P>
DevCommand<T, R, P...>* CommandRegister<T, R, P...>::getCommand(const string & name)
{
	/* find command by name */
	typename HashType::iterator it = _hashTable.find(name);

	if(it != _hashTable.end())/* command exists */
		return it->second;
	return NULL; /* command does not exist */
}

template<class T, typename R, typename ... P>
void CommandRegister<T, R, P...>::getAllCommandDescriptions(list<const CommandDescription*> & cmd_list)
{
	/* going through every bucket and fetching each command */
	int i = 0;
	while(i < (int)_hashTable.bucket_count())
	{
		/* iterator over every item in that bucket */
		typename HashType::local_iterator it = _hashTable.begin(i);
		while(it != _hashTable.end(i))
		{
			DevCommand<T, R, P...> *d = it->second;
			const CommandDescription& descr = d->getDescription();
			cmd_list.push_back(&descr);
			it++;
		}
		i++;
	}
}

#endif
