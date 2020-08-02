#ifndef CONTEXT_H
#define CONTEXT_H

/* generic context serving as current state of the application (e.g. menu, lobby, game)
*/

#include "zer0/zFramework.hpp"
#include "ApplicationMode.hpp"
#include "Command.hpp"

class Context
{
public:
	/* context types */
	enum Type{NONE=0, EXIT, ARENA, NUM_TYPES};

	/* constructor/destructor called every time a context is switched */
	Context(Type t, ApplicationMode m) {_type = t; _isVisual = (m == ApplicationMode::VISUAL); _nextContext = NONE;}
	virtual ~Context(){}

	/* init
		is called by application after required resources have been loaded */
	virtual void init(){}

	/* render stuff to screen */
	virtual void render(){}

	/* 	normal input-event updater
		return: what context should be switched to (see Context::Type, NONE=0 for continue, EXIT=-1 for quitting application) */
	virtual Type update(zEventList & evtList){return NONE;}

	/* parallel updater (runs in separate thread) */
	virtual void t_update(SocketEventList & evtList){}

	/* get current resource list
		called by app after calling constructor of context and before init() */
	const ResourceList& getResourceList(){return _resourceList;}

	/* return context name */
	virtual const char* getContextName() = 0;

	/* execute command at runtime returns 0 on success*/
	CommandStatus command(const char * c);
	virtual CommandStatus command(const char * name, int argc, const char * argv[]) = 0;/* execute command */
	virtual const CommandDescription* getCommand(const char * name) = 0; /* get command description */
	virtual void getAllCommands(list<const CommandDescription*> & cmds) = 0; /* get all command descriptions */

	/* get this context type */
	Type getType(){return _type;}
protected:	
	/* adding resources from array to resource list */
	void addResources(const RESOURCE_ID * res, int num){for(int i=0; i < num; i++)_resourceList.push_back(res[i]);}

	bool _isVisual; /* current runtime environment of application can handle rendering stuff */
	ResourceList _resourceList;/* resource list contains all resources that are needed by this context */
	Type _type;/* this context type */
	Type _nextContext;
};
#endif
