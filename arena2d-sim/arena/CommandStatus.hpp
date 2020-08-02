#ifndef COMMANDSTATUS_H
#define COMMANDSTATUS_H

/* error codes returned by exec() */
enum class CommandStatus{SUCCESS=0, 		/* everything ok */
					 INVALID_ARG, 	/* one or more arguments are incorrect */
					 EXEC_FAIL,		/* execution of the command failed */
					 UNKNOWN_COMMAND	/* command does not exist */
					};
#endif
