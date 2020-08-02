/* author: Cornelius Marx */
#ifndef CONSOLE_PARAMETER_H
#define CONSOLE_PARAMETER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

// storing and accessing parameters passed via console
struct ConsoleParameters
{
	/* constructor
	 * @param argc argument count
	 * @param argv argument values
	 * @note make sure the argv reference is valid during lifetime of this object
	 */
	ConsoleParameters(int _argc, const char * _argv[]);

	/* get integer value of given parameter
	 * @param name name of the parameter
	 * @param value if name exists the next argument (if exists) in argv is parsed to an integer and written into this variable
	 * @return 1 if name could be found, else 0
	 */
	int getInt(const char * name, int & value)const;

	/* get integer value of argument at given index
	 * @param index index in argv
	 * @param value if index is valid, argv[index] is parsed to an integer and written into this variable
	 * @return 1 if index is valid, else 0
	 */
	int getIntAt(int index, int & value)const;

	/* get string value of given parameter
	 * @param name name of the parameter
	 * @param value if name exists its value is written into this variable
	 * @return 1 if name could be found, else 0
	 */
	int getString(const char * name, const char* & value)const;

	/* get string value of argument at given index
	 * @param index index in argv
	 * @param value if index is valid, argv[index] (pointer) is written into this variable
	 * @return 1 if index is valid, else 0
	 */
	int getStringAt(int index, const char *& value)const;

	/* get float value of given parameter
	 * @param name name of the parameter
	 * @param value if name exists its value is written into this variable
	 * @return 1 if name could be found, else 0
	 */
	int getFloat(const char * name, float & value)const;

	/* get float value of argument at given index
	 * @param index index in argv
	 * @param value if index is valid, argv[index] is parsed to a float and written into this variable
	 * @return 1 if index is valid, else 0
	 */
	int getFloatAt(int index, float & value)const;

	/* get flag set
	 * @param name
	 * @return 1 if name could be found, else 0
	 */
	int getFlag(const char * name)const{return getIndex(name) >= 0;}

	/* get index within the argv array that indicates the first parameter with the given name 
	 * @param name parameter to get index of
	 */
	int getIndex(const char * name)const;

	/* check if a given index is valid in argv
	 * @param index index to check
	 * @return 1 if index is valid, else 0
	 */
	int indexValid(int index)const{return index >= 0 && index < argc;}

	/* argument count */
	int argc;

	/* argument values */
	const char ** argv;
};

#endif

