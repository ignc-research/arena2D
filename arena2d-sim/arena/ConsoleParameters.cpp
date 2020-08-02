/* author: Cornelius Marx */
#include "ConsoleParameters.hpp"

ConsoleParameters::ConsoleParameters(int _argc, const char * _argv[]) : argc(_argc), argv(_argv){}

int ConsoleParameters::getInt(const char * name, int & value)const
{
	int index = getIndex(name);
	return getIntAt(index+1, value);
}

int ConsoleParameters::getIntAt(int index, int & value)const
{
	if(indexValid(index)){
		value = atoi(argv[index]);
		return 1;
	}

	return 0;
}

int ConsoleParameters::getString(const char * name, const char* & value)const
{
	int index = getIndex(name);
	return getStringAt(index+1, value);
}

int ConsoleParameters::getStringAt(int index, const char* & value)const
{
	if(indexValid(index)){
		value = argv[index];
		return 1;
	}

	return 0;
}

int ConsoleParameters::getFloat(const char * name, float& value)const
{
	int index = getIndex(name);
	return getFloatAt(index+1, value);
}

int ConsoleParameters::getFloatAt(int index, float & value)const
{
	if(indexValid(index)){
		value = atof(argv[index]);
		return 1;
	}
	return 0;
}

int ConsoleParameters::getIndex(const char * name)const
{
	for(int i = 0; i < argc; i++)
	{
		if(!strcmp(name, argv[i]))
			return i;
	}
	
	return -2;
}
