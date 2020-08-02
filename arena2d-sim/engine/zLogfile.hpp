//Created on: 16th Jan 2017
//Author: Cornelius Marx

#ifndef ZER0_LOGFILE_H
#define ZER0_LOGFILE_H

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "zSingleton.hpp"

#define Z_LOG zLogfile::get()

// makro for logging something (info, error, warning)
#define INFO_F(X, ...) zLogfile::get()->printfMode(zLogfile::LOG_INFO, true, X, __VA_ARGS__)
#define WARNING_F(X, ...) zLogfile::get()->printfMode(zLogfile::LOG_WARNING, true, X, __VA_ARGS__)
#define ERROR_F(X, ...) zLogfile::get()->printfMode(zLogfile::LOG_ERROR, true, X, __VA_ARGS__)

#define INFO(X) zLogfile::get()->printfMode(zLogfile::LOG_INFO, true, X)
#define WARNING(X) zLogfile::get()->printfMode(zLogfile::LOG_WARNING, true, X)
#define ERROR(X) zLogfile::get()->printfMode(zLogfile::LOG_ERROR, true, X)

#define Z_LOG_HEAD_CHARACTER '#'
#define Z_LOG_DEFAULT_FILE_NAME "log.txt"


class zLogfile : public zTSingleton<zLogfile>
{
public:
	// logging modes
	enum Mode{LOG_INFO, LOG_WARNING, LOG_ERROR};

	//constructor
	zLogfile();

	//destructor
	~zLogfile();
	
	//initializer
	// @logfile_path where to store the logfile (NULL for no output file)
	void createLog(bool std_print, const char * logfile_path);

	//write-out functions

	// close Log
	void closeLog();

	// printf with given mode
    #ifndef WIN32
        __attribute__((format(printf, 4, 5)))
    #endif
    void printfMode(enum Mode mode, bool new_line, const char * fmt, ...);
	void printfMode(enum Mode mode, bool new_line, const char * fmt, va_list args);

	// printing to all streams that were registerd (std_print/file_print)
	void printFormatted(const char * fmt, va_list args);
	#ifndef WIN32
        __attribute__((format(printf, 2, 3)))
    #endif
	void printFormatted(const char * fmt, ...);

	// writing out given number of linebreaks to all registered streams
	void writeLines(int num);

private:
	FILE * _logfile;
	bool _std_print; //shall the output be written to console? (std)
};

#endif
