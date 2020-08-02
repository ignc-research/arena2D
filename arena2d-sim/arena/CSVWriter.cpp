/* author: Cornelius Marx */

#include "CSVWriter.hpp"

CSVWriter::CSVWriter(char delimiter) : _file(NULL), _delimiter(delimiter)
{
}


CSVWriter::~CSVWriter()
{
	close();
}

void CSVWriter::close()
{
	if(_file != NULL){
		fclose(_file);
		_file = NULL;
		_numCols = 0;
	}
}

void CSVWriter::writeHeader(const std::vector<const char*> & col_names)
{
	for(unsigned int i = 0; i < col_names.size(); i++)
	{
		fputs(col_names[i], _file);

		if(i < col_names.size()-1)// not last column
			fputc(_delimiter, _file);
		else
			fputc('\n', _file);
	}
	_numCols = col_names.size();
	_numLinesWritten++;
}

int CSVWriter::open(const char * path)
{
	// open file
	close();
	_file = fopen(path, "w");
	if(_file == NULL){
		return -1;
	}
	_numLinesWritten = 0;
	_numCols = 0;

	return 0;
}

void CSVWriter::write(const std::vector<float> & values)
{
	// nothing opened
	if(_file == NULL)
		return;

	for(unsigned int i = 0; i < values.size(); i++)
	{
		if(i < values.size()-1)// not last column
			fprintf(_file, "%f,", values[i]);
		else
			fprintf(_file, "%f\n", values[i]);
	}

	_numLinesWritten++;
}

void CSVWriter::flush()
{
	if(_file != NULL)
		fflush(_file);
}
