/* author: Cornelius Marx
 * description: simple csv writer class
 */
#ifndef CSV_WRITER_H
#define CSV_WRITER_H

#include <vector>
#include <stdio.h>

class CSVWriter
{
public:
	/* constructor
	 */
	CSVWriter(char delimiter=',');

	/* destructor
	 */
	~CSVWriter();
	
	/* open file for writing
	 * @param path the path to write file to
	 * @return -1 on error, 0 on success
	 */
	int open(const char * path);

	/* done writing to file
	 */
	void close();

	/* @return true if file is open */
	bool isOpen(){return _file != NULL;}

	/* write header
	 */
	void writeHeader(const std::vector<const char*> & col_names);

	/* write line of values
	 * @param col_values array of size #columns, each value at index i corresponds to the column i
	 */
	void write(const std::vector<float> & values);

	/* flush file stream so data is written immediately
	 */
	void flush();

	/* get number of cols
	 * @return number of cols
	 */ 
	int getNumCols(){return _numCols;}

	/* @return total number of lines written (including header)
	 */
	int getNumLines(){return _numLinesWritten;}
private:
	/* number of columns */
	int _numCols;

	/* csv file for writing*/
	FILE * _file;

	/* csv column delimiter */
	char _delimiter;

	/* keep track of the total number of lines written (including header)*/
	int _numLinesWritten;
};

#endif
