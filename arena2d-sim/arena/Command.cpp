#include "Command.hpp"

void CommandTools::splitCommand_free(char ** c, int size)
{
	for(int i = 0; i < size; i++)
		delete[](c[i]);
	
	delete[] c;
}

/* helper function to add a token */
char* CommandTools::splitCommand_add(const char * s, int start, int end)
{
	int size = end - start + 1;
	if(size < 0)
		return NULL;
	char * t = new char[size+1];
	memcpy(t, s+start, size);
	t[size] = '\0';
	return t;
}

const char * CommandTools::splitCommand(const char * s, int * num_tokens, char *** tokens)
{
	list<char*> token_list;
	bool quote_marks = false; /* quotation marks have been found */
	bool last_white = true; /* last character was separation */
	int start = 0;
	int i = 0;
	/* skip all whitespaces at the beginning */
	while(s[i] != '\0' && s[i] != ';')
	{
		if(!quote_marks)
		{
			if(s[i] == ' ')
			{
				if(last_white)
				{
				}
				else
				{
					token_list.push_back(splitCommand_add(s, start, i-1));
				}
				start = i+1;
				last_white = true;
			}
			else if(s[i] == '\"')
			{
				if(!last_white){
					token_list.push_back(splitCommand_add(s, start, i-1));
				}
				last_white = true;
				quote_marks = true;
				start = i+1;
			}
			else
			{
				last_white = false;
			}
		}
		else/* inside a string literal */
		{
			if(s[i] == '\"')
			{
				quote_marks = false;
				token_list.push_back(splitCommand_add(s, start, i-1));
				start = i+1;
				last_white = true;
			}
			else
				last_white = false;
		}
		i++;
	}
	/* add last token */
	if(i != start)
	{
		char * t = splitCommand_add(s, start, i-1);
		token_list.push_back(t);
	}
	if(s[i] == ';')
		i++;

	/* create new array of strings - add all element from list into that array */
	char ** token_array = new char*[token_list.size()];
	list<char*>::iterator it = token_list.begin();
	int count = 0;
	while(it != token_list.end())
	{
		token_array[count] = *it;
		it++;
		count++;
	}
	*num_tokens = token_list.size();

	*tokens = token_array;
	return s+i;
}
