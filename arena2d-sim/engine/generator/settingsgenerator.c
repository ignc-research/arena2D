#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "../list.h"

#define WRITE_COMMENT_PREFIX "   #"
#define BUFFER_SIZE 256

int isSpecialChar(char c){return (c == ',' || c == '{' || c == '}' || c == '[' || c == ']' || c == ';' || c == '/' || c == '*');}

int isWhiteSpace(char c){return (c == ' ' || c == '\t' || c == '\n' || c == '\r');}

int isEndOfLine(char c){return (c == '\n' || c == '\r');}

void printIndents(int num){for(int i = 0; i < num; i++)printf("\t");}
void fprintIndents(FILE * f, int num){for(int i = 0; i < num; i++)fprintf(f, "\\t");}

//returns 1 on success else 0
int toNumber(char * word, int word_len, int * number)
{
	int _num = 0;
	for(int i = word_len-1; i >= 0; i--)
	{
		_num *= 10;
		if(word[i] < '0' || word[i] > '9')
			return 0;
		_num += (int)(word[i] - '0');
	}
	if(number != NULL)
		*number = _num;

	return 1;
}


//reading next word/token
//returns 0 if eof reached
int readNextWord(char * buffer, int buffer_size);

// reading till new line
// returns 0 if eof reached
int readNextLine(char * buffer, int buffer_size);

typedef struct{
	int type;
	char * name;
	sList * data;//points to a list of variables if this is a container
}dataType;

typedef struct{
	dataType * type;
	char * name;
	int count;//supporting arrays
	char * comment;
}variable;
void printSymbols(variable * t, int indents);

void writeSymbolList(sList * l, variable * v);
void writeSymbolListRec(sList * l, variable * v, char * buffer, int len);

void writeSaveFunction(variable * v);
void writeSaveFunctionRec(variable * v, int indents, char * buffer, int len);

FILE * F;
char lastCharRead = ' ';
int currentLine = 0; //the file is currently at;
sList * structList = 0;

enum ReadMode{
	OUTER,
	OUTER_TYPEDEF,
	OUTER_STRUCT,
	OUTER_STRUCT_BEGIN,
	OUTER_STRUCT_END,
	OUTER_STRUCT_DATA_TYPE,
	OUTER_STRUCT_DATA_ARRAY,
	OUTER_STRUCT_DATA_END
};

enum DataTypes{
	DATA_INT,
	DATA_FLOAT,
	DATA_STRING,
	DATA_KEY,
	DATA_CONTAINER
};

char * DATA_TYPE_NAMES[4] = {"int", "float", "string", "SDL_Keycode"};

int main(int argc, char ** argv)
{
	//checking arguments
	if(argc != 3)
	{
		printf("usage: %s <input> <output>\n", argv[0]);
		return 1;
	}

	//opening file for reading
	F = fopen(argv[1], "r");
	if(F == NULL)//error occured?
	{
		fprintf(stderr, "Unable to open file %s: %s\n", argv[1], strerror(errno));
		return 1;
	}
	//reading word by word, parsing c-structs
	char word[BUFFER_SIZE];
	int mode = OUTER;
	int comment_count = 0;
	int array_count = 0;
	dataType * last_data_type = NULL;//last dataType that has been found for variable
	variable * last_variable = NULL;
	structList = list_init();//struct list containing all data types
	//adding default data types
	for(int i = 0; i < 4; i++)
	{
		dataType * defaultType = (dataType*)malloc(sizeof(dataType));
		defaultType->type = DATA_INT + i;
		defaultType->name = malloc(strlen(DATA_TYPE_NAMES[i])+1);
		strcpy(defaultType->name, DATA_TYPE_NAMES[i]);
		defaultType->data = NULL;
		list_pushBack(structList, defaultType);
	}
	while(readNextWord(word, BUFFER_SIZE))
	{
		int word_len = strlen(word);
		if(word_len <= 0)
			continue;

		//check for comments
		int skip = 0;
		if(comment_count == 0)
		{
			if(strcmp(word, "/") == 0)
			{
				comment_count++;
				skip = 1;
			}
		}
		else if(comment_count == 1)
		{
			if(strcmp(word, "/") == 0)
			{
				// read whole comment into buffer
				readNextLine(word, BUFFER_SIZE);
				word_len = strlen(word);
				if(last_variable != NULL)
				{
					last_variable->comment = malloc(word_len+1);
					strcpy(last_variable->comment, word);
				}
				skip = 1;
				comment_count = 0;
			}
			else//error
			{
				return 1;
			}
		}

		if(skip == 0)
		{
			//check for keywords
			switch(mode)
			{

				case OUTER:
				{
					//printf("OUTER\n");
					if(strcmp(word, "typedef") == 0)
					{
						mode = OUTER_TYPEDEF;
					}
					else
					{
						fprintf(stderr, "expected 'typedef' on line %d!\n", currentLine+1);
						return 1;
					}
				}break;
				case OUTER_TYPEDEF:
				{
					//printf("OUTER_TYPEDEF\n");
					if(strcmp(word, "struct") == 0)
					{
						mode = OUTER_STRUCT;
					}
					else
					{
						fprintf(stderr, "expected 'struct' on line %d!\n", currentLine+1);
						return 1;
					}
				}break;
				case OUTER_STRUCT:
				{
					//printf("OUTER_STRUCT\n");
					if(strcmp(word, "{") == 0)
					{
						mode = OUTER_STRUCT_BEGIN;
						dataType * t = (dataType*)malloc(sizeof(dataType));
						t->data = list_init();
						t->name = NULL;
						t->type = DATA_CONTAINER;
						list_pushBack(structList, t);
					}
					else
					{
						fprintf(stderr, "expected '{' on line %d!\n", currentLine+1);
						return 1;
					}
				}break;
				case OUTER_STRUCT_BEGIN:
				{
					//printf("OUTER_STRUCT_BEGIN\n");
					//parsing data-type

					if(strcmp(word, "}") == 0)
					{
						mode = OUTER_STRUCT_END;
					}
					else
					{
						//try finding other struct
						int found = 0;
						for(sListItem * it = structList->first; it != NULL; it = it->next)
						{
							char * name = ((dataType*)it->data)->name;
							if(name != NULL && strcmp(word, name) == 0)
							{
								found = 1;
								last_data_type = (dataType*)it->data;
								mode = OUTER_STRUCT_DATA_TYPE;
								break;
							}
						}
						if(!found)
						{
							fprintf(stderr, "expected data type on line %d (got '%s')!\n", currentLine+1, word);
							return 1;
						}
					}

				}break;
				case OUTER_STRUCT_DATA_TYPE:
				{
					//printf("OUTER_STRUCT_DATA_TYPE\n");
					if(last_data_type != NULL)
					{
						//printf("found data: %s %s\n", ((dataType*)last_container->data)->name, word);
						//adding variable to current container list
						dataType * d = (dataType*)structList->last->data;
						variable * var = (variable*)malloc(sizeof(variable));
						var->type = last_data_type;
						var->name = (char*)malloc(word_len+1);
						var->comment = NULL;
						var->count = 1;
						strcpy(var->name, word);
						last_variable = var;
						list_pushBack(d->data, var);
						mode = OUTER_STRUCT_DATA_END;
					}
					else
					{
						fprintf(stderr, "no data type was given on line %d!\n", currentLine+1);
						return 1;
					}
				}break;
				case OUTER_STRUCT_DATA_END:
				{
					//printf("OUTER_STRUCT_DATA_END\n");
					if(array_count == 0 && strcmp(word, "[") == 0)
					{
						mode = OUTER_STRUCT_DATA_ARRAY;
						array_count = 1;
					}
					else if(strcmp(word, ",") == 0)
					{
						mode = OUTER_STRUCT_DATA_TYPE;
					}
					else if(strcmp(word, ";") == 0)
					{
						mode = OUTER_STRUCT_BEGIN;
						last_data_type = NULL;
					}
					else
					{
						fprintf(stderr, "expected ';' on line %d!\n", currentLine+1);
						return 1;
					}
				}break;
				case OUTER_STRUCT_DATA_ARRAY:
				{
					//printf("OUTER_STRUCT_DATA_ARRAY\n");
					int num = 0;
					if(strcmp(word, "]") == 0)
					{
						mode = OUTER_STRUCT_DATA_END;
						array_count = 0;
					}
					else if(toNumber(word, word_len, &num))
					{
						mode = OUTER_STRUCT_DATA_ARRAY;
						variable * v = ((variable*)((dataType*)structList->last->data)->data->last->data);
						v->count = num;
						//printf("setting count to %d\n", v->count);
					}
					else
					{
						fprintf(stderr, "expected array on line %d!\n", currentLine+1);
						return 1;
					}
				}break;
				case OUTER_STRUCT_END:
				{
					//printf("OUTER_STRUCT_END\n");
					if(strcmp(word, ";") == 0)
						mode = OUTER;
					else
					{
						if(structList->last == NULL)
						{
							fprintf(stderr, "no data in list on line %d\n", currentLine+1);
							return 1;
						}
						else
						{
							dataType * t = (dataType*)structList->last->data;
							//adding name of data-type
							t->name = (char*)malloc(word_len+1);
							strcpy(t->name, word);
							//printf("adding data type: %s\n", word);
						}
					}

				}break;
			}
		}
	}

	//closing file
	fclose(F);

	//open file for writing
	F = fopen(argv[2], "w");
	if(F == NULL)//error occured?
	{
		fprintf(stderr, "Unable to open file %s: %s\n", argv[1], strerror(errno));
		return 1;
	}
	//print resulting symbols
	variable global_var;
	global_var.name = "global_settings";
	global_var.type = (dataType*)structList->last->data;
	global_var.count = 1;
//	printSymbols(&global_var, 0);
		
	//writing to file
	fprintf(F,	"//THIS FILE IS AUTO-GENERATED! DO NOT EDIT!\n\n" 
				"#include <engine/GlobalSettings.hpp>\n\n"
				"void GlobalSettings::initSymbolTable()\n"
				"{\n"
				"\t_hashTable = h_init(97, GlobalSettings::stringHash, NULL);\n"
				"\tsSettingsOption * option = NULL;\n\n"
				);

	sList * symbolList = list_init();
	writeSymbolList(symbolList, &global_var);
	fprintf(F, "}\n\n");
	
	fprintf(F, "void GlobalSettings::writeToFile(FILE * f)\n{\n\tchar float_buffer[64];\n\n");
	writeSaveFunction(&global_var);
	fprintf(F, "}\n");


	list_freeAll(symbolList);

	//closing file
	fclose(F);

	//freeing lists
	for(sListItem * it = structList->first; it != NULL; it = it->next)
	{
		dataType * d = (dataType*)it->data;
		free(d->name);
		if(d->data != NULL)//removing variables if exist
		{
			for(sListItem * itVar = d->data->first; itVar != NULL; itVar = itVar->next)
			{
				variable * v = (variable*)itVar->data;
				free(v->name);
				free(v->comment);
				free(v);
			}
			list_free(d->data);
		}
		free(d);
	}
	list_free(structList);

	//done
	return 0;
}

void writeSaveFunction(variable * v)
{
	char b[512];
	for(sListItem * it = v->type->data->first; it != NULL; it = it->next)
	{
		writeSaveFunctionRec((variable*)it->data, 0, b, 0);
	}
}

void writeSaveFunctionRec(variable * v, int indents, char * buffer, int len)
{
	int sym_len = strlen(v->name);
	if(v->type->type == DATA_CONTAINER)
	{
		for(int i = 0; i < v->count; i++)
		{
			int new_len = len;
			fprintf(F, "\tfprintf(f, \"");
			fprintIndents(F, indents);
			if(v->count <= 1)
			{
				fprintf(F, "%s{", v->name);
				sprintf(buffer+len, "%s.", v->name);
				new_len += sym_len+1;
			}
			else
			{
				fprintf(F, "%s[%d]{", v->name, i);
				sprintf(buffer+len, "%s[%d].", v->name, i);
				while(buffer[new_len] != '\0')new_len++;
			}
			if(v->comment != NULL){
				fprintf(F, WRITE_COMMENT_PREFIX"%s", v->comment);
			}
			fprintf(F, "\\n\");\n");

			for(sListItem * it = v->type->data->first; it != NULL; it = it->next)
			{
				writeSaveFunctionRec((variable*)it->data, indents+1, buffer, new_len); 
			}
		fprintf(F, "\tfprintf(f, \"");
		fprintIndents(F, indents);
		fprintf(F, "}\\n");
		if(indents == 0)
			fprintf(F, "\\n");

		fprintf(F, "\");\n");
		}
	}
	else
	{
		sprintf(buffer+len, "%s", v->name);
		if(v->type->type == DATA_FLOAT){// special parsing for floating points
			fprintf(F, "\tzStringTools::fromFloat(_settings.%s, float_buffer);\n", buffer);
		}
		fprintf(F, "\tfprintf(f, \"");
		fprintIndents(F, indents);
		fprintf(F, "%s = ", v->name);
		switch(v->type->type)
		{
			case DATA_INT:
			{
				fprintf(F, "%%d");
			}break;
			case DATA_FLOAT:
			{
				fprintf(F, "%%s");
			}break;
			case DATA_KEY:
			{
				//key parsing
				fprintf(F, "%%s");
			}break;
			case DATA_STRING:
			{
				fprintf(F, "\\\"%%s\\\"");
			}break;
		}
		if(v->comment != NULL){
			fprintf(F, WRITE_COMMENT_PREFIX"%s", v->comment);
		}
		if(v->type->type == DATA_KEY)
		{
			fprintf(F, "\\n\", getSettingsKeyName(_settings.%s));\n", buffer);
		}
		else if(v->type->type == DATA_FLOAT)
		{
			fprintf(F, "\\n\", float_buffer);\n");
		}
		else if(v->type->type == DATA_STRING)
		{
			fprintf(F, "\\n\", _settings.%s.c_str());\n", buffer);
		}
		else
		{
			fprintf(F, "\\n\", _settings.%s);\n", buffer);
		}
	}
}

void writeSymbolList(sList * l, variable * v)
{
	char b[512] ;
	for(sListItem * it = v->type->data->first; it != NULL; it = it->next)
	{
		writeSymbolListRec(l, (variable*)it->data, b, 0); 
	}

}

void writeSymbolListRec(sList * l, variable * v, char * buffer, int len)
{
	int sym_len = strlen(v->name);
	if(v->type->type == DATA_CONTAINER)
	{
		for(int i = 0; i < v->count; i++)
		{
			int new_len = len;
			if(v->count > 1)
			{
				sprintf(buffer+len, "%s[%d].", v->name, i);
				while(buffer[new_len] != '\0')new_len++;
			}
			else
			{
				sprintf(buffer+len, "%s.", v->name);
				new_len += sym_len+1;
			}

			for(sListItem * it = v->type->data->first; it != NULL; it = it->next)
			{
				writeSymbolListRec(l, (variable*)it->data, buffer, new_len); 
			}
		}
	}
	else
	{
		sprintf(buffer+len, "%s", v->name);
		int new_len = len+sym_len;
		char * b = (char*)malloc(len+sym_len + 1);
		strcpy(b, buffer);
		list_pushBack(l, b);
		fprintf(F, "\t//%s\n", buffer);
		fprintf(F, "\toption = (sSettingsOption*)malloc(sizeof(sSettingsOption));\n");
		switch(v->type->type)
		{
			case DATA_INT:
			{
				fprintf(F, "\toption->type = SETTINGS_TYPE_INT;\n");
			}break;
			case DATA_FLOAT:
			{
				fprintf(F, "\toption->type = SETTINGS_TYPE_FLOAT;\n");
			}break;
			case DATA_STRING:
			{
				fprintf(F, "\toption->type = SETTINGS_TYPE_STRING;\n");
			}break;
			case DATA_KEY:
			{
				fprintf(F, "\toption->type = SETTINGS_TYPE_KEY;\n");
			}break;
		}
		fprintf(F, "\toption->data = &(_settings.%s);\n", buffer);
		fprintf(F, "\th_add(_hashTable, \"%s\", %d, option, sizeof(sSettingsOption));\n\n", buffer, new_len+1);
	}
}

//recursive printing
void printSymbols(variable * v, int indents)
{
	if(v->type->type == DATA_CONTAINER)
	{
		printIndents(indents);
		printf("%s", v->name);
		if(v->count > 1)//is array?
			printf("[%d]", v->count);

		printf("\n");
		printIndents(indents);
		printf("{\n");
		for(sListItem * it = v->type->data->first; it != NULL; it = it->next)
			printSymbols((variable*)it->data, indents+1);
		printIndents(indents);
		printf("}\n");
	}
	else
	{
		printIndents(indents);
		printf("%s %s\n", v->type->name, v->name);
	}
}

int readNextWord(char * buffer, int buffer_size)
{
	if(fread(&lastCharRead, 1,1, F) == 0)// eof
		return 0;
	//last character read was a special char
	if(isSpecialChar(lastCharRead))
	{
		if(buffer_size >= 2)
		{
			buffer[0] = lastCharRead;
			buffer[1] = '\0';
			return 1;
		}
	}

	//read as long as there are white characters
	while(isWhiteSpace(lastCharRead))
	{
		if(lastCharRead == '\n')//next line found
			currentLine++;

		if(fread(&lastCharRead, 1,1, F) != 1)//eof or error
		{
			if(buffer_size > 0)//set buffer to empty
				buffer[0] = '\0';
			lastCharRead = '\0';
			return 0;
		}
	}
	
	//check for special character again
	if(isSpecialChar(lastCharRead))
	{
		if(buffer_size >= 2)
		{
			buffer[0] = lastCharRead;
			buffer[1] = '\0';
			return 1;
		}
	}
	//write into buffer
	int i = 0;
	do{
		if(i < buffer_size)//buffer still has capacity
		{
			buffer[i] = lastCharRead;
			i++;
		}
		if(fread(&lastCharRead, 1,1, F) != 1)//eof or error
			break;
	}
	while(!(isWhiteSpace(lastCharRead) || isSpecialChar(lastCharRead)));

	// rewind to last char to be read in next call
	fseek(F, -1, SEEK_CUR);

	//set 0-terminator
	if(i >= buffer_size)
		i = buffer_size-1;

	buffer[i] = '\0';
	return 1;
}

int readNextLine(char * buffer, int buffer_size)
{
	char c;
	int count = 0;
	int res;
	while((res = fread(&c, 1, 1, F)) && !isEndOfLine(c)){
		if(count < buffer_size){
			buffer[count] = c;
			count++;
		}
	}
	if(count >= buffer_size)
		count--;
	buffer[count] = '\0';


	return res;
}
