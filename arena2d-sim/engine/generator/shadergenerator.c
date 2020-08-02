#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#define BUFFER_SIZE 10000
#define BUFFER_FILE_SIZE 4000
const char * FRAG_START = "static const char * FRAGMENT_SHADER_SOURCE = ";
const char * VERT_START = "static const char * VERTEX_SHADER_SOURCE = ";

int saveFile(const char * path, const char * text, int len);//saving file to @path with content in @text
int loadFile(const char * path, char * text);//loading file at @path and putting content in @text
int checkIncludeDirective(const char * line, int line_size, int num_line, const char * relative_path, char * include_path);

char* getRelativePath(const char * path);

int main(int argc, char ** args)
{	
	char path[256] = "./shaderSource.h";

	if(argc == 4 || argc == 3)//argument is given
	{
		if(argc == 4)
		{
			strcpy(path, args[3]);
		}

		char frag_file[BUFFER_FILE_SIZE];
		char vert_file[BUFFER_FILE_SIZE];
		int frag_len = 0;
		int vert_len = 0;
		vert_len = loadFile(args[1], vert_file);
		if(vert_len < 0)
			return 1;

		frag_len = loadFile(args[2], frag_file);
		if(frag_len < 0)
			return 1;

		char out[BUFFER_SIZE*2];
		int frag_start_len = strlen(FRAG_START);
		int vert_start_len = strlen(VERT_START);
		sprintf(out, "%s\n%s;\n\n%s\n%s;",VERT_START, vert_file, FRAG_START, frag_file);
		if(saveFile(path, out,6+ frag_len + vert_len + frag_start_len + vert_start_len) < 0)
			return 1;
	}
	else
	{
		printf("Usage: %s <vertex_shader_file> <fragment_shader_file> [output_file]\n\n", *args);
        return 1;
	}
	
	return 0;
}

int loadFile(const char * path, char * text)
{
	FILE * f = fopen(path, "r");
	if(f == NULL)
	{
		fprintf(stderr, "shadergenerator: Could not open file %s: %s\n", path, strerror(errno));
		return -1;
	}
	//reading till the end of file
	int count = 0;
	int last_linebreak = 1;
	int last_linebreak_index = 1;
	int line_count = 0;
	char c;
	char last_c = '\0';
	char include_path[256];
    char * relative_path = getRelativePath(path);
	while(fread(&c, 1, 1, f))
	{		
		if(last_linebreak)
		{
			sprintf(text+count, "\"");
			count++;
			last_linebreak = 0;
			last_linebreak_index = count;
		}
		if(c == '\n' || c == '\r' )
		{
			if(!(c == '\n' && last_c == '\r'))//already did a linebreak
			{
				last_linebreak = 1;
				line_count++;
				//check if #include directive has been found and load corresponding file
				int res = checkIncludeDirective(&text[last_linebreak_index], count-last_linebreak_index, line_count, relative_path, include_path);
				if(res < 0)
				{
					return -1;
				}
				if(res)	//include directive found
				{
					res = loadFile(include_path, &text[last_linebreak_index-1]);
					if(res < 0)
					{
						fprintf(stderr, "shadergenerator: Include directive in line %d failed: #include \"%s\"\n", line_count, include_path);
						return -1;
					}
					count = last_linebreak_index-1 + res-1; 
					sprintf(text+count, "\n");
					count +=1;
				}
				else
				{
					sprintf(text+count, "\\n\"\n");
					count+=4; 
				}
			}
		}
		else
		{
			text[count] = c;
			count++;
		}
		last_c = c;
	}
	text[count] = '\0';
	free(relative_path);
	fclose(f);
	return count;
}

int saveFile(const char * path, const char * text, int size)
{
	FILE * f = fopen(path, "w");
	if(f == NULL)
	{
		fprintf(stderr, "shadergenerator: Could not open file %s: %s\n", path, strerror(errno));
		return -1;
	}
	//writing out
	fwrite(text, size, 1, f);
	
	fclose(f);
	return 1;
}
 
int checkIncludeDirective(const char * line, int line_size, int num_line, const char * relative_path, char * include_path)
{
	int count = 0;
	const char * directive = "#include";
	int directive_len = strlen(directive);
	if(line_size < directive_len)
		return 0;
	while(line[count] == ' ' || line[count] == '\t')//removing pre white characters
	{
		count++;
		if(count >= line_size)
			return 0;
	}
	
	for(int i = 0; i < directive_len; i++)
	{
		if(count >= line_size)//#include not found
			return 0;
		if(line[count] != directive[i])
			return 0;

		count++;
	}

	while(line[count] == ' ' || line[count] == '\t')//removing pre white characters
	{
		count++;
		if(count >= line_size)
		{
			fprintf(stderr, "shadergenerator: line %d: '%.*s' missing path!\n", num_line, line_size, line);
			return -1;
		}
	}

	if(line[count] != '"')
	{
		fprintf(stderr, "shadergenerator: line %d: '%.*s' expected '\"'!\n", num_line, line_size, line);
		return -1;
	}
	count++;
	if(count >= line_size)
	{
		fprintf(stderr, "shadergenerator: line %d: '%.*s' expected string after '\"'!\n", num_line, line_size, line);
		return -1;
	}

	int rel_path_len = strlen(relative_path);
	int include_count = rel_path_len;
	while(line[count] != '"')
	{
		include_path[include_count] = line[count];
		include_count++;
		count++;
		if(count >= line_size)
		{
			fprintf(stderr, "shadergenerator: line %d: '%.*s' expected '\"'!\n", num_line, line_size, line);
			return -1;
		}
	}
	for(int i = 0; i < rel_path_len; i++)
    {
        include_path[i] = relative_path[i];
    }

	include_path[include_count] = '\0';
	return 1;
}

char* getRelativePath(const char * path)
{
    int path_len = strlen(path);
    char * rel_path = malloc(path_len+1);
    int i = path_len-1;
    for(; i >= 0; i--){
        if(path[i] == '/')// finding last front slash
            break;
    }
    strcpy(rel_path, path);
    rel_path[i+1] = '\0';
    return rel_path;
}
