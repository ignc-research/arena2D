/* author: Cornelius Marx */
#include "zStringTools.hpp"

//loadFromFile
int zStringTools::loadFromFile(const char * filename, std::string * text)
{
	//open file
	FILE * f = fopen(filename, "r");

	//error?
	if(f == 0)
	{
		return -1;
	}
	
	if(text)
	{
		//read file and write it in text
		(*text) = "";
		while(1)
		{
			char c;
			if(fread(&c, 1, 1, f) <= 0)//end of file?
				break;

			//append character
			(*text) += c;
		}
	}
	//closing file
	fclose(f);
	return 1;
}

//storeToFile
int zStringTools::storeToFile(const char * filename, const char * text)
{
	//open file
	FILE * f = fopen(filename, "w");

	//error?
	if(f == 0)
	{
		return -1;
	}
	if(text)
	{
		//write to file
		int i = 0;
		while(1)
		{
			if(text[i] == '\0')//end of string
				break;
			//write out
			fwrite(text + i, 1, 1, f);
			i++;
		}
	}
	fclose(f);
	return 1;
}

int zStringTools::fromInt(int a, char * str)
{
    /*check for negative sign*/
    int invert = 0;
    if(a < 0){
        *str = '-';
        str++;
        a = -a;/*invert a so it can be passed to kstr_from_uint()*/
        invert = 1;/*setting flag*/
    }
    /*retrieving the total number of digits*/
    int num_digits = 0;/*digit counter*/
    int tmp_a = a;
    do{
        tmp_a /= 10;
        num_digits++;
    }while(tmp_a != 0);

    str[num_digits] = '\0';/*writing '\0'-terminator*/
    for(int i = num_digits-1; i >= 0; i--)/*going from right to left and writing each digit*/
    {
        str[i] = '0' + a%10;	
        a /= 10;
    }

    return num_digits+invert;
}

int zStringTools::toInt(const char * c, int * error)
{
	int sign = 1;
	zStringTools::skipWhiteSpace(&c);
	if(*c == '-')
	{
		sign = -1;
		c++;
	}
	else if(*c == '+')
	{
		sign = 1;
		c++;
	}
	zStringTools::skipWhiteSpace(&c);

	int num = 0;
	while(!zStringTools::isWhiteSpace(*c) && *c != '\0')
	{
		if(*c < '0' || *c > '9')//no number
		{
			if(error != NULL)
				*error = 1;
			return 0;
		}

		num *= 10;
		num += ((int)*c) - ((int)'0');
		c++;
	}

	if(error != NULL)
		*error = 0;
	return num*sign;
}

float zStringTools::toFloat(const char * c, int * error)
{
	float sign = 1.f;
	zStringTools::skipWhiteSpace(&c);
	if(*c == '-')
	{
		sign = -1.f;
		c++;
	}
	else if(*c == '+')
	{
		sign = 1.f;
		c++;
	}
	zStringTools::skipWhiteSpace(&c);

	float num = 0.f;
	int point = -1;
	int num_digits = 0;
	while(!zStringTools::isWhiteSpace(*c) && *c != '\0')
	{
		if(*c == '.')
		{
			if(point >= 0)//error, point already found
			{
				if(error != NULL)
					*error = 1;
				return 0;
			}
			point = num_digits;
			c++;
			continue;
		}
		if(*c < '0' || *c > '9')//no number
		{
			if(error != NULL)
				*error = 1;
			return 0;
		}

		num *= 10.f;
		num += (float)(((int)*c) - ((int)'0'));
		num_digits++;
		c++;
	}
	//shift number by considering point-position
	if(point >= 0 && point < num_digits)
	{
		int div = 1;
		for(int i = 0; i < num_digits-point; i++)
			div*= 10;

		num /= (float)div;
	}

	if(error != NULL)
		*error = 0;

	return num*sign;
}

int zStringTools::isNumber(char c){
	return c >= '0' && c <= '9';
}

int zStringTools::isLetter(char c){
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

char zStringTools::goTo(char c, const char ** text)
{
	while(**text != c && **text != '\0') (*text)++;

	return **text;
}

int zStringTools::fromFloat(float value, char * buffer){
	sprintf(buffer, "%f", value);
	int len = strlen(buffer);
	// special codes -> just return
	if((buffer[1] < '0' || buffer[1] > '9') && buffer[1] != '.')
		return len;

	// normal value -> cutoff trailing 0s
	int last_index = len-1;
	while(buffer[last_index] == '0'){
		last_index--;
	}

	if(buffer[last_index] == '.'){
		last_index++;
	}
	buffer[last_index+1] = '\0';
	return last_index+1;
}

int zStringTools::isWhiteSpace(char c)
{
	return c == ' ' || c == '\t' || c == '\n' || c== '\r';
}

int zStringTools::startsWith(const char * a, const char * b)
{
	while(*a != '\0' && *b != '\0'){
		if(*a != *b)
			return 0;

		a++;
		b++;
	}

	return 1;
}

int zStringTools::skipWhiteSpace(const char ** c)
{
	int lines = 0;
	while(zStringTools::isWhiteSpace(**c))
	{
		if(**c == '\n')// count lines
			lines++;

		(*c)++;
	}

	return lines;
}

int zStringTools::skipWhiteLine(const char ** c)
{
	int line_count = 0;
	while(zStringTools::isWhiteSpace(**c))
	{
		if(**c == '\n')
			line_count++;
		(*c)++;
	}
	return line_count;
}

int zStringTools::charIsElementOf(char c, const char *str, int str_len)
{
	for(int i = 0; i < str_len; i++)
	{
		if(c == str[i])//character found
			return 1;
	}
	return 0;
}

int zStringTools::isAlphanum(char c)
{
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || (c == '-' || c == '+');
}

void zStringTools::toLower(std::string & s)
{
	int l = s.length();
	for(int i = 0; i < l; i++)
		std::tolower(s[i]);
}

char* zStringTools::createCString(const std::string & s)
{
	char * new_s = (char*)malloc(sizeof(char)*(s.length()+1));
	strcpy(new_s, s.c_str());
	return new_s;
}

char* zStringTools::createCString(const char * s)
{
	char * new_s = (char*)malloc(sizeof(char)*(strlen(s)+1));
	strcpy(new_s, s);
	return new_s;
}

