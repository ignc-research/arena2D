#include "zShaderProgram.hpp"

zShaderProgram * zShaderProgram::_currentShader = NULL;

//Constructor
zShaderProgram::zShaderProgram()
{
	//initialize
	_programID = 0;
	_vertexLocation = -1;
	_texcoordLocation = -1;
	_colorLocation = -1;
	_textureLocation = -1;
	_projectionMatrixLocation = -1;
	_cameraMatrixLocation = -1;
	_modelViewMatrixLocation = -1;
}

//freeProgram
void zShaderProgram::freeProgram()
{
	//delete program
	glDeleteProgram(_programID);
	_programID = -1;
}

//loadProgramFromFile
GLuint zShaderProgram::loadProgramFromFile(const char * vertex_shader_path, const char * fragment_shader_path)
{
	std::string vs_source, fs_source;
    INFO_F("Loading Shader-Program from file ('%s', '%s')", vertex_shader_path, fragment_shader_path);
	//loading vertex shader source
	if(zStringTools::loadFromFile(vertex_shader_path, &vs_source) < 0)
	{
		ERROR_F("Could not open vertex shader source from '%s'", vertex_shader_path);
		return 0;
	}
	//loading fragment shader source
	if(zStringTools::loadFromFile(fragment_shader_path, &fs_source) < 0)
	{
		ERROR_F("Could not open fragment shader source from '%s'", fragment_shader_path);
        return 0;
	}

	return loadProgram(vs_source.c_str(), fs_source.c_str());
}

//loadProgram
GLuint zShaderProgram::loadProgram(const char * vertex_shader_source, const char * fragment_shader_source)
{
	GLint success = 0;
	int error = 0;
	//create and compile vertex shader
	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &vertex_shader_source, 0);
	glCompileShader(vertex_shader);
	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
	if(success == GL_FALSE){
		ERROR("error during vertex compilation");
		error++;	
	}

	//create and compile fragment shader
	GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &fragment_shader_source, 0);
	glCompileShader(fragment_shader);
	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
	if(success == GL_FALSE){
		ERROR("error during fragment compilation");
		error++;	
	}

	//create and link program with vs and fs
	_programID = glCreateProgram();
	glAttachShader(_programID, vertex_shader);
	glAttachShader(_programID, fragment_shader);
	glLinkProgram(_programID);
	glGetShaderiv(_programID, GL_LINK_STATUS, &success);
	if(success == GL_FALSE){
		ERROR("error during shader linking");
		error++;	
	}

	//Error-handling:
		static const int bufSize = 1024;
		GLchar *buffer = new GLchar [bufSize];

		//Vertex Shader Error-Log
		glGetShaderInfoLog(vertex_shader, bufSize, 0, buffer);
		buffer[bufSize-1] = '\0';
		if(buffer[0] != '\0')//non-empty
		{
			ERROR_F("While compiling vertex shader:\n%s", (const char*) buffer);
		}
		//Fragment Shader Error-Log
		glGetShaderInfoLog(fragment_shader, bufSize, 0, buffer);
		buffer[bufSize-1] = '\0';
		if(buffer[0] != '\0')//non-empty
		{
            ERROR_F("While compiling fragment shader:\n%s", (const char*) buffer);
		}

		//Program Error-Log
		glGetProgramInfoLog(_programID, bufSize, 0, buffer);
		buffer[bufSize-1] = '\0';
		if(buffer[0] != '\0')//non-empty
		{
            ERROR_F("While linking shader program:\n%s", (const char*)buffer);
		}
		delete[] buffer;
	/////
	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);
	//return ID
	return error > 0? 0 : _programID;
}

void zShaderProgram::setDefaultAttributes( const char * vertex_name, const char * color_name, const char * texcoord_name, 
								const char * texture_name, const char * projMat_name, const char * cameraMat_name, const char * modelViewMat_name)
{
	_vertexLocation = getAttribLocation(vertex_name, false);
	_colorLocation = getAttribLocation(color_name, false);
	_texcoordLocation = getAttribLocation(texcoord_name, false);
	_textureLocation = getAttribLocation(texture_name, true);
	_projectionMatrixLocation = getAttribLocation(projMat_name, true);
	_cameraMatrixLocation = getAttribLocation(cameraMat_name, true);
	_modelViewMatrixLocation = getAttribLocation(modelViewMat_name, true);
	//default: enabling vertex array setting active texture
	enableVertexArray(true);
	setTexture(0);
}

GLint zShaderProgram::getAttribLocation(const char * name, bool isUniform)
{
	GLint loc = -1;
	if(name != NULL)
	{
		if(isUniform)
		{
			loc = glGetUniformLocation(_programID, name);
			if(loc < 0)
				WARNING_F("Could not find uniform in shader: %s", name);
		}
		else
		{
			loc = glGetAttribLocation(_programID, name);
			if(loc < 0)
                WARNING_F("Could not find attribute in shader: %s", name);
		}
	}

	return loc;
}

//useProgram
void zShaderProgram::useProgram()
{
	if(_currentShader != NULL)
		_currentShader->unuseProgram();
	_currentShader = this;
	glUseProgram(_programID);
}

void zShaderProgram::setProjectionMatrixFromGL()
{
	float m[16];
	glGetFloatv(GL_PROJECTION_MATRIX, m);
	glUniformMatrix4fv(_projectionMatrixLocation, 1, GL_FALSE, m);
}

void zShaderProgram::setCameraMatrixFromGL()
{
	float m[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, m);
	glUniformMatrix4fv(_cameraMatrixLocation, 1, GL_FALSE, m);
}

void zShaderProgram::setModelViewMatrixFromGL()
{
	float m[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, m);
	glUniformMatrix4fv(_modelViewMatrixLocation, 1, GL_FALSE, m);
}
