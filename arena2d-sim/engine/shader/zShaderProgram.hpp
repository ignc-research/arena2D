/*
Created on: 24th March 2018
Author: Cornelius Marx
Description:
	- class represents a shader program in application
	- it is thought of as beeing a superclass for every shaderprogram
*/

#ifndef Z_SHADER_PROGRAM_H
#define Z_SHADER_PROGRAM_H

//quick accessing current shader
#define Z_SHADER zShaderProgram::getCurrentShader()

#include "../glew/glew.h"
#include <string.h>//for strcpy and strcmp
#include <stdio.h>
#include "../zStringTools.hpp"
#include "../zLogfile.hpp"
#include "../zColor.hpp"

//shader-program-class
class zShaderProgram
{
public:
	//constructor
	zShaderProgram();

	//destructor
	virtual ~zShaderProgram(){if(_currentShader == this)_currentShader = NULL; freeProgram();}

	//free Program and its attributes
	void freeProgram();

	//loading shader from string or file
	//@return: the program ID (-> 0 on error)
	GLuint loadProgram(const char * vertex_shader_source, const char * fragment_shader_source);
	GLuint loadProgramFromFile(const char * vertex_shader_path, const char * fragment_shader_path);

	//getting program-ID
	GLuint getProgramID(){return _programID;}

	//useProgram: sets the current program, calls glUseProgram(this_programID)
	virtual void useProgram();

	// called if shader program was switched
	virtual void unuseProgram(){}

	//fetch attribute/uniform location from shader
	GLint getAttribLocation(const char * name, bool isUniform); 

	//setting default attribute/uniform locations from name in shader source
	//@params can be set to NULL if it doesn't exist in shader to avoid warnings
	void setDefaultAttributes( const char * vertex_name, const char * color_name, const char * texcoord_name, 
								const char * texture_name, const char * projMat_name, const char  * cameraMat_name, const char * modelViewMat_name);

	//getting locations
	GLint getVertexLoc(){return _vertexLocation;}
	GLint getTexcoordLoc(){return _texcoordLocation;}
	GLint getColorLoc(){return _colorLocation;}
	GLint getTextureLoc(){return _textureLocation;}
	GLint getProjectionMatrixLoc(){return _projectionMatrixLocation;}
	GLint getCameraMatrixLoc(){return _cameraMatrixLocation;}
	GLint getmodelViewMatrixLoc(){return _modelViewMatrixLocation;}

	//setting attributes/uniforms
	void enableVertexArray(bool enabled){if(enabled)glEnableVertexAttribArray(_vertexLocation); else glDisableVertexAttribArray(_vertexLocation);}
	void setVertexAttribPointer(GLsizei stride, GLsizei offset){glVertexAttribPointer(_vertexLocation, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offset));}
	void setVertexAttribPointer2D(GLsizei stride, GLsizei offset){glVertexAttribPointer(_vertexLocation, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offset));}//setting 2d-vector
	void enableTexcoordArray(bool enabled){if(enabled)glEnableVertexAttribArray(_texcoordLocation); else glDisableVertexAttribArray(_texcoordLocation);}
	void setTexcoordAttribPointer(GLsizei stride, GLsizei offset){glVertexAttribPointer(_texcoordLocation, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offset));}
	void enableColorArray(bool enabled){if(enabled)glEnableVertexAttribArray(_colorLocation); else glDisableVertexAttribArray(_colorLocation);}
	void setColorAttribPointer(GLsizei stride, GLsizei offset){glVertexAttribPointer(_colorLocation, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offset));}
	void setColor(const float * rgba){glVertexAttrib4fv(_colorLocation, rgba);}
	void setColor(float r, float g, float b, float a){glVertexAttrib4f(_colorLocation, r, g, b, a);}
	void setColor(const zColor & c){setColor((const float*)&c);}
	void setTexture(int active){glUniform1i(_textureLocation, active);}
	void setProjectionMatrix(const float * matrix){glUniformMatrix4fv(_projectionMatrixLocation, 1, GL_FALSE, matrix);}
	void setCameraMatrix(const float * matrix){glUniformMatrix4fv(_cameraMatrixLocation, 1, GL_FALSE, matrix);}
	void setModelviewMatrix(const float * matrix){glUniformMatrix4fv(_modelViewMatrixLocation, 1, GL_FALSE, matrix);}
	//setting matrices from GL's matrix-stack
	void setProjectionMatrixFromGL();//setting with GL_PROJECTION_MATRIX
	void setCameraMatrixFromGL();//setting with GL_MODELVIEW_MATRIX projection
	void setModelViewMatrixFromGL();//setting with GL_MODELVIEW_MATRIX projection

	//getting global, currently used shader object
	static zShaderProgram* getCurrentShader(){return _currentShader;}
protected:
	///private members
	GLuint _programID;

	//default attributes/uniforms most shaders have
	GLint _vertexLocation;
	GLint _texcoordLocation;
	GLint _colorLocation;
	GLint _textureLocation;
	GLint _projectionMatrixLocation;
	GLint _cameraMatrixLocation;
	GLint _modelViewMatrixLocation;

	//global shader program
	static zShaderProgram * _currentShader;
};

#endif
