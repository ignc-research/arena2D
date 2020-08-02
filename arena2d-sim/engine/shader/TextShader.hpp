/*
 * TextShader.h
 *
 *  Created on: May 20, 2018
 *      Author: zer0divider
 */

#ifndef TEXTSHADER_H_
#define TEXTSHADER_H_

#include "zShaderProgram.hpp"

class TextShader : public zShaderProgram
{
public:
	TextShader();
	~TextShader(){}

	void initAttribLocations(const char * vertex, const char * color, const char * uv,
								const char * texSampler, const char * model_mat, const char * camera_mat, const char * proj_mat);
    
    // compile shader from source, returns 0 on success, -1 on error
    int load();
	//textshader uses only 2 cordinates for vertices
	void setVertexAttribPointer(GLsizei stride, GLsizei offset){glVertexAttribPointer(_vertexLocation, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offset));}
};


#endif /* TEXTSHADER_H_ */
