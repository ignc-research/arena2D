/*
	void initAttribLocations(const char * vertex, const char * color,
								const char * model_matrix, const char * camera_matrix, const char * projection_matrix);
 * SpriteShader.h
 *
 *  Created on: August 12, 2019
 *      Author: zer0divider
 */

#ifndef SPRITE_SHADER_H
#define SPRITE_SHADER_H

#include "zShaderProgram.hpp"

class SpriteShader : public zShaderProgram
{
public:
	SpriteShader();
	~SpriteShader(){}

    // compile shader from source, returns 0 on success, -1 on error
    int load();
    
	//SpriteShader uses only 2 cordinates for vertices
	void setVertexAttribPointer(GLsizei stride, GLsizei offset){glVertexAttribPointer(_vertexLocation, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offset));}
};


#endif
