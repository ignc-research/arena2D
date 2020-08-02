/*
 * SpriteShader.cpp
 *
 *  Created on: August 12, 2019
 *      Author: zer0divider
 */

#include "SpriteShader.hpp"

SpriteShader::SpriteShader() : zShaderProgram()
{
}

int SpriteShader::load(){
    #include <spriteShader.generated.h>
    if(loadProgram(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE) == 0){
        return -1;
	}
    
	_vertexLocation = getAttribLocation("vertex", false);
	_colorLocation = getAttribLocation("color", false);
	_texcoordLocation = getAttribLocation("uv", false);
	_textureLocation = getAttribLocation("tex", true);
	_cameraMatrixLocation = getAttribLocation("cameraMat", true);
	_modelViewMatrixLocation = getAttribLocation("modelMat", true);
	_projectionMatrixLocation = getAttribLocation("projMat", true);
    setTexture(0);
    return 0;
}
