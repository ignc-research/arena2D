/*
 * TextShader.cpp
 *
 *  Created on: May 20, 2018
 *      Author: zer0divider
 */

#include "TextShader.hpp"

TextShader::TextShader() : zShaderProgram()
{

}

int TextShader::load()
{
    #include <textShader.generated.h>
    if(loadProgram(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE) == 0)
        return -1;
    
    initAttribLocations("vertex", "color", "uv", "tex", "modelMat", "cameraMat", "projMat");
    return 0;
}

void TextShader::initAttribLocations(const char * vertex, const char * color, const char * uv,
								const char * texSampler, const char * model_mat, const char * camera_mat, const char * proj_mat)
{
	_vertexLocation = getAttribLocation(vertex, false);
	_colorLocation = getAttribLocation(color, false);
	_texcoordLocation = getAttribLocation(uv, false);
	_textureLocation = getAttribLocation(texSampler, true);
	_cameraMatrixLocation = getAttribLocation(camera_mat, true);
	_modelViewMatrixLocation = getAttribLocation(model_mat, true);
	_projectionMatrixLocation = getAttribLocation(proj_mat, true);
    setTexture(0);
}


