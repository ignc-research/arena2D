/*
 * Color2DShader.cpp
 *
 *  Created on: May 19, 2018
 *      Author: zer0divider
 */

#include "Color2DShader.hpp"

Color2DShader::Color2DShader() : zShaderProgram()
{

}

int Color2DShader::load()
{
	#include <color2dShader.generated.h>
	if(loadProgram(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE) == 0)
		return -1;

	_vertexLocation = getAttribLocation("vertex", false);
	_colorLocation = getAttribLocation("color", false);
	_cameraMatrixLocation = getAttribLocation("camera_mat", true);
	_modelViewMatrixLocation = getAttribLocation("modelview_mat", true);
	_projectionMatrixLocation = getAttribLocation("projection_mat", true);

	return 0;
}

