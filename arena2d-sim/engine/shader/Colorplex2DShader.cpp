#include "Colorplex2DShader.hpp"

Colorplex2DShader::Colorplex2DShader() : zShaderProgram()
{
	_color2Loc = -1;
	_patternTypeLoc = -1;
	_patternSizeLoc = -1;
}

int Colorplex2DShader::load()
{
    #include <colorplex2dShader.generated.h>
    if(loadProgram(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE) == 0){
        return -1;
	}

	_vertexLocation = getAttribLocation("vertex", false);
	_colorLocation = getAttribLocation("color", false);
	_color2Loc = getAttribLocation("color2", false);
	_cameraMatrixLocation = getAttribLocation("camera_mat", true);
	_modelViewMatrixLocation = getAttribLocation("modelview_mat", true);
	_projectionMatrixLocation = getAttribLocation("projection_mat", true);
	_patternTypeLoc = getAttribLocation("pattern_type", true);
	_patternSizeLoc = getAttribLocation("pattern_size", true);

	return 0;
}
