#ifndef COLORPLEX2D_SHADER_H
#define COLORPLEX2D_SHADER_H

#include "zShaderProgram.hpp"

class Colorplex2DShader : public zShaderProgram
{
public:
	enum PatternType{NONE, HORIZONTAL, VERTICAL, DIAGONAL_NEGATIVE, DIAGONAL_POSITIVE, TILES};
	Colorplex2DShader();
	~Colorplex2DShader(){}

	int load();

	void setColor2(const float* rgba){glVertexAttrib4fv(_color2Loc, rgba);}
	void setColor2(const zColor & c){glVertexAttrib4fv(_color2Loc, (const float*)&c);}
	void setPatternType(PatternType type){glUniform1i(_patternTypeLoc, type);}
	void setPatternSize(int size){glUniform1i(_patternSizeLoc, size);}

private:
	GLint _color2Loc;
	GLint _patternTypeLoc;
	GLint _patternSizeLoc;
};

#endif
