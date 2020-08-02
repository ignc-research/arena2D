/*
 * Color2DShader.h
 *
 *  Created on: May 19, 2018
 *      Author: zer0divider
 */

#ifndef COLOR2DSHADER_H_
#define COLOR2DSHADER_H_

#include "zShaderProgram.hpp"

class Color2DShader : public zShaderProgram
{
public:
	Color2DShader();
	~Color2DShader(){}

	int load();
private:

};


#endif /* COLOR2DSHADER_H_ */
