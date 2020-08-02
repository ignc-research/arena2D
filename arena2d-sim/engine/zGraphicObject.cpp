/*
 * zGraphicObject.cpp
 *
 *  Created on: May 23, 2018
 *      Author: Cornelius Marx
 */

#include "zGraphicObject.hpp"

zGraphicObject::zGraphicObject()
{
	_buffer = 0;
	_drawMode = 0;
	_vertexCount = 0;
	_numComponentsPerPos = 0;
	_numComponentsPerUV = 0;
}

void zGraphicObject::init(GLenum draw_mode, int num_verts, const float * data, int num_components_per_position, int num_components_per_uv)
{
	free();
	_drawMode = draw_mode;
	_vertexCount = num_verts;
	_numComponentsPerPos = num_components_per_position;
	_numComponentsPerUV = num_components_per_uv;

	//creating new buffer
	glGenBuffers(1, &_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, _buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*(num_verts*(num_components_per_position + num_components_per_uv)), data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void zGraphicObject::free()
{
	if(_buffer != 0)
	{
		glDeleteBuffers(1, &_buffer);
		_buffer = 0;
		_drawMode = 0;
		_vertexCount = 0;
		_numComponentsPerPos = 0;
		_numComponentsPerUV = 0;
	}
}

float* zGraphicObject::createCircleData(int poly_count)
{
	float *circle_data = new float[(poly_count+2)*2];
	float a_per_i = 2.0f*static_cast<float>(M_PI)/static_cast<float>(poly_count);
	for(int i = 1; i < poly_count+1; i++)
	{
		float angle = (i-1)*a_per_i;
		circle_data[i*2 + 0] = static_cast<float>(cos(angle));
		circle_data[i*2 + 1] = static_cast<float>(sin(angle));
	}
	circle_data[0] = 0.f;
	circle_data[1] = 0.f;
	circle_data[poly_count*2 + 2] = circle_data[2];
	circle_data[poly_count*2 + 3] = circle_data[3];
	return circle_data;
}

void zGraphicObject::bind(GLint vertexLocation)
{
	glBindBuffer(GL_ARRAY_BUFFER, _buffer);
	glVertexAttribPointer(vertexLocation, _numComponentsPerPos, GL_FLOAT, GL_FALSE,
							sizeof(float)*(_numComponentsPerPos + _numComponentsPerUV),0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void zGraphicObject::bind(GLint vertexLocation, GLint textureLocation)
{
	glBindBuffer(GL_ARRAY_BUFFER, _buffer);
	glVertexAttribPointer(vertexLocation, _numComponentsPerPos, GL_FLOAT, GL_FALSE,
							sizeof(float)*(_numComponentsPerPos + _numComponentsPerUV),0);
	if(_numComponentsPerUV > 0)//has texture-coordinate
	{
		glVertexAttribPointer(textureLocation, _numComponentsPerUV, GL_FLOAT, GL_FALSE,
								sizeof(float)*(_numComponentsPerPos + _numComponentsPerUV),
								reinterpret_cast<void*>(sizeof(float)*_numComponentsPerPos));
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
