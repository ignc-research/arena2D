/*
 * zGraphicObject.h
 *
 *  Created on: May 23, 2018
 *      Author: Cornelius Marx
 */

#ifndef ZGRAPHICOBJECT_H_
#define ZGRAPHICOBJECT_H_

#include "glew/glew.h"
#define _USE_MATH_DEFINES
#include <math.h>


class zGraphicObject
{
public:
	zGraphicObject();
	~zGraphicObject(){free();}

	//initialize graphic object and fill gl-buffer
	//@param draw_mode : gl draw mode
	//@param num_verts : number of vertices
	//@param data : contains for each vertex the position and uv-coords(optional)
	//@param num_components: number of floats per vertex
	void init(GLenum draw_mode, int num_verts, const float * data, int num_components_per_position, int num_components_per_uv = 0);

	//delete gl buffer
	void free();

	//setting vertex attributes in shader
	//NOTE: needed vertex attributes must have array enabled
	void bind(GLint vertexLocation);
	void bind(GLint vertexLocation, GLint textureLocation);

	//draw object via gl drawing command
	//NOTE: bind() needs to be called before draw()
	void draw(){glDrawArrays(_drawMode, 0, _vertexCount);}

	//creating circle vertex data with given amount
	//created data has a size of sizeof(float)*(poly_count+2)*2
	//NOTE: vertices must be drawn with GL_TRIANGLE_FAN
	static float* createCircleData(int poly_count);

	//this does this go have uv components?
	bool hasUV(){return _numComponentsPerUV > 0;}

	int getVertexCount(){return _vertexCount;}

	// get buffer
	GLuint getBuffer(){return _buffer;}
private:
	GLuint _buffer;//gl buffer, data for uv and position is stored for each vertex
	GLenum _drawMode;//how are the vertices draw (e.g. GL_TRIANGLES)
	int _vertexCount;//number of vertices to draw
	GLubyte _numComponentsPerPos;//number of floats per vertex for position
	GLubyte _numComponentsPerUV;//number of floats per vertex for uv
};


#endif /* ZGRAPHICOBJECT_H_ */
