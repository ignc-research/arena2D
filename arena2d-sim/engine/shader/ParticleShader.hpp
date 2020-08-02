#ifndef PARTICLE_SHADER_H
#define PARTICLE_SHADER_H

#include "zShaderProgram.hpp"
#include <engine/zVector2d.hpp>
#include <engine/zColor.hpp>
#include <assert.h>

struct ParticleTriangle{
	ParticleTriangle(){}
	ParticleTriangle(float _x, float _y, float _scale, float _rad, float _r, float _g, float _b, float _a):
	x(_x), y(_y), scale(_scale), rad(_rad), r(_r), g(_g), b(_b), a(_a){}

	void set(float _x, float _y, float _scale, float _rad, float _r, float _g, float _b, float _a){
		x = _x;
		y = _y;
		scale = _scale;
		rad = _rad;
		r = _r;
		g = _g;
		b = _b;
		a = _a;
	}
	void set(const zVector2D & pos, float _scale, float _rad, const zColor & c){
		x = pos.x;
		y = pos.y;
		scale = _scale;
		rad = _rad;
		r = c.r;
		g = c.g;
		b = c.b;
		a = c.a;
	}
	float x, y, scale, rad;
	float r, g, b, a;
};

class ParticleShader : public zShaderProgram
{
public:
	ParticleShader();
	~ParticleShader();

	int load();
	// overwriting useProgram for binding vertexID/triangleID buffers
	void useProgram();
	void unuseProgram();

	int getMaxNumParticles(){return _maxNumParticles;}

	// uploading triangles to gpu uniform buffer
	void setTriangles(int size, int offset, ParticleTriangle* triangles){
		assert(size+offset <= _maxNumParticles);
		glUniform4fv(_triangleUniform+offset*2, 2*size, (const GLfloat*)triangles);
	}
	void draw(int size, int offset = 0){glDrawArrays(GL_TRIANGLES, offset, size*3);}
private:
	GLint _triangleUniform;
	GLint _vertexIDLocation;
	GLint _triangleIDLocation;
	int _maxNumParticles;
	GLuint _vertexIDBuffer;
	GLuint _triangleIDBuffer;
};

#endif
