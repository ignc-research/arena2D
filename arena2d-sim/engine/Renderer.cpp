/*
 * renderer.cpp
 *
 *  Created on: May 23, 2018
 *      Author: Cornelius Marx
 */

#include "Renderer.hpp"


Renderer::Renderer()
{
	_textShader = NULL;
	_spriteShader = NULL;
	_particleShader = NULL;
	_boundGO = GO_NONE;
}

Renderer::~Renderer()
{
	//free shader
    delete _textShader;
	delete _spriteShader;
	glDeleteTextures(1, &_blankTexture);
}

int Renderer::init(bool verbose)
{
	INFO("Initializing OpenGL Renderer...");
	//loading shader
	if(verbose)
		INFO("-> Initializing text shader");
	_textShader = new TextShader();
	if(_textShader->load())
		return -1;

	if(verbose)
		INFO("-> Initializing sprite shader");
    _spriteShader = new SpriteShader();
    if(_spriteShader->load())
		return -1;

	if(verbose)
		INFO("-> Initializing particle shader");
	_particleShader = new ParticleShader();
	if(_particleShader->load())
		return -1;

	if(verbose)
		INFO("-> Initializing color shader");
	_color2DShader = new Color2DShader();
	if(_color2DShader->load())
		return -1;

	if(verbose)
		INFO("-> Initializing colorplex shader");
	_colorplex2DShader = new Colorplex2DShader();
	if(_colorplex2DShader->load())
		return -1;


	// init blank texture
	if(verbose)
		INFO("-> Creating blank texture");
	createBlankTexture();

	if(verbose)
		INFO("-> Creating 2D primitives");
	//creating graphic objects
	float quad_center_data[] = {
		-0.5f, 0.5f,
		0.f, 0.f,

		-0.5f, -0.5f,
		0.f, 1.f,

		0.5f, 0.5f,
		1.f, 0.f,

		0.5f, -0.5f,
		1.f, 1.f
	};
	float quad_center_data_no_uv[] = {
		-0.5f, 0.5f,
		-0.5f, -0.5f,
		0.5f, -0.5f,
		0.5f, 0.5f,
	};
	_graphicObjects[GO_CENTERED_QUAD].init(GL_TRIANGLE_STRIP, 4, quad_center_data, 2, 2);
	_graphicObjects[GO_CENTERED_QUAD_LINE].init(GL_LINE_LOOP, 4, quad_center_data_no_uv, 2, 0);

	float quad_data[] = {
		0.f, 1.f,
		0.f, 0.f,

		1.f, 1.f,
		1.f, 0.f,

		0.f, 0.f,
		0.f, 1.f,

		1.f, 0.f,
		1.f, 1.f
	};
	float quad_data_no_uv[] = {
		0.f, 1.f,
		1.f, 0.f,
		0.f, 0.f,
		1.f, 1.f,
	};
	_graphicObjects[GO_QUAD].init(GL_TRIANGLE_STRIP, 4, quad_data, 2, 2);
	_graphicObjects[GO_QUAD_LINE].init(GL_LINE_LOOP, 4, quad_data_no_uv, 2, 0);

	float tri_data[6];
	for(int i = 0; i < 3; i++)
	{
		float angle = static_cast<float>(i)*(2.f/3.f)*static_cast<float>(M_PI);
		tri_data[i*2 + 0] = static_cast<float>(sin(angle));
		tri_data[i*2 + 1] = static_cast<float>(cos(angle));
	}
	_graphicObjects[GO_TRIANGLE].init(GL_TRIANGLES, 3, tri_data, 2, 0);
	_graphicObjects[GO_TRIANGLE_LINE].init(GL_LINE_LOOP, 3, tri_data, 2, 0);

	float *circle_data = zGraphicObject::createCircleData(CIRCLE_VERTEX_COUNT);
	_graphicObjects[GO_CIRCLE].init(GL_TRIANGLE_FAN, CIRCLE_VERTEX_COUNT+2, circle_data, 2, 0);
	_graphicObjects[GO_CIRCLE_LINE].init(GL_LINE_STRIP, CIRCLE_VERTEX_COUNT+1, (circle_data +2), 2, 0);
	delete[] circle_data;

	circle_data = zGraphicObject::createCircleData(CIRCLE_SMOOTH_VERTEX_COUNT);
	_graphicObjects[GO_CIRCLE_SMOOTH].init(GL_TRIANGLE_FAN, CIRCLE_SMOOTH_VERTEX_COUNT+2, circle_data, 2, 0);
	_graphicObjects[GO_CIRCLE_SMOOTH_LINE].init(GL_LINE_STRIP, CIRCLE_SMOOTH_VERTEX_COUNT+1, (circle_data +2), 2, 0);
	delete[] circle_data;

	float line_data[] = {0.0f, 0.0f, 1.f, 1.f};
	_graphicObjects[GO_LINE].init(GL_LINES, 2, line_data, 2, 0);

	return 0;
}

void Renderer::createBlankTexture()
{
	glGenTextures(1, &_blankTexture);
	glBindTexture(GL_TEXTURE_2D, _blankTexture);
	GLuint white = 0xFFFFFFFF;
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, &white);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::bindGO(RENDERER_GO g)
{
	_boundGO = g;
	if(_boundGO != GO_NONE)
	{
		GLint texloc = Z_SHADER->getTexcoordLoc();
		if(texloc < 0)
			_graphicObjects[g].bind(Z_SHADER->getVertexLoc());
		else
			_graphicObjects[g].bind(Z_SHADER->getVertexLoc(), texloc);
	}
}

void Renderer::drawGO()
{
	if(_boundGO != GO_NONE)
	{
		//enabling attrib arrays
		GLint texloc = Z_SHADER->getTexcoordLoc();
		bool enableTex = texloc >= 0 && _graphicObjects[_boundGO].hasUV();//need to enable texture coordinates -array
		if(enableTex)
			Z_SHADER->enableTexcoordArray(true);
		Z_SHADER->enableVertexArray(true);

		_graphicObjects[_boundGO].draw();

		//restore default (no attrib arrays)
		Z_SHADER->enableVertexArray(false);
		if(enableTex)
			Z_SHADER->enableTexcoordArray(false);
	}
}

void Renderer::refreshProjectionMatrizes(int screenW, int screenH)
{
	//setting projection matrices
	_projScreenMatrix.setOrtho(0, static_cast<float>(screenW), static_cast<float>(screenH), 0, -1.f, 1.f);
	_screenRect.x = screenW/2.f;
	_screenRect.y = screenH/2.f;
	_screenRect.w = _screenRect.x;
	_screenRect.h = _screenRect.y;

	//keeping aspect ratio in gl-coordinates
	float x,y;
	if(screenW > screenH)
	{
		y = 1.f;
		x = static_cast<float>(screenW)/screenH;
	}
	else
	{
		x = 1.f;
		y = static_cast<float>(screenH)/screenW;
	}
	_projGLMatrix.setOrtho(-x, x, -y, y, -1, 1);
	_GLRect.x = 0.f;
	_GLRect.y = 0.f;
	_GLRect.w = x;//half dimension
	_GLRect.h = y;
}

void Renderer::set2DTransform(const zVector2D & pos, float uniform_scale){
	set2DTransform(pos, zVector2D(uniform_scale, uniform_scale));
}

void Renderer::set2DTransform(const zVector2D & pos, float uniform_scale, float rad){
	set2DTransform(pos, zVector2D(uniform_scale, uniform_scale), rad);
}

void Renderer::set2DTransform(const zVector2D & pos, const zVector2D & scale, float rad)
{
	zMatrix4x4 transform;
	transform.set2DTransform(pos, scale, rad);
	Z_SHADER->setModelviewMatrix((float*)transform);
}

void Renderer::set2DTransform(const zVector2D & pos, const zVector2D & scale)
{
	zMatrix4x4 transform;
	transform.set2DTransform(pos, scale);
	Z_SHADER->setModelviewMatrix((float*)transform);
}

void Renderer::getTransformedGLRect(const zMatrix4x4& inverse_camera, Quadrangle * q)
{
	getTransformedRect(&_GLRect, inverse_camera, q);
}

void Renderer::getTransformedScreenRect(const zMatrix4x4& inverse_camera, Quadrangle * q)
{
	getTransformedRect(&_screenRect, inverse_camera, q);
}

void Renderer::getTransformedRect(const zRect * r, const zMatrix4x4& inverse_camera, Quadrangle * q)
{
	zVector4D v[4];
	v[0].set(r->x+r->w, r->y+r->h, 0, 1);
	v[1].set(r->x+r->w, r->y-r->h, 0, 1);
	v[2].set(r->x-r->w, r->y-r->h, 0, 1);
	v[3].set(r->x-r->w, r->y+r->h, 0, 1);
	zVector2D * q_v = (zVector2D*)q;
	for(int i = 0; i < 4; i++)
	{
		v[i] = inverse_camera*v[i];
		q_v[i].set(v[i].x, v[i].y);
	}
}

void Renderer::toGLCoordinates(zRect * r)
{
	r->w *= (_GLRect.w/_screenRect.w);
	r->h *= (_GLRect.h/_screenRect.h);
	r->x = r->x*(_GLRect.w/_screenRect.w) - _GLRect.w;
	r->y = (_screenRect.h - r->y)*(_GLRect.h/_screenRect.h);
}

void Renderer::toGLCoordinates(zVector2D & v, bool relative)
{
	if(relative){
		v.x *= (_GLRect.w/_screenRect.w);
		v.y *= -(_GLRect.h/_screenRect.h);
	}
	else{
		v.x = v.x*(_GLRect.w/_screenRect.w) - _GLRect.w;
		v.y = (_screenRect.h - v.y)*(_GLRect.h/_screenRect.h);
	}
}

void Renderer::toScreenCoordinates(zRect * r)
{	
	r->w *= (_screenRect.w/_GLRect.w);
	r->h *= (_screenRect.h/_GLRect.h);
	r->x = (r->x+_GLRect.w)*(_screenRect.w/_GLRect.w);
	r->y = _screenRect.h - r->y*(_screenRect.h/_GLRect.h);
}

void Renderer::pushStencilSettings()
{
	//enabled
	glGetBooleanv(GL_STENCIL_TEST, &_savedStencil.enabled); 
	//func
	glGetIntegerv(GL_STENCIL_FUNC, &_savedStencil.func);
	glGetIntegerv(GL_STENCIL_REF, &_savedStencil.ref);
	glGetIntegerv(GL_STENCIL_VALUE_MASK, (GLint*)&_savedStencil.mask);

	//op
	glGetIntegerv(GL_STENCIL_FAIL, &_savedStencil.fail);
	glGetIntegerv(GL_STENCIL_PASS_DEPTH_FAIL, &_savedStencil.zfail);
	glGetIntegerv(GL_STENCIL_PASS_DEPTH_PASS, &_savedStencil.zpass);
}

void Renderer::popStencilSettings()
{
	setEnabled(GL_STENCIL_TEST, _savedStencil.enabled);	
	glStencilFunc(_savedStencil.func, _savedStencil.ref, _savedStencil.mask);
	glStencilOp(_savedStencil.fail, _savedStencil.zfail, _savedStencil.zpass);
}


void Renderer::pushBlendSettings()
{
	//enabled
	glGetBooleanv(GL_BLEND, &_savedBlend.enabled);

	//func
	glGetIntegerv(GL_BLEND_SRC_RGB, &_savedBlend.sRGB);
	glGetIntegerv(GL_BLEND_SRC_ALPHA, &_savedBlend.sALPHA);
	glGetIntegerv(GL_BLEND_DST_RGB, &_savedBlend.dRGB);
	glGetIntegerv(GL_BLEND_DST_ALPHA, &_savedBlend.dALPHA);

	//mode
	glGetIntegerv(GL_BLEND_EQUATION, &_savedBlend.mode);
}

void Renderer::popBlendSettings()
{
	setEnabled(GL_BLEND, _savedBlend.enabled);
	glBlendFuncSeparate(_savedBlend.sRGB, _savedBlend.dRGB, _savedBlend.sALPHA, _savedBlend.dALPHA);
	glBlendEquation(_savedBlend.mode);
}
