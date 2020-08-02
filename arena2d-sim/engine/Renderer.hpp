/*
 * renderer.h
 *
 *  Created on: May 23, 2018
 *      Author: Cornelius Marx
 *      Description: OpenGL interface: handles shader programs and graphical objects
 */

#ifndef RENDERER_H_
#define RENDERER_H_

#include "f_math.h"
#include "zSingleton.hpp"
#include "zGraphicObject.hpp"
#include "zMatrix4x4.hpp"
#include "shader/TextShader.hpp"
#include "shader/SpriteShader.hpp"
#include "shader/ParticleShader.hpp"
#include "shader/Color2DShader.hpp"
#include "shader/Colorplex2DShader.hpp"
#include "Quadrangle.hpp"
#include <limits>

#define _RENDERER Renderer::get()

//default graphic objects
enum RENDERER_GO{GO_NONE, GO_CENTERED_QUAD, GO_CENTERED_QUAD_LINE, GO_QUAD, GO_QUAD_LINE, GO_TRIANGLE, GO_TRIANGLE_LINE, GO_LINE, GO_CIRCLE, GO_CIRCLE_LINE, GO_CIRCLE_SMOOTH, GO_CIRCLE_SMOOTH_LINE,
							GO_NUMBER};

#define CIRCLE_VERTEX_COUNT 16
#define CIRCLE_SMOOTH_VERTEX_COUNT 32 

//
struct Vertex2D
{
	zVector2D pos;
	zVector2D uv;
    void set(const zVector2D & _pos, const zVector2D & _uv){pos = _pos; uv = _uv;}
	static inline void setUVPointer(int uv_loc){glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex2D), reinterpret_cast<void*>(sizeof(zVector2D)));}
	static inline void setPosPointer(int pos_loc){glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex2D), 0);}
};

typedef Vertex2D FrameVert;

class Renderer : public zTSingleton<Renderer>
{
public:
	Renderer();
	~Renderer();

	//initializing shader and graphic objects
	//@return -1 on failure, else 0
	int init(bool verbose = true);

	//accessing/using shader
	TextShader * getTextShader(){return _textShader;}
	void useTextShader(){_textShader->useProgram();}
	SpriteShader * getSpriteShader(){return _spriteShader;}
	void useSpriteShader(){_spriteShader->useProgram();}
	ParticleShader * getParticleShader(){return _particleShader;}
	void useParticleShader(){_particleShader->useProgram();}
	Colorplex2DShader * getColorplexShader(){return _colorplex2DShader;}
	void useColorplexShader(){_colorplex2DShader->useProgram();}
	Color2DShader * getColorShader(){return _color2DShader;}
	void useColorShader(){_color2DShader->useProgram();}


	//calculate matrizes from given screen dimensions
	void refreshProjectionMatrizes(int screenW, int screenH);
	void getProjectionScreenMatrix(zMatrix4x4 *proj){*proj = _projScreenMatrix;}
	void getProjectionGLMatrix(zMatrix4x4 *proj){*proj = _projGLMatrix;}

	//setting pre-calculated matrix in current shader
	void setScreenMatrix(){Z_SHADER->setProjectionMatrix(_projScreenMatrix);}
	
	void setGLMatrix(){Z_SHADER->setProjectionMatrix(_projGLMatrix);}

	// load identity to modelviewmatrix
	void resetModelviewMatrix(){zMatrix4x4 m; m.loadIdentity(); Z_SHADER->setModelviewMatrix(m);}
	void resetCameraMatrix(){zMatrix4x4 m; m.loadIdentity(); Z_SHADER->setCameraMatrix(m);}
	void resetProjectionMatrix(){zMatrix4x4 m; m.loadIdentity(); Z_SHADER->setProjectionMatrix(m);}

	// setting modelview matrix to 2D scale and translation
	void set2DTransform(const zVector2D & pos, const zVector2D & scale);
	void set2DTransform(const zVector2D & pos, float uniform_scale);
	void set2DTransform(const zVector2D & pos, float uniform_scale, float rad);
	void set2DTransform(const zVector2D & pos, const zVector2D & scale, float rad);

	//getting current screen view
	//NOTE: r mustn't be NULL!
	void getGLRect(zRect * r){*r = _GLRect;}
	void getScreenRect(zRect * r){*r = _screenRect;}

	/* transforms a rect from  screencoords/glcoords to glcoords/screencoords*/
	void toGLCoordinates(zRect * r);
	void toGLCoordinates(zVector2D & v, bool relative = false);
	void toScreenCoordinates(zRect * r);

	//returns the current bounding box of visible screen
	void getTransformedGLRect(const zMatrix4x4& inverse_camera, Quadrangle * q);
	void getTransformedScreenRect(const zMatrix4x4& inverse_camera, Quadrangle * q);

	void getTransformedRect(const zRect * r, const zMatrix4x4& inverse_camera, Quadrangle * q);

	//setting render object (Graphic Object)
	void bindGO(RENDERER_GO go);

	//renders currently bound Graphic object
	void drawGO();

	// draw given rect using GO_CENTERED_QUAD
	// @r is a rect with half dimensions and the origin in the center
	void drawRect(const zRect & r){set2DTransform(zVector2D(r.x, r.y), zVector2D(r.w*2, r.h*2)); bindGO(GO_CENTERED_QUAD); drawGO();}

	void bindBlankTexture(GLenum target = GL_TEXTURE_2D){glBindTexture(target, _blankTexture);}

	//enabling/disabling gl attributes with one command
	void setEnabled(GLenum cap, GLboolean enable){if(enable == GL_TRUE)glEnable(cap); else glDisable(cap);}

	///saving/restoring openGL-settings
	//stencil func and op
	void pushStencilSettings();
	void popStencilSettings();

	//blend func and equation
	void pushBlendSettings();
	void popBlendSettings();

private:
	TextShader * _textShader;
    SpriteShader * _spriteShader;
	ParticleShader *_particleShader;
	Colorplex2DShader * _colorplex2DShader;
	Color2DShader * _color2DShader;

	void createBlankTexture();

	//default graphic objects
	zGraphicObject _graphicObjects[GO_NUMBER];
	RENDERER_GO _boundGO;//currently bound GO

	zMatrix4x4 _projScreenMatrix;//native screen-viewport
	zMatrix4x4 _projGLMatrix;//homogene clipspace (adjusted to screen aspect)

	zRect _GLRect;//current dimensions of screen when using gl projection matrix (w, h are half dimensions)
	zRect _screenRect;//current dimensions of screen when using screen projection matrix (w, h are half dimensions)

	GLuint _blankTexture;

	//save variables for restoring openGL-settings
	//stencil func and op
	struct StencilSettings
	{
		//stencil enabled
		GLboolean enabled;
		//func
		GLint func;
		GLint ref;
		GLuint mask;
		//op
		GLint fail, zfail, zpass;
	} _savedStencil;

	//blend func and equation
	struct BlendSettings
	{
		//blending enabled
		GLboolean enabled;
		//func
		GLint sRGB;
		GLint sALPHA;
		GLint dRGB;
		GLint dALPHA;
		//equation
		GLint mode;
	} _savedBlend;
};

#endif /* RENDERER_H_ */
