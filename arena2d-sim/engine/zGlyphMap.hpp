/*
Created on: 16th March 2018
Author: Cornelius Marx

Description: zGlyphMap
	- class represents a gl-texture containing a bitmap font (rendered by class zFont) 
*/
#ifndef Z_GLYPH_MAP_H
#define Z_GLYPH_MAP_H

#include "zLogfile.hpp"
#include "zRect.hpp"
#include "glew/glew.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include <SDL2/SDL.h>

#define Z_GLYPH_MAP_NUM_CHARACTERS 128

//representing a glyph in graphics buffer
//vertices start at top left corner and go around CCW (0,0) -> (0,1) -> (1,1) -> (1,0)
struct zBufferGlyph
{
	float v0[2];//vertex coordinate
	float t0[2];//texture coordinate

	float v1[2];
	float t1[2];

	float v2[2];
	float t2[2];

	float v3[2];
	float t3[2];
};//each vertex is represented by a vec4 in shader (2 for vertex coord, 2 for texture coord)

//representing a rendered glyph
struct zGlyph
{
	unsigned int width;//dimensions of glyph
	unsigned int height;
	int leftPos;//position relative to cursor
	int topPos;
	float advanceX;//number of 1/64th pixels to offset for next character
	float advanceY;//number of 1/64th pixels to offset for next character
};

//enum zTextAlign {Z_ALIGN_LEFT, Z_ALIGN_CENTER, Z_ALIGN_RIGHT};

class zGlyphMap
{
public:
	///constructors
	zGlyphMap();

	///destructor
	~zGlyphMap();

	//free map (is called automatically before renderMap() or by destructor)
	void freeMap();

	//render glyph map
	int renderMap(FT_Face face, float pt_size, unsigned int DPI);//point size
	int renderMap(FT_Face face, int px_size);//pixel size
	// calculate pixels without setting up texture etc.
	// free return value by calling delete[](@return)
	unsigned char* renderMapPixels(FT_Face, int px_size, int & width, int & height);

	//bind map to GL_TEXTURE_2D target
	void bindMap();

	//creating buffer for fast text rendering
	//@param text: the text to be put in buffer
	//@param xoff, yoff: translating each vertex coordinate by that amount
	//@param buffer: buffer must be provided to put glyphs into
	//@param buffer_size: number of zBufferGlyph-Objects in buffer
	//@param rect: sets the dimensions of the total text and the offset to the upper left corner, can be set to NULL
	//@return : number of glyphs in buffer
	int getTextBuffer(const char * text, zBufferGlyph * buffer, int buffer_size, zRect * rect);

	///get
	float getSize(){return _size;}
	bool isPointSize(){return _pointSize;}
	unsigned int getMaxGlyphWidth(){return _maxGlyphWidth;}
	unsigned int getMaxGlyphHeight(){return _maxGlyphHeight;}
	int getMaxGlyphTop(){return _maxGlyphTop;}
	int getMinGlyphBot(){return _minGlyphBot;}

	void getGlyph(unsigned char c, zGlyph * g){*g = _glyphs[static_cast<int>(c)];}

	void setMap(GLuint texture_id){_map = texture_id;}
private:
	///private functions
	//render map from current settings of this object (size, pointSize)
	//this function shall only be called by the renderMap()-functions hence beeing private
	GLubyte* renderGlyphMap(FT_Face face, int DPI, int & map_w, int & map_h);

	// create texture from pixels, number ofpixels in given array must be 16*_maxGlyphWidth*16*_maxGlyphHeight
	void createTexture(GLubyte * pixels, int map_w, int map_h);

	float _size;
	bool _pointSize;//true: _size is a point size else _size is a pixel size
	unsigned int _maxGlyphWidth;//maximum dimensions a glyph can have
	unsigned int _maxGlyphHeight;
	int _maxGlyphTop;//position of the glyph reaching highest relative to base line (used for calculating bounding box)
	int _minGlyphBot;//bottom position of the glyph reaching the lowest relative to base line (used for calculating bounding box)
	GLuint _map;//GL texture map with dimensions 16*maxGlyphWidth, 16*maxGlyphHeight
	zGlyph _glyphs[256];
};

#endif
