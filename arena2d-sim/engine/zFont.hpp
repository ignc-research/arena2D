/*
Created on: 16th March 2018
Author: Cornelius Marx

Description: zFont
	- class for loading and handling true-type fonts
	- it is possible to handle up to 7 different text sizes in one zFont-Object
*/
#ifndef Z_FONT_H
#define Z_FONT_H

#include "zGlyphMap.hpp"
#include "zLogfile.hpp"

//abstract font sizes from small to big
enum zFontSize {Z_TINY, Z_VERY_SMALL, Z_SMALL, Z_NORMAL, Z_BIG, Z_VERY_BIG, Z_HUGE, Z_NUM_SIZE_SLOTS};
//default pt-sizes mapped to abstract font sizes (e.g. Z_TINY-> 8.f)
const float Z_DEFAULT_FONT_SIZES[Z_NUM_SIZE_SLOTS] = {8.f, 10.f, 12.f, 16.f, 20.f, 24.f, 32.f};

class zFont
{
public:
	///constructors
	zFont();

	///destructor
	~zFont();

	//free all data allocated for this font (called on destructor or when a font is reloaded from file)
	void free();

	//load font from file
	//@return -1 on error else 0
	int loadFromFile(const char * path);

	//loading font from already loaded font face
	//@param free tells whether the face source should be freed by this object
	void loadFromFace(FT_Face face, bool freeFace);

	//render bit map to a size-slot at given size
	//@param pointSize : if false -> @param size is considered to be a pixel-size
	void renderMap(zFontSize slot, float size, bool pointSize = false);

	//renders all slots using default point sizes
	void renderDefaultSizes();

	///getter
	//get glyph map for given font size (e.g. Z_SMALL)
	//@return if size slot is not filled NULL is returned
	zGlyphMap * getGlyphMap(zFontSize size);
	

	// get freetype library font
	FT_Face getFace(){return _face;}

private:
	bool _freeFace;
	zGlyphMap * _glyphs[Z_NUM_SIZE_SLOTS];
	FT_Face _face;

	//static global ft-library
	static FT_Library _FT_LIB;

};


#endif
