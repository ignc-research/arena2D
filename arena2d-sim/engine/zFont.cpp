/* author: Cornelius Marx */
#include "zFont.hpp"
//creating static global ft library pointer
FT_Library zFont::_FT_LIB = 0;

zFont::zFont()
{
	//library uninitialized?
	if(_FT_LIB == NULL)
	{
		FT_Error error = FT_Init_FreeType(&_FT_LIB);
		if(error)
		{
			ERROR("Could not initialize FreeType!");
		}
	}
	//setting values
	_freeFace = false;
	_face = NULL;
	memset(_glyphs, 0, sizeof(_glyphs));
}

zFont::~zFont()
{
	free();
}

void zFont::free()
{
	if(_face != NULL)
	{
		//free glyphs
		for(int i = 0; i < Z_NUM_SIZE_SLOTS; i++)
		{
			if(_glyphs[i] != NULL)
			{
				delete(_glyphs[i]);
				_glyphs[i] = NULL;
			}
		}

		//free face
		if(_freeFace)
			FT_Done_Face(_face);
		_face = NULL;
	}
}

int zFont::loadFromFile(const char * path)
{
	FT_Face f;
	FT_Error error = FT_New_Face(_FT_LIB, path, 0, &f);
	if(error)
	{
		ERROR_F("Could not load font '%s'", path);
		return -1;
	}
	loadFromFace(f, true);
	return 0;
}

void zFont::loadFromFace(FT_Face face, bool freeFace)
{
	//freeing old font
	free();
	_face = face;
	_freeFace = freeFace;
}

void zFont::renderMap(zFontSize slot, float size, bool pointSize)
{
	if(_glyphs[slot] == NULL)//glyph has not been created yet
		_glyphs[slot] = new zGlyphMap();
	//creating new glyph map
	if(pointSize)
	{
		_glyphs[slot]->renderMap(_face, size, 0/*dpi*/);
	}
	else//pixel size
	{
		_glyphs[slot]->renderMap(_face, static_cast<int>(size));
	}
}

void zFont::renderDefaultSizes()
{
	//rendering all slots
	for(int i = 0; i < Z_NUM_SIZE_SLOTS; i++)
	{
		renderMap(static_cast<zFontSize>(i), Z_DEFAULT_FONT_SIZES[i], true);
	}
}

zGlyphMap * zFont::getGlyphMap(zFontSize size)
{
	//does glyph slot exist?
	if(_glyphs[size] == NULL)
	{
        WARNING_F("Font size %d has not been rendered, setting to default size", size);
		renderMap(size, Z_DEFAULT_FONT_SIZES[size], true);
	}

	return _glyphs[size];

}
