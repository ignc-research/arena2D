/* author: Cornelius Marx */
#include "zGlyphMap.hpp"
zGlyphMap::zGlyphMap()
{
	_map = 0;
	_maxGlyphWidth = 0;
	_maxGlyphHeight = 0;
	_maxGlyphTop = 0;
	_minGlyphBot = 0;
}

zGlyphMap::~zGlyphMap()
{
}

void zGlyphMap::freeMap()
{
	if(_map != 0)
	{
		glDeleteTextures(1,  &_map);
		_map = 0;
		_maxGlyphWidth = 0;
		_maxGlyphHeight = 0;
		_maxGlyphTop = 0;
		_minGlyphBot = 0;
	}
}

void zGlyphMap::bindMap()
{
	glBindTexture(GL_TEXTURE_2D, _map);
}

int zGlyphMap::renderMap(FT_Face face, float pt_size, unsigned int DPI)
{
	_size = pt_size;
	_pointSize = true;
	int map_w, map_h;
	GLubyte* pixels =  renderGlyphMap(face, 0, map_w, map_h);
	if(pixels == NULL)
		return -1;
	createTexture(pixels, map_w, map_h);
	delete[]pixels;
	return 0;
}

int zGlyphMap::renderMap(FT_Face face, int px_size)
{
	_size = static_cast<float>(px_size);
	_pointSize = false;
	int map_w, map_h;
	GLubyte* pixels =  renderGlyphMap(face, 0, map_w, map_h);
	if(pixels == NULL)
		return -1;
	createTexture(pixels, map_w, map_h);
	delete[]pixels;
	return 0;
}


unsigned char* zGlyphMap::renderMapPixels(FT_Face face, int px_size, int & map_w, int & map_h)
{
	_size = static_cast<float>(px_size);
	_pointSize = false;
	GLubyte* pixels =  renderGlyphMap(face, 0, map_w, map_h);
	return pixels;
}

unsigned char* zGlyphMap::renderGlyphMap(FT_Face face, int DPI, int & map_w, int & map_h)
{
	//free old map
	freeMap();

	GLubyte * bitmaps[Z_GLYPH_MAP_NUM_CHARACTERS];
	memset(bitmaps, 0, sizeof(bitmaps));

	//rendering every glyph separately
	if(_pointSize)
	{
		FT_Set_Char_Size(face, static_cast<int>(64*_size), static_cast<int>(64*_size), DPI, DPI);
	}
	else//pixel size
	{
		FT_Set_Pixel_Sizes(face, 0, static_cast<int>(_size));
	}

	for(int i = 0; i < Z_GLYPH_MAP_NUM_CHARACTERS; i++)
	{
		//loading character
		if(FT_Load_Char(face, i, FT_LOAD_RENDER))//Error?
		{
			memset((_glyphs+i), 0, sizeof(zGlyph));
			ERROR_F("While glyph rendering at ascii character '%c' (%d)\n", static_cast<unsigned char>(i), i);
			continue;
		}
		FT_GlyphSlot g = face->glyph;//shortcut
		//setting glyph dimensions
		unsigned int w = g->bitmap.width;
		unsigned int h = g->bitmap.rows;
		_glyphs[i].width = w;
		_glyphs[i].height = h;
		_glyphs[i].leftPos = g->bitmap_left;
		_glyphs[i].topPos = g->bitmap_top;
		_glyphs[i].advanceX = g->advance.x;
		_glyphs[i].advanceY = g->advance.y;
		if(w*h > 0)
		{
			bitmaps[i] = new GLubyte[w*h];
			memcpy(bitmaps[i],g->bitmap.buffer, w*h);

			//new max dimension found?
			if(w > _maxGlyphWidth)
				_maxGlyphWidth = w;

			if(h > _maxGlyphHeight)
				_maxGlyphHeight = h;

			//new lowest/highest reaching
			if(g->bitmap_top > _maxGlyphTop)
				_maxGlyphTop = g->bitmap_top;

			int bitmap_low = g->bitmap_top - static_cast<int>(h);
			if(_minGlyphBot > bitmap_low)
				_minGlyphBot = bitmap_low;
		}
	}

	//create  master glyph map
	map_w = 16 * _maxGlyphWidth;
	map_h = 16 * _maxGlyphHeight;
	GLuint num_pixels = map_w * map_h;
	GLubyte * map_pixels = new GLubyte[num_pixels];
	//making all transparent first
	memset(map_pixels, 0, num_pixels); 
	//copy over every glyph into big collection
	for(int i = 0; i < Z_GLYPH_MAP_NUM_CHARACTERS; i++)
	{
		if(bitmaps[i] != NULL)
		{
			int start_col = (i%16)*_maxGlyphWidth;
			int start_row = (i/16)*_maxGlyphHeight;
			int g_w = (int)_glyphs[i].width;
			int g_h = (int)_glyphs[i].height;
			for(int y = 0; y < g_h; y++)
			{
				for(int x = 0; x < g_w; x++)
				{
					map_pixels[(start_row+y)*map_w + start_col + x] = bitmaps[i][(y*g_w) + x];
				}
			}
			//free glyph bitmap
			delete[](bitmaps[i]);
		}
	}


	return map_pixels;
}

void zGlyphMap::createTexture(GLubyte * pixels, int map_w, int map_h)
{
	//create texture for glyph map
	glGenTextures(1, &_map);
	glBindTexture(GL_TEXTURE_2D, _map);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);//setting pixel alignment
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, map_w, map_h, 0, GL_RED, GL_UNSIGNED_BYTE, pixels);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);//reset pixel alignment to default
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}

int zGlyphMap::getTextBuffer(const char * text, zBufferGlyph * buffer, int buffer_size, zRect * rect)
{
	int count = 0;
	zRect b;
	float cursorX = 0.f;
	float cursorY = 0.f;
	float maxCursorX = 0.f;
	b.x = 0.f;
	b.y = 0.f;
	float topH = 0.f;//highest and lowest y-coordinate 
	float lowH = 0.f;
	float mapW = _maxGlyphWidth*16;
	float mapH = _maxGlyphHeight*16;
	while(*text != '\0' && count < buffer_size)
	{
		int c = static_cast<int>(static_cast<unsigned char>(*text));
		if((char)c != '\n' && _glyphs[c].width * _glyphs[c].height > 0)//glyph is actually visible
		{
			zBufferGlyph * g = (buffer+count);
			int mapX = _maxGlyphWidth*(c%16);
			int mapY = _maxGlyphHeight*(c/16);
			float originX = cursorX + _glyphs[c].leftPos;
			float originY = cursorY - _glyphs[c].topPos;
			//finding hightest and lowest character
			if(_glyphs[c].topPos > topH)
				topH = _glyphs[c].topPos;
			float botPos = static_cast<float>(_glyphs[c].topPos) - static_cast<float>(_glyphs[c].height);
			if(botPos < lowH)
				lowH = botPos;

			g->v0[0]= originX;
			g->v0[1] = originY;
			g->t0[0] = mapX/mapW; 
			g->t0[1] = mapY/mapH; 

			g->v1[0] = originX;
			g->v1[1] = originY	+ _glyphs[c].height;
			g->t1[0] = (mapX						)/mapW; 
			g->t1[1] = (mapY	+ _glyphs[c].height )/mapH; 

			g->v2[0] = originX	+ _glyphs[c].width;
			g->v2[1] = originY	+ _glyphs[c].height;
			g->t2[0] = (mapX	+ _glyphs[c].width)/mapW; 
			g->t2[1] = (mapY	+ _glyphs[c].height)/mapH; 

			g->v3[0]= originX	+ _glyphs[c].width;
			g->v3[1] = originY;
			g->t3[0] = (mapX	+ _glyphs[c].width)/mapW; 
			g->t3[1] = (mapY					  )/mapH; 
			count++;
		}
		//shift cursor
		if((char)c == '\n'){// line break
			if(cursorX > maxCursorX)
				maxCursorX = cursorX;
			cursorX = 0.0f;
			cursorY += _maxGlyphHeight;
		}else{
			cursorX += _glyphs[c].advanceX/64.f;
			cursorY += _glyphs[c].advanceY/64.f;
		}
		text++;
	}
	if(cursorX > maxCursorX)
		maxCursorX = cursorX;
	b.w = maxCursorX;
	b.h = cursorY + _maxGlyphHeight;
	b.y = _maxGlyphTop;
	if(rect != NULL)
		*rect = b;

	return count;
}
