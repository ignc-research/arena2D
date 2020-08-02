/* author: Cornelius Marx */
#include "zTextView.hpp"

zTextView::zTextView()
{
	_textBuffer = 0;
	_capacity = 0;
	_font = NULL;
	_text = NULL;
	_xPos = 0.f;
	_yPos = 0.f;
	_size = Z_NORMAL;
	_used = 0;
	_boundingBox.x = 0.f;
	_boundingBox.y = 0.f;
	_boundingBox.w = 0.f;
	_boundingBox.h = 0.f;
	_color[0] = 0.f;
	_color[1] = 0.f;
	_color[2] = 0.f;
	_color[3] = 1.f;
	_aligned = false;
	_horizontalAlignment = 0;
	_verticalAlignment = 0;
	_alignMargin = 0;
	_pixelPerfectAlignment = false;
}

zTextView::zTextView(zFont * font, zFontSize size, int capacity): zTextView()
{
	//minimum capacity required
	if(capacity <= 0)
		capacity = 1;

	_font = font;
	_size = size;
	generateTextBuffer(capacity, false);
}

zTextView::zTextView(zFont * font, zFontSize size, const char * text, bool setBuffer): zTextView()
{
	init(font, size, text, setBuffer);
}

void zTextView::init(zFont * font, zFontSize size, const char * text, bool setBuffer)
{
	_font = font;
	_size = size;
	setText(text, setBuffer);
}

zTextView::~zTextView()
{
	//free text buffer if it was allocated
	if(_textBuffer != 0)
		glDeleteBuffers(1, &_textBuffer);

	//free text if it was allocated
	if(_text != NULL)
		delete[](_text);
}


void zTextView::generateTextBuffer(int capacity, bool copyToNew)
{
	//Z_LOG->printText("zTextView:generateTextBuffer");
	if(_textBuffer == 0)
		glGenBuffers(1, &_textBuffer);

	_capacity = capacity;
	glBindBuffer(GL_ARRAY_BUFFER, _textBuffer);
	if(copyToNew)//restore old content in new buffer
	{
		zBufferGlyph * g = new zBufferGlyph[_used];
		glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(zBufferGlyph)*_used, g);
		glBufferData(GL_ARRAY_BUFFER, sizeof(zBufferGlyph)*_capacity, NULL, GL_STATIC_DRAW);//NOTE: already allocated buffer will be automatically deleted
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(zBufferGlyph)*_used, g); 
		delete[](g);
	}
	else//old content is discarded
	{
		glBufferData(GL_ARRAY_BUFFER, sizeof(zBufferGlyph)*_capacity, NULL, GL_STATIC_DRAW);//NOTE: already allocated buffer will be automatically deleted
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void zTextView::setText(const char * text, bool setBuffer)
{
	//Z_LOG->printText("zTextView:setText ", text);
	int len = strlen(text);
	if(len > _capacity)//more capacity is needed
		generateTextBuffer(len, false);

	//copying text
	if(_text != NULL)
		delete[](_text);
	_text = new char[len+1];
	memcpy(_text, text, len+1);

	if(setBuffer)
	{
		//setting text buffer
		setTextBuffer(_text, len);
	}
}

void zTextView::refresh()
{
	if(_text != NULL)//text has been set
	{
		setTextBuffer(_text, strlen(_text));	
	}
}

void zTextView::setTextBuffer(const char * text, int len)
{
	//Z_LOG->printText("zTextView:setBuffer");
	//rendering text to buffer
	zBufferGlyph * glyphs = new zBufferGlyph[len];
	_used = _font->getGlyphMap(_size)->getTextBuffer(text, glyphs, len, &_boundingBox);
	glBindBuffer(GL_ARRAY_BUFFER, _textBuffer);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(zBufferGlyph)*_used, glyphs);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//free glyphs (openGL has them now)
	delete[](glyphs);

	//realign if text was aligned
	if(_aligned)
	{
		align();
	}
}

void zTextView::setCapacity(int capacity)
{
	if(capacity < _used)//shrink to fit
		generateTextBuffer(_used, true);
	else if(capacity > _used)//enlarge buffer
		generateTextBuffer(capacity, true);
	//nothing to do if capacity == _used
}

void zTextView::setPosition(float x, float y)
{
	_xPos = x;
	_yPos = y;
	_aligned = false;
}

void zTextView::setAlignment(float horizontal, float vertical, float margin, bool pixel_perfect)
{
	_aligned = true;
	_horizontalAlignment = horizontal;
	_verticalAlignment = vertical;
	_alignMargin = margin;
	_pixelPerfectAlignment = pixel_perfect;
}

void zTextView::setAlignment(float horizontal, float vertical, const zRect & box, float margin, bool pixel_perfect)
{
	_aligned = true;
	_horizontalAlignment = horizontal;
	_verticalAlignment = vertical;
	_alignBox = box;
	_alignMargin = margin;
	_pixelPerfectAlignment = pixel_perfect;
}

void zTextView::setAlignBox(const zRect & box)
{
	_alignBox = box;
}

void zTextView::align()
{
	if(!_aligned)
		return;
	float text_w_half = _boundingBox.w/2.f;

	//aligning horizontally
	_xPos = _alignBox.x - text_w_half;
	if(_horizontalAlignment < 0.f) {
		_xPos = _xPos - (_alignBox.x-_alignBox.w+_alignMargin - _xPos)*_horizontalAlignment;
	}
	else if(_horizontalAlignment > 0.f) {
		_xPos = _xPos + (_alignBox.x+_alignBox.w-_alignMargin-_boundingBox.w - _xPos)*_horizontalAlignment;
	}

	if(_pixelPerfectAlignment)
		_xPos = floor(_xPos);

	//aligning vertically
	_yPos = _alignBox.y - _boundingBox.h/2.0f + _boundingBox.y;
	if(_verticalAlignment < 0.f){
		_yPos = _yPos - (_alignBox.y-_alignBox.h+_alignMargin +_boundingBox.y - _yPos)*_verticalAlignment;
	}
	else if(_verticalAlignment > 0.f){
		_yPos = _yPos + (_alignBox.y+_alignBox.h-_alignMargin-_boundingBox.h+_boundingBox.y -_yPos)*_verticalAlignment;
	}

	if(_pixelPerfectAlignment)
		_yPos = floor(_yPos);
}

void zTextView::render()
{
	GLint 	vertexLocation = Z_SHADER->getVertexLoc(),
			texcoordLocation = Z_SHADER->getTexcoordLoc();
	//setting up vertex attrib pointers
	glBindBuffer(GL_ARRAY_BUFFER, _textBuffer);
	glEnableVertexAttribArray(vertexLocation);
	glEnableVertexAttribArray(texcoordLocation);
	glVertexAttribPointer(vertexLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float)*4, 0);
	glVertexAttribPointer(texcoordLocation, 2, GL_FLOAT, GL_FALSE, sizeof(float)*4, (const void*)(sizeof(float)*2));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	Z_SHADER->setColor(_color);
	//binding needed glyph texture
	_font->getGlyphMap(_size)->bindMap();
	//rendering at current position
	zMatrix4x4 m;
	m.setTranslation(zVector3D(_xPos, _yPos, 0.f));
	Z_SHADER->setModelviewMatrix(m);
	glDrawArrays(GL_QUADS, 0, _used*4);
	glDisableVertexAttribArray(vertexLocation);
	glDisableVertexAttribArray(texcoordLocation);
}
