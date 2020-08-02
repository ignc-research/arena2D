/* Author: Cornelius Marx */
#include "StatsDisplay.hpp"

MetricHandle::MetricHandle(zFont * font, zFontSize font_size, const char * text):_text(NULL)
{
	_textView = new zTextView(font, font_size, 0);
	_valueView = new zTextView(font, font_size, METRIC_VALUE_CAPACITY);
	setText(text);
}

MetricHandle::~MetricHandle()
{
	delete[] _text;
	delete _textView;
	delete _valueView;
}

void MetricHandle::setText(const char * text)
{
	delete[] _text;
	_textLen = strlen(text);
	_text = new char[_textLen+1];
	strcpy(_text, text);
	_textView->setText(_text);
	_textView->align();
}

void MetricHandle::setFloatValue(float value, int precision)
{
	static char buffer[32];
	sprintf(buffer, "%.*f", precision, value);
	_valueView->setText(buffer);
	_textView->align();
}

void MetricHandle::setIntValue(int value)
{
	static char buffer[32];
	sprintf(buffer, "%d", value);
	_valueView->setText(buffer);
	_textView->align();
}

void MetricHandle::setRatioValue(int value1, int value2)
{
	static char buffer[32];
	sprintf(buffer, "%d/%d", value1, value2);
	_valueView->setText(buffer);
	_textView->align();
}

void MetricHandle::setStringValue(const char * value)
{
	_valueView->setText(value);
	_textView->align();
}

void MetricHandle::render()
{
	_textView->render();
	_valueView->render();
}

/** Stats Display **/
StatsDisplay::StatsDisplay(zFont * font, zFontSize font_size,
					const zColor & text_color,
					const zColor & value_color,
					const zColor & background_color): _font(font), _fontSize(font_size),
					_textColor(text_color), _valueColor(value_color), _backgroundColor(background_color)
{

}

StatsDisplay::~StatsDisplay()
{
	for(unsigned int i = 0; i < _metrics.size(); i++){
		delete _metrics[i];
	}
}

MetricHandle* StatsDisplay::addMetric(const char * text)
{
	MetricHandle * m = new MetricHandle(_font, _fontSize, text);
	m->getTextView()->setColor(_textColor);
	m->getValueView()->setColor(_valueColor);

	_metrics.push_back(m);

	return m;
}

void StatsDisplay::renderText()
{
	for(unsigned int i = 0; i < _metrics.size(); i++){
		_metrics[i]->render();
	}
}

void StatsDisplay::renderBackground()
{
	Z_SHADER->setColor(_backgroundColor);
	_RENDERER->drawRect(_box);
}

void StatsDisplay::refresh()
{
	if(_metrics.size() == 0){
		return;
	}

	// get longest text
	int max_index = 0;
	int max_len = _metrics[0]->getTextLen();
	for(unsigned int i = 1; i < _metrics.size(); i++){
		int l = _metrics[i]->getTextLen();
		if(l > max_len){
			max_index = i;
			max_len = l;
		}
	}

	// dummy text view for measuring size of value views
	static char buffer[METRIC_VALUE_CAPACITY+1];
	memset(buffer, '0', sizeof(buffer));
	buffer[METRIC_VALUE_CAPACITY] = '\0';
	zTextView * t = new zTextView(_font, _fontSize, buffer);
	zRect dummy_box = t->getBoundingBox();
	delete t;
	t = NULL;

	// calculate box
	const float line_spacing = _font->getGlyphMap(_fontSize)->getSize()*STATS_DISPLAY_LINE_SPACING;
	zRect text_box = _metrics[max_index]->getTextView()->getBoundingBox();
	_box.w = text_box.w + dummy_box.w;
	_box.h = _metrics.size()*line_spacing + 2*STATS_DISPLAY_MARGIN;
	_box.w /= 2.0f;
	_box.h /= 2.0f;
	
	_box.x = Z_FW->getWindowW()-_box.w;
	_box.y = _box.h;

	// align all text views
	for(unsigned int i = 0; i < _metrics.size(); i++){
		float i_per = (float)i/(_metrics.size()-1);
		zTextView * text_view = _metrics[i]->getTextView();
		zTextView * value_view = _metrics[i]->getValueView();
		text_view->setAlignment(-1, -1+2*i_per, _box, STATS_DISPLAY_MARGIN, true);
		text_view->align();
		value_view->setAlignment(1, -1+2*i_per, _box, STATS_DISPLAY_MARGIN, true);
		value_view->align();
	}
}
