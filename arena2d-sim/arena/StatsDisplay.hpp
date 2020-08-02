/* Author: Cornelius Marx */
#ifndef STATS_DISPLAY_H
#define STATS_DISPLAY_H

#include <engine/zFramework.hpp>
#include <engine/Renderer.hpp>
#include <engine/GlobalSettings.hpp>
#include <engine/zTextView.hpp>
#include <vector>

#define METRIC_VALUE_CAPACITY 10 // capacity of value text view in MetricHandle
#define STATS_DISPLAY_LINE_SPACING 1.5 // factor multiplied with font size to determine space in pixels between lines
#define STATS_DISPLAY_MARGIN 5

/* forward declarations */
class StatsDisplay;
class MetricHandle;

/* single metric used by StatsDisplay*/
class MetricHandle
{
public:
	/* constructor
	 */
	MetricHandle(zFont * font, zFontSize font_size, const char * text);

	/* destructor
	 */
	~MetricHandle();

	/* updates text view
	 */
	void updateView();

	/* render text and value view
	 * text shader must be bound, vertex and texcoord attributes must be enabled
	 */
	void render();

	/* set value to floating point number
	 * @param value the float to set the value view text to
	 * @param precision number of decimal places to show
	 */
	void setFloatValue(float value, int precision = 1);

	/* set value to integer number
	 * @param value the integer to set the value view text to
	 */
	void setIntValue(int value);

	/* set value to string "value1/value2"
	 * @param value1 left value in ratio
	 * @param value2 right value in ratio
	 */
	void setRatioValue(int value1, int value2);

	/* set value to string
	 * @param value string to set the value view text to
	 */
	void setStringValue(const char * value);

	/* set text
	 * @param text new text to set the text view to
	 */
	void setText(const char * text);

	/* get text
	 * @return text of the metric (without value)
	 */
	const char* getText(){return _text;}

	/* get length of text
	 * @return number of characters in the text without null-terminator
	 */
	int getTextLen(){return _textLen;}

	zTextView* getTextView(){return _textView;}
	zTextView* getValueView(){return _valueView;}
private:
	char * _text;
	int _textLen;
	zTextView * _textView;
	zTextView * _valueView;
}; 

/* display for easy visualization of metrics such as mean reward */
class StatsDisplay
{
public:
	/* constructor
	 * @param font font to use for metrics
	 * @param font_size font size for metrics
	 * @param text_color color of metric value view
	 * @param value_color color of metric text view
	 * @param background_color color of background box
	 */
	StatsDisplay(zFont * font, zFontSize font_size,
					const zColor & text_color,
					const zColor & value_color,
					const zColor & background_color);

	/* destructor
	 */
	~StatsDisplay();

	/* call if window was resized or new metrics added to list to update text positions 
	 */
	void refresh();

	/* render background
	 * color shader (or similar) must be bound with screen projection matrix
	 */
	void renderBackground();

	/* render text views
	 * text shader must be bound with screen projection matrix
	 */
	void renderText();

	/* add metric to metric list
	 * @param text metric text
	 * @return pointer to newly created metric, can be used to set the metric value
	 * NOTE: memory is handled by this class, do not free returned pointer
	 */
	MetricHandle* addMetric(const char * text);
private:

	/* vector storing all metrics */
	std::vector<MetricHandle*> _metrics;

	/* background box */
	zRect _box;

	/* font to use for metrics */
	zFont * _font;

	/* font size to use for metrics */
	zFontSize _fontSize;

	/* color of metric text view */
	zColor _textColor;

	/* color of metric value view */
	zColor _valueColor;

	/* background color of box */
	zColor _backgroundColor;
};

#endif
