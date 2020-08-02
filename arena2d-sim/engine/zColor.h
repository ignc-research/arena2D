//Created: 15th Jan 2017
//Author: Cornelius Marx

//this class describes an RGBA-Color

#ifndef ZER0_COLOR_H
#define ZER0_COLOR_H
#include <math.h>

struct zColor
{
    static const zColor WHITE;
    static const zColor BLACK;
    static const zColor GREY;
	//constructors
	zColor(): r(0.f), g(0.f), b(0.f), a(1.f) {}
	zColor(float _r, float _g, float _b, float _a): r(_r), g(_g), b(_b), a(_a) {}
	zColor(const float * rgba): r(rgba[0]), g(rgba[1]), b(rgba[2]), a(rgba[3]) {}
	zColor(const zColor & c){*this = c;}
	zColor(unsigned int rgba, bool msb_red = true){set(rgba, msb_red);}
	//zColor(unsigned char _r, unsigned char _g, unsigned char _b, unsigned char _a){setFromByte(_r,_g,_b,_a);}

	void set(float _r, float _g, float _b, float _a = 1.f){r = _r; g = _g; b = _b; a = _a;}
	void setRGB(float _r, float _g, float _b){r = _r; g = _g; b = _b;}//alpha wont change
	void set(unsigned int rgba, bool msb_red = true);

	//destructor
	~zColor(){}

	//initializers
	void loadWhite(){r = 1.f; g = 1.f; b = 1.f; a = 1.f;}
	void loadBlack(){r = 0.f; g = 0.f; b = 0.f; a = 1.f;}
	void loadZero(){r = 0.f; g = 0.f; b = 0.f; a = 0.f;}

	//converters
	void setFromByte(unsigned char _r, unsigned char _g, unsigned char _b, unsigned char _a);
	unsigned int getHex(bool msb_red = true) const;

	//cast to float-pointer
	operator float* () {return (float*)this;}
	operator const float* ()const{return (const float*)this;}
	
	//color operations
	void darken(float d);//make color darker by given amount (percentage)
	void brighten(float b);//make color brighter by given amount (percentage)
	void desaturate(float s);//desaturate color (percentage) 1 -> grey
	void saturate(float s);//saturate color (percentage) 1 -> crisp color

	// linear interpolate between two given colors
	static zColor getInterpolated(const zColor & a, const zColor & b, float t){return zColor(a.r + (b.r-a.r)*t, a.g + (b.g-a.g)*t, a.b + (b.b-a.b)*t, a.a + (b.a-a.a)*t);}

	//members
	float r, g, b, a;
};

#endif
