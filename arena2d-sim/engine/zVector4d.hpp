//Created: 15th Jan 2017
//Author: Cornelius Marx

#ifndef ZER0_VECTOR4D_H
#define ZER0_VECTOR4D_H

#include <math.h>

class zVector4D
{
public:
	//constructors
	zVector4D(): x(0.f), y(0.f), z(0.f), w(0.f){}
	zVector4D(float _x, float _y, float _z, float _w): x(_x), y(_y), z(_z), w(_w){}
	zVector4D(float * v): x(v[0]), y(v[1]), z(v[2]), w(v[3]){}
	zVector4D(const zVector4D& v){*this = v;}

	~zVector4D(){}

	//initializers
	void loadZero(){x=0.f; y=0.f; z=0.f; w=0.f;}
	void loadOne(){x=1.f; y=1.f; z=1.f; w=1.f;}
	void set(float new_x, float new_y, float new_z, float new_w){x = new_x; y = new_y; z = new_z; w = new_w;}

	//operators
	//transform vector from homogene clipspace to cameraspace through dividing by the w-component
	void fromHomogene(){x/=w; y/=w; z/=w; w=1;}
	zVector4D getFromHomogene()const {return zVector4D(x/w, y/w, z/w, 1.0f);}
	
    //members
    float x, y, z, w;
};


#endif
