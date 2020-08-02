//Created: 15th Jan 2017
//Author: Cornelius Marx

#ifndef ZER0_VECTOR3D_H
#define ZER0_VECTOR3D_H

#include <math.h>

class zVector3D
{
public:

	//constructors
	zVector3D() : x(0.f), y(0.f), z(0.f){}
	zVector3D(float _x, float _y, float _z) : x(_x), y(_y), z(_z){}
	zVector3D(const float * v) : x(v[0]), y(v[1]), z(v[2]){}
	zVector3D(const zVector3D & v) : x(v.x), y(v.y), z(v.z){}

	//destructor
	~zVector3D(){}

	//initializers
	void loadZero(){x = 0.f; y = 0.f; z = 0.f;}
	void loadOne(){x = 1.f; y = 1.f; z = 1.f;}
	void set(float new_x, float new_y, float new_z){x = new_x; y = new_y; z = new_z;}

	//calculations
	void normalize();
	zVector3D getNormalized() const;

	float getLength() const
	{return sqrt(x*x + y*y + z*z);}

	float squaredLength() const
	{return x*x + y*y + z*z;}
	
	//applying signumfunction to every coordinate
	zVector3D getSign()const;

	//overloaded operators
	zVector3D operator+(const zVector3D & rhs) const
	{return zVector3D(x + rhs.x, y + rhs.y, z + rhs.z);	}

	zVector3D operator-(const zVector3D & rhs) const
	{return zVector3D(x - rhs.x, y - rhs.y, z - rhs.z);	}

	zVector3D operator*(const float rhs) const
	{return zVector3D(x*rhs, y*rhs, z*rhs);	}
	
	zVector3D operator/(const float rhs) const
	{return (rhs==0) ? zVector3D(0.0f, 0.0f, 0.f) : zVector3D(x / rhs, y / rhs, z / rhs); }

	//allow operations like: 3*v
	friend zVector3D operator*(float scale, const zVector3D & rhs)
	{return zVector3D(rhs.x * scale, rhs.y * scale, scale * rhs.z);}

	bool operator==(const zVector3D & rhs) const
	{return (x == rhs.x && y == rhs.y && z == rhs.z);}

	bool operator!=(const zVector3D & rhs) const
	{return (x != rhs.x || y != rhs.y || z != rhs.z);}

	void operator+=(const zVector3D & rhs)
	{x+=rhs.x; y+=rhs.y; z+=rhs.z;}

	void operator-=(const zVector3D & rhs)
	{x-=rhs.x; y-=rhs.y; z-=rhs.z;}

	void operator*=(const float rhs)
	{x*=rhs; y*=rhs; z*=rhs;}

	void operator/=(const float rhs)
	{	if(rhs==0.0f)
			return;
		else
		{x/=rhs; y/=rhs;}
	}

	zVector3D operator-() const {return zVector3D(-x, -y, -z);}
	zVector3D operator+() const {return *this;}

	//cast to pointer
	operator float* () const {return (float*) this;}

	//members
	float x, y, z;
};

#endif
