/*
    Author: Cornelius Marx
    description:
        - providing 4x4 Matrix 
        - values are in such order that the translation part takes up the 12th, 13th, 14th and 15th index (similar to openGL's order)
*/

#ifndef Z_MATRIX4X4_H
#define Z_MATRIX4X4_H

#include "zVector2d.hpp"
#include "zVector3d.hpp"
#include "zVector4d.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h> //needed for memset

class zMatrix4x4
{
public:
	//constructors
	zMatrix4x4(){}
	zMatrix4x4 (float e0, float e1, float e2, float e3,
				float e4, float e5, float e6, float e7,
				float e8, float e9, float e10, float e11,
				float e12, float e13, float e14, float e15);

	zMatrix4x4(const float * mat);
	zMatrix4x4(const zMatrix4x4 & mat);

	//destructor
	~zMatrix4x4(){}

	//access data
	zVector4D getRow(int row) const;
	zVector4D getColumn(int col) const;

	//load identity matrix
	void loadIdentity();

	//operators
	zMatrix4x4 operator+(const zMatrix4x4 & m) const;
	zMatrix4x4 operator-(const zMatrix4x4 & m) const;
	zMatrix4x4 operator*(const zMatrix4x4 & m) const;
	zMatrix4x4 operator*(const float m) const;
	zMatrix4x4 operator/(const float m) const;
    void multLeft(const zMatrix4x4 & m);//multiply a matrix from left: this=m*this
    void multRight(const zMatrix4x4 & m);//multiply a matrix from right: this=this*m

	bool operator==(const zMatrix4x4 & m) const;
	bool operator!=(const zMatrix4x4 & m) const;
	void operator+=(const zMatrix4x4 & m);
	void operator-=(const zMatrix4x4 & m);
	void operator*=(const zMatrix4x4 & m);
	void operator*=(const float m);
	void operator/=(const float m);

	zMatrix4x4 operator-() const;
	zMatrix4x4 operator+() const {return (*this);}
	
	operator float* () {return (float*)this;}
	operator const float*() const {return (const float*)this;}
	
	//transform a 4D vector
	zVector4D operator* (const zVector4D & v) const;

	//transform a 3D vector
	zVector3D translateVector3D(const zVector3D & v) const;
	zVector3D rotateVector3D(const zVector3D & v) const;

    //transpose matrix
	void transpose();
	zMatrix4x4 getTranspose() const;

    //calculating the inverse
    void invert();
	zMatrix4x4 getInverse() const;
	void invertTranspose();
	zMatrix4x4 getInverseTranspose() const;
    
    //calculating inverse assuming that the M-part (non-translational) is an orthogonal matrix
	// NOTE: the following functions appear to be implemented incorrectly, DO NOT USE!
    void affineInvert();
	zMatrix4x4 getAffineInverse() const;
	void affineInvertTranspose();
	zMatrix4x4 getAffineInverseTranspose() const;

	void set2DTransform(const zVector2D & pos, const zVector2D & scale);
	void set2DTransform(const zVector2D & pos, float uniform_scale);
	void set2DTransform(const zVector2D & pos, float uniform_scale, float rad);
	void set2DTransform(const zVector2D & pos, const zVector2D & scale, float rad);
	void set2DCameraTransform(const zVector2D & pos, float scale, float rad);
	void setInverse2DCameraTransform(const zVector2D & pos, float scale, float rad);
    
	void setTranslation(const zVector3D & translation);
	void setScale(const zVector3D & scaleFactor);
	void setScale(const float scaleFactor);//uniform scaling
	void setRotationAxis(const double angle, const zVector3D & axis);
	void setRotationX(const double angle);
	void setRotationY(const double angle);
	void setRotationZ(const double angle);
	void setPerspective(float left, float right, float bottom, float top, float n, float f);
	void setPerspectiveY(float fovy, float aspect, float n, float f);
    void setPerspectiveX(float fovx, float aspect, float n, float f);
	void setOrtho(float left, float right, float bottom, float top, float n, float f);

	//members
	float values[16];
};


#endif
