/* author: Cornelius Marx */
#include "zMatrix4x4.hpp"

zMatrix4x4::zMatrix4x4( float e0, float e1, float e2, float e3,
                        float e4, float e5, float e6, float e7,
                        float e8, float e9, float e10, float e11,
                        float e12, float e13, float e14, float e15)
{
	values[0] = e0;
	values[1] = e1;
	values[2] = e2;
	values[3] = e3;
	values[4] = e4;
	values[5] = e5;
	values[6] = e6;
	values[7] = e7;
	values[8] = e8;
	values[9] = e9;
	values[10] = e10;
	values[11] = e11;
	values[12] = e12;
	values[13] = e13;
	values[14] = e14;
	values[15] = e15;
}

zMatrix4x4::zMatrix4x4(const float * mat)
{
	memcpy(values, mat, sizeof(values));
}

zMatrix4x4::zMatrix4x4(const zMatrix4x4 & mat)
{
    memcpy(values, mat.values, sizeof(values));
}

zVector4D zMatrix4x4::getRow(int position) const
{
    return zVector4D(values[position], values[4 + position], values[8+position], values[12+position]);
}

zVector4D zMatrix4x4::getColumn(int position) const
{
	return zVector4D(values[position*4], values[position*4+1], values[position*4+2], values[position*4+3]);
}

void zMatrix4x4::loadIdentity()
{
	values[0]=1.0f; values[4]=0.0f; values[ 8]=0.0f; values[12]=0.0f;
	values[1]=0.0f; values[5]=1.0f; values[ 9]=0.0f; values[13]=0.0f;
	values[2]=0.0f; values[6]=0.0f; values[10]=1.0f; values[14]=0.0f;
	values[3]=0.0f; values[7]=0.0f; values[11]=0.0f; values[15]=1.0f;
}

void zMatrix4x4::set2DTransform(const zVector2D & pos, float uniform_scale){
	set2DTransform(pos, zVector2D(uniform_scale, uniform_scale));
}

void zMatrix4x4::set2DTransform(const zVector2D & pos, float uniform_scale, float rad){
	set2DTransform(pos, zVector2D(uniform_scale, uniform_scale), rad);
}

void zMatrix4x4::set2DTransform(const zVector2D & pos, const zVector2D & scale, float rad)
{
	float cos_rad = cos(rad);
	float sin_rad = sin(rad);
	values[0] = scale.x*cos_rad;
	values[1] = -sin_rad*scale.y;
	values[2] = 0;
	values[3] = 0;

	values[4] = sin_rad*scale.x;
	values[5] = scale.y*cos_rad;
	values[6] = 0;
	values[7] = 0;

	values[8] = 0;
	values[9] = 0;
	values[10] = 1;
	values[11] = 0;

	values[12] = pos.x;
	values[13] = pos.y;
	values[14] = 0;
	values[15] = 1;
}

void zMatrix4x4::set2DCameraTransform(const zVector2D & pos, float scale, float rad)
{
	float cos_rad = cos(rad);
	float sin_rad = sin(rad);
	zMatrix4x4 transform;
	values[0] = scale*cos_rad;
	values[1] = -scale*sin_rad;
	values[2] = 0;
	values[3] = 0;

	values[4] = scale*sin_rad;
	values[5] = scale*cos_rad;
	values[6] = 0;
	values[7] = 0;

	values[8] = 0;
	values[9] = 0;
	values[10] = 1;
	values[11] = 0;

	values[12] = -(pos.x*cos_rad+pos.y*sin_rad)*scale;
	values[13] = -(pos.y*cos_rad-pos.x*sin_rad)*scale;
	values[14] = 0;
	values[15] = 1;
}

void zMatrix4x4::setInverse2DCameraTransform(const zVector2D & pos, float scale, float rad)
{
	float cos_rad = cos(rad);
	float sin_rad = sin(rad);
	float fraction1 = cos_rad/scale;
	float fraction2 = sin_rad/scale;

	values[0] = fraction1;
	values[1] = fraction2;
	values[2] = 0;
	values[3] = 0;

	values[4] = fraction2;
	values[5] = fraction1;
	values[6] = 0;
	values[7] = 0;

	values[8] = 0;
	values[9] = 0;
	values[10] = 1;
	values[11] = 0;

	values[12] = pos.x;
	values[13] = pos.y;
	values[14] = 0;
	values[15] = 1;
}

void zMatrix4x4::set2DTransform(const zVector2D & pos, const zVector2D & scale)
{
	zMatrix4x4 transform;
	values[0] = scale.x;
	values[1] = 0;
	values[2] = 0;
	values[3] = 0;

	values[4] = 0;
	values[5] = scale.y;
	values[6] = 0;
	values[7] = 0;

	values[8] = 0;
	values[9] = 0;
	values[10] = 1;
	values[11] = 0;

	values[12] = pos.x;
	values[13] = pos.y;
	values[14] = 0;
	values[15] = 1;
}


zMatrix4x4 zMatrix4x4::operator+(const zMatrix4x4 & m) const		//overloaded operators
{
	
	return zMatrix4x4(	values[0]+m.values[0],
						values[1]+m.values[1],
						values[2]+m.values[2],
						values[3]+m.values[3],
						values[4]+m.values[4],
						values[5]+m.values[5],
						values[6]+m.values[6],
						values[7]+m.values[7],
						values[8]+m.values[8],
						values[9]+m.values[9],
						values[10]+m.values[10],
						values[11]+m.values[11],
						values[12]+m.values[12],
						values[13]+m.values[13],
						values[14]+m.values[14],
						values[15]+m.values[15]);
}

zMatrix4x4 zMatrix4x4::operator-(const zMatrix4x4 & m) const		//overloaded operators
{
	return zMatrix4x4(	values[0]-m.values[0],
						values[1]-m.values[1],
						values[2]-m.values[2],
						values[3]-m.values[3],
						values[4]-m.values[4],
						values[5]-m.values[5],
						values[6]-m.values[6],
						values[7]-m.values[7],
						values[8]-m.values[8],
						values[9]-m.values[9],
						values[10]-m.values[10],
						values[11]-m.values[11],
						values[12]-m.values[12],
						values[13]-m.values[13],
						values[14]-m.values[14],
						values[15]-m.values[15]);
}

zMatrix4x4 zMatrix4x4::operator*(const zMatrix4x4 & m) const
{
    //WARNING: Multiplication can be optimized for common cases (e.g. bottom row = {0,0,0,1})
	return zMatrix4x4(	values[0]*m.values[0]+values[4]*m.values[1]+values[8]*m.values[2]+values[12]*m.values[3],
						values[1]*m.values[0]+values[5]*m.values[1]+values[9]*m.values[2]+values[13]*m.values[3],
						values[2]*m.values[0]+values[6]*m.values[1]+values[10]*m.values[2]+values[14]*m.values[3],
						values[3]*m.values[0]+values[7]*m.values[1]+values[11]*m.values[2]+values[15]*m.values[3],
						values[0]*m.values[4]+values[4]*m.values[5]+values[8]*m.values[6]+values[12]*m.values[7],
						values[1]*m.values[4]+values[5]*m.values[5]+values[9]*m.values[6]+values[13]*m.values[7],
						values[2]*m.values[4]+values[6]*m.values[5]+values[10]*m.values[6]+values[14]*m.values[7],
						values[3]*m.values[4]+values[7]*m.values[5]+values[11]*m.values[6]+values[15]*m.values[7],
						values[0]*m.values[8]+values[4]*m.values[9]+values[8]*m.values[10]+values[12]*m.values[11],
						values[1]*m.values[8]+values[5]*m.values[9]+values[9]*m.values[10]+values[13]*m.values[11],
						values[2]*m.values[8]+values[6]*m.values[9]+values[10]*m.values[10]+values[14]*m.values[11],
						values[3]*m.values[8]+values[7]*m.values[9]+values[11]*m.values[10]+values[15]*m.values[11],
						values[0]*m.values[12]+values[4]*m.values[13]+values[8]*m.values[14]+values[12]*m.values[15],
						values[1]*m.values[12]+values[5]*m.values[13]+values[9]*m.values[14]+values[13]*m.values[15],
						values[2]*m.values[12]+values[6]*m.values[13]+values[10]*m.values[14]+values[14]*m.values[15],
						values[3]*m.values[12]+values[7]*m.values[13]+values[11]*m.values[14]+values[15]*m.values[15]);
}

void zMatrix4x4::multLeft(const zMatrix4x4 & m)
{
    (*this) = m * (*this);
}

void zMatrix4x4::multRight(const zMatrix4x4 & m)
{
    (*this) = (*this) * m;
}

zMatrix4x4 zMatrix4x4::operator*(const float s) const
{
	return zMatrix4x4(	values[0]*s,
						values[1]*s,
						values[2]*s,
						values[3]*s,
						values[4]*s,
						values[5]*s,
						values[6]*s,
						values[7]*s,
						values[8]*s,
						values[9]*s,
						values[10]*s,
						values[11]*s,
						values[12]*s,
						values[13]*s,
						values[14]*s,
						values[15]*s);
}

zMatrix4x4 zMatrix4x4::operator/ (const float m) const
{
	if (m==0.0f || m==1.0f)
		return (*this);
		
	float temp=1/m;

	return (*this)*temp;
}

zMatrix4x4 operator*(float scaleFactor, const zMatrix4x4 & m)
{
	return m*scaleFactor;
}

bool zMatrix4x4::operator==(const zMatrix4x4 & m) const
{
	for(int i=0; i<16; i++)
	{
		if(values[i]!=m.values[i])
			return false;
	}
	return true;
}

bool zMatrix4x4::operator!=(const zMatrix4x4 & m) const
{
	return !((*this)==m);
}

void zMatrix4x4::operator+=(const zMatrix4x4 & m)
{
	(*this)=(*this)+m;
}

void zMatrix4x4::operator-=(const zMatrix4x4 & m)
{
	(*this)=(*this)-m;
}

void zMatrix4x4::operator*=(const zMatrix4x4 & m)
{
	(*this)=(*this)*m;
}

void zMatrix4x4::operator*=(const float s)
{
	(*this)=(*this)*s;
}

void zMatrix4x4::operator/=(const float s)
{
	(*this)=(*this)/s;
}

zMatrix4x4 zMatrix4x4::operator-() const
{
	zMatrix4x4 result(*this);

	for(int i=0; i<16; i++)
		result.values[i]=-result.values[i];

	return result;
}

zVector4D zMatrix4x4::operator*(const zVector4D & v) const
{
	if(values[3]==0.0f && values[7]==0.0f && values[11]==0.0f && values[15]==1.0f)
	{
		return zVector4D(values[0]*v.x
					+	values[4]*v.y
					+	values[8]*v.z
					+	values[12]*v.w,

						values[1]*v.x
					+	values[5]*v.y
					+	values[9]*v.z
					+	values[13]*v.w,

						values[2]*v.x
					+	values[6]*v.y
					+	values[10]*v.z
					+	values[14]*v.w,

						v.w);
	}
	
	return zVector4D(	values[0]*v.x
					+	values[4]*v.y
					+	values[8]*v.z
					+	values[12]*v.w,

						values[1]*v.x
					+	values[5]*v.y
					+	values[9]*v.z
					+	values[13]*v.w,

						values[2]*v.x
					+	values[6]*v.y
					+	values[10]*v.z
					+	values[14]*v.w,

						values[3]*v.x
					+	values[7]*v.y
					+	values[11]*v.z
					+	values[15]*v.w);
}

void zMatrix4x4::transpose()
{
	(*this)=getTranspose();
}

zMatrix4x4 zMatrix4x4::getTranspose() const
{
	zMatrix4x4 m;
	m.values[0] = values[0];
	m.values[1] = values[4];
	m.values[2] = values[8];
	m.values[3] = values[12];
	m.values[4] = values[1];
	m.values[5] = values[5];
	m.values[6] = values[9];
	m.values[7] = values[13];
	m.values[8] = values[2];
	m.values[9] = values[6];
	m.values[10] = values[10];
	m.values[11] = values[14];
	m.values[12] = values[3];
	m.values[13] = values[7];
	m.values[14] = values[11];
	m.values[15] = values[15];
	return m;
}

zVector3D zMatrix4x4::rotateVector3D(const zVector3D & v) const
{
	return zVector3D(values[0]*v.x + values[4]*v.y + values[8]*v.z,
					values[1]*v.x + values[5]*v.y + values[9]*v.z,
					values[2]*v.x + values[6]*v.y + values[10]*v.z);
}

zVector3D zMatrix4x4::translateVector3D(const zVector3D & v) const
{
	return zVector3D(v.x+values[12], v.y+values[13], v.z+values[14]);
}

void zMatrix4x4::setTranslation(const zVector3D & translation)
{
	loadIdentity();
	values[12] =translation.x;
	values[13] =translation.y;
	values[14] =translation.z;
}

void zMatrix4x4::setScale(const zVector3D & scaleFactor)
{
	loadIdentity();

	values[0]=scaleFactor.x;
	values[5]=scaleFactor.y;
	values[10]=scaleFactor.z;
}

void zMatrix4x4::setScale(float s)
{
	loadIdentity();

	values[0]=s;
	values[5]=s;
	values[10]=s;
}

void zMatrix4x4::setRotationAxis(const double angle, const zVector3D & axis)
{
	zVector3D u=axis.getNormalized();

	float sinAngle=static_cast<float>(sin(M_PI*angle/180));
	float cosAngle=static_cast<float>(cos(M_PI*angle/180));
	float oneMinusCosAngle=1.0f-cosAngle;

	loadIdentity();

	values[0]=(u.x)*(u.x) + cosAngle*(1-(u.x)*(u.x));
	values[4]=(u.x)*(u.y)*(oneMinusCosAngle) - sinAngle*u.z;
	values[8]=(u.x)*(u.z)*(oneMinusCosAngle) + sinAngle*u.y;

	values[1]=(u.x)*(u.y)*(oneMinusCosAngle) + sinAngle*u.z;
	values[5]=(u.y)*(u.y) + cosAngle*(1-(u.y)*(u.y));
	values[9]=(u.y)*(u.z)*(oneMinusCosAngle) - sinAngle*u.x;
	
	values[2]=(u.x)*(u.z)*(oneMinusCosAngle) - sinAngle*u.y;
	values[6]=(u.y)*(u.z)*(oneMinusCosAngle) + sinAngle*u.x;
	values[10]=(u.z)*(u.z) + cosAngle*(1-(u.z)*(u.z));
}

void zMatrix4x4::setRotationX(const double angle)
{
	loadIdentity();

	values[5]=(float)cos(M_PI*angle/180);
	values[6]=(float)sin(M_PI*angle/180);

	values[9]=-values[6];
	values[10]=values[5];
}

void zMatrix4x4::setRotationY(const double angle)
{
	loadIdentity();

	values[0]=(float)cos(M_PI*angle/180);
	values[2]=-(float)sin(M_PI*angle/180);

	values[8]=-values[2];
	values[10]=values[0];
}

void zMatrix4x4::setRotationZ(const double angle)
{
	loadIdentity();

	values[0]=(float)cos(M_PI*angle/180);
	values[1]=(float)sin(M_PI*angle/180);

	values[4]=-values[1];
	values[5]=values[0];
}

void zMatrix4x4::setPerspective(float left, float right, float bottom,
								float top, float n, float f)
{
	values[0]  = (2*n)/(right-left);
    values[1]  = 0.f;
    values[2]  = 0.f;
    values[3]  = 0.f;
    values[4]  = 0.f;
	values[5]  = (2*n)/(top-bottom);
    values[6]  = 0.f;
    values[7]  = 0.f;
	values[8]  = (right+left)/(right-left);
	values[9]  = (top+bottom)/(top-bottom);
    values[10] =-(f+n)/(f-n);
	values[11] =-1.f;
    values[12] = 0.f;
    values[13] = 0.f;
    values[14] =-(2*f*n)/(f-n);
    values[15] = 0.f;
}

void zMatrix4x4::setPerspectiveY(float fovy, float aspect, float n, float f)
{
	float deltaZ = n - f;
	float a= 1.f/tan((M_PI * fovy)/(360.f));
	
	values[0] = a/aspect;
	values[1] = 0.f;
	values[2] = 0.f;
	values[3] = 0.f;
	
	values[4] = 0.f;
	values[5] = a;
	values[6] = 0.f;
	values[7] = 0.f;
	
	values[8] = 0.f;	
	values[9] = 0.f;	
	values[10] = (n + f)/deltaZ;	
	values[11] = -1.f;	

	values[12]  = 0.f;
	values[13]  = 0.f;
	values[14]  = (2*n*f)/deltaZ;
	values[15]  = 0.f;
}

void zMatrix4x4::setPerspectiveX(float fovx, float aspect, float n, float f)
{
    float deltaZ = n - f;
	float a =1.f/tan((M_PI * fovx)/(360.f));
	values[0] = a;
	values[1] = 0.f;
	values[2] = 0.f;
	values[3] = 0.f;

	values[4] = 0;
	values[5] = a/aspect;
	values[6] = 0.f;
	values[7] = 0.f;

	values[8] = 0.f;	
	values[9] = 0.f;	
	values[10] = (n + f)/deltaZ;	
	values[11] = -1.f;	

	values[12]  = 0.f;
	values[13]  = 0.f;
	values[14]  = (2*n*f)/deltaZ;
	values[15]  = 0.f;
}

void zMatrix4x4::setOrtho(	float left, float right, float bottom,
							float top, float n, float f)
{
	values[0]  = 2.0f/(right-left);
    values[1]  = 0.f;
    values[2]  = 0.f;
    values[3]  = 0.f;
    values[4]  = 0.f;
	values[5]  = 2.0f/(top-bottom);
    values[6]  = 0.f;
	values[7]  = 0.f;
    values[8]  = 0.f;
    values[9]  = 0.f;
    values[10] =-2.f/(f-n);
    values[11] = 0.f;
	values[12] =-(right+left)/(right-left);
	values[13] =-(top+bottom)/(top-bottom);
	values[14] =-(f+n)/(f-n);
    values[15] = 1.f;
}

void zMatrix4x4::invert()
{
	*this=getInverse();
}

zMatrix4x4 zMatrix4x4::getInverse() const
{
	zMatrix4x4 result=getInverseTranspose();

	result.transpose();

	return result;
}

void zMatrix4x4::invertTranspose()
{
	*this=getInverseTranspose();
}

zMatrix4x4 zMatrix4x4::getInverseTranspose() const
{
	zMatrix4x4 result;

	float tmp[12];												//temporary pair storage
	float det;													//determinant

	//calculate pairs for first 8 elements (cofactors)
	tmp[0] = values[10] * values[15];
	tmp[1] = values[11] * values[14];
	tmp[2] = values[9] * values[15];
	tmp[3] = values[11] * values[13];
	tmp[4] = values[9] * values[14];
	tmp[5] = values[10] * values[13];
	tmp[6] = values[8] * values[15];
	tmp[7] = values[11] * values[12];
	tmp[8] = values[8] * values[14];
	tmp[9] = values[10] * values[12];
	tmp[10] = values[8] * values[13];
	tmp[11] = values[9] * values[12];

	//calculate first 8 elements (cofactors)
	result.values[0]=		tmp[0]*values[5] + tmp[3]*values[6] + tmp[4]*values[7]
					-	tmp[1]*values[5] - tmp[2]*values[6] - tmp[5]*values[7];

	result.values[1]=		tmp[1]*values[4] + tmp[6]*values[6] + tmp[9]*values[7]
					-	tmp[0]*values[4] - tmp[7]*values[6] - tmp[8]*values[7];

	result.values[2]=		tmp[2]*values[4] + tmp[7]*values[5] + tmp[10]*values[7]
					-	tmp[3]*values[4] - tmp[6]*values[5] - tmp[11]*values[7];

	result.values[3]=		tmp[5]*values[4] + tmp[8]*values[5] + tmp[11]*values[6]
					-	tmp[4]*values[4] - tmp[9]*values[5] - tmp[10]*values[6];

	result.values[4]=		tmp[1]*values[1] + tmp[2]*values[2] + tmp[5]*values[3]
					-	tmp[0]*values[1] - tmp[3]*values[2] - tmp[4]*values[3];

	result.values[5]=		tmp[0]*values[0] + tmp[7]*values[2] + tmp[8]*values[3]
					-	tmp[1]*values[0] - tmp[6]*values[2] - tmp[9]*values[3];

	result.values[6]=		tmp[3]*values[0] + tmp[6]*values[1] + tmp[11]*values[3]
					-	tmp[2]*values[0] - tmp[7]*values[1] - tmp[10]*values[3];

	result.values[7]=		tmp[4]*values[0] + tmp[9]*values[1] + tmp[10]*values[2]
					-	tmp[5]*values[0] - tmp[8]*values[1] - tmp[11]*values[2];

	//calculate pairs for second 8 elements (cofactors)
	tmp[0] = values[2]*values[7];
	tmp[1] = values[3]*values[6];
	tmp[2] = values[1]*values[7];
	tmp[3] = values[3]*values[5];
	tmp[4] = values[1]*values[6];
	tmp[5] = values[2]*values[5];
	tmp[6] = values[0]*values[7];
	tmp[7] = values[3]*values[4];
	tmp[8] = values[0]*values[6];
	tmp[9] = values[2]*values[4];
	tmp[10] = values[0]*values[5];
	tmp[11] = values[1]*values[4];

	//calculate second 8 elements (cofactors)
	result.values[8]=		tmp[0]*values[13] + tmp[3]*values[14] + tmp[4]*values[15]
					-	tmp[1]*values[13] - tmp[2]*values[14] - tmp[5]*values[15];

	result.values[9]=		tmp[1]*values[12] + tmp[6]*values[14] + tmp[9]*values[15]
					-	tmp[0]*values[12] - tmp[7]*values[14] - tmp[8]*values[15];

	result.values[10]=		tmp[2]*values[12] + tmp[7]*values[13] + tmp[10]*values[15]
					-	tmp[3]*values[12] - tmp[6]*values[13] - tmp[11]*values[15];

	result.values[11]=		tmp[5]*values[12] + tmp[8]*values[13] + tmp[11]*values[14]
					-	tmp[4]*values[12] - tmp[9]*values[13] - tmp[10]*values[14];

	result.values[12]=		tmp[2]*values[10] + tmp[5]*values[11] + tmp[1]*values[9]
					-	tmp[4]*values[11] - tmp[0]*values[9] - tmp[3]*values[10];

	result.values[13]=		tmp[8]*values[11] + tmp[0]*values[8] + tmp[7]*values[10]
					-	tmp[6]*values[10] - tmp[9]*values[11] - tmp[1]*values[8];

	result.values[14]=		tmp[6]*values[9] + tmp[11]*values[11] + tmp[3]*values[8]
					-	tmp[10]*values[11] - tmp[2]*values[8] - tmp[7]*values[9];

	result.values[15]=		tmp[10]*values[10] + tmp[4]*values[8] + tmp[9]*values[9]
					-	tmp[8]*values[9] - tmp[11]*values[10] - tmp[5]*values[8];

	// calculate determinant
	det	=	 values[0]*result.values[0]
			+values[1]*result.values[1]
			+values[2]*result.values[2]
			+values[3]*result.values[3];

	if(det==0.0f)//Matrix is not invertible
	{
		zMatrix4x4 id;
		return id;
	}
	
	result=result/det;

	return result;
}

//Invert if only composed of rotations & translations
void zMatrix4x4::affineInvert()
{
	(*this)=getAffineInverse();
}

zMatrix4x4 zMatrix4x4::getAffineInverse() const
{
	//return the transpose of the rotation part
	//and the negative of the inverse rotated translation part
	return zMatrix4x4(	values[0],
						values[4],
						values[8],
						0.0f,
						values[1],
						values[5],
						values[9],
						0.0f,
						values[2],
						values[6],
						values[10],
						0.0f,
						-(values[0]*values[12]+values[1]*values[13]+values[2]*values[14]),
						-(values[4]*values[12]+values[5]*values[13]+values[6]*values[14]),
						-(values[8]*values[12]+values[9]*values[13]+values[10]*values[14]),
						1.0f);
}

void zMatrix4x4::affineInvertTranspose()
{
	(*this)=getAffineInverseTranspose();
}

zMatrix4x4 zMatrix4x4::getAffineInverseTranspose() const
{
	//return the transpose of the rotation part
	//and the negative of the inverse rotated translation part
	//transposed
	return zMatrix4x4(	values[0],
						values[1],
						values[2],
						-(values[0]*values[12]+values[1]*values[13]+values[2]*values[14]),
						values[4],
						values[5],
						values[6],
						-(values[4]*values[12]+values[5]*values[13]+values[6]*values[14]),
						values[8],
						values[9],
						values[10],
						-(values[8]*values[12]+values[9]*values[13]+values[10]*values[14]),
						0.0f, 0.0f, 0.0f, 1.0f);
}
