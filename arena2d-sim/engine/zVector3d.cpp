/* author: Cornelius Marx */
#include "zVector3d.hpp"

void zVector3D::normalize()
{
	*this = getNormalized();
}

zVector3D zVector3D::getNormalized() const
{
	return *this/getLength();
}

zVector3D zVector3D::getSign() const 
{
	zVector3D s;
	if(x > 0.f)
		s.x = 1.f;
	else if(x < 0.f)
		s.x = -1.f;
	else
		s.x = 0.f;

	if(y > 0.f)
		s.y = 1.f;
	else if(y < 0.f)
		s.y = -1.f;
	else
		s.y = 0.f;

	if(z > 0.f)
		s.z = 1.f;
	else if(z < 0.f)
		s.z = -1.f;
	else
		s.z = 0.f;

	return s;
}
