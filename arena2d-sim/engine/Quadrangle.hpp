/* author: Cornelius Marx */
#ifndef QUADRANGLE_H
#define QUADRANGLE_H

#include "zVector2d.hpp"
#include "zRect.hpp"

//4 vertices representing a quadrangle
struct Quadrangle
{
	Quadrangle(){};
	Quadrangle(const zVector2D & _v0, const zVector2D & _v1, const zVector2D & _v2, const zVector2D & _v3)
	{
		v0 = _v0;
		v1 = _v1;
		v2 = _v2;
		v3 = _v3;
	}

	void getAABB(zRect * aabb)
	{
		//finding minimum and maximum coordinates
		float max_x = v0.x,
			  max_y = v0.y,
			  min_x = v0.x,
			  min_y = v0.y;
		zVector2D * v = (zVector2D*)this;
		for(int i = 1; i < 4; i++)
		{
			if(v[i].x > max_x)
				max_x = v[i].x;
			else if(v[i].x < min_x)
				min_x = v[i].x;

			if(v[i].y > max_y)
				max_y = v[i].y;
			else if(v[i].y < min_y)
				min_y = v[i].y;
		}
		aabb->x = (min_x + max_x)/2.f;
		aabb->y = (min_y + max_y)/2.f;
		aabb->w = (max_x - min_x)/2.f;
		aabb->h = (max_y - min_y)/2.f;
	}

	zVector2D v0, v1, v2, v3;
};

#endif
