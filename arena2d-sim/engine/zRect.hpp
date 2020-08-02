/* author: Cornelius Marx */
#ifndef Z_RECT_H
#define Z_RECT_H

#include "zVector2d.hpp"
#include "f_math.h"

/* axis aligned rect, w and h are mostly used as half dimensions */
struct zRect
{
	/* construct from each component
	 */
	zRect(float r_x, float r_y, float r_w, float r_h): x(r_x), y(r_y), w(r_w), h(r_h){}

	/* construct from vectors
	 */
	zRect(const zVector2D & pos, const zVector2D & dim): x(pos.x), y(pos.y), w(dim.x), h(dim.y){}

	/* empty constructor, initialize with zero
	 */
	zRect():x(0), y(0), w(0), h(0){}

	/* set each component
	 */
	void set(float r_x, float r_y, float r_w, float r_h){x = r_x; y = r_y; w = r_w; h = r_h;}

	/* set from vectors
	 */
	void set(const zVector2D & pos, const zVector2D & dim){x=pos.x; y=pos.y; w=dim.x; h=dim.y;}

	/* set only x and y from vector
	 */
	void setPos(const zVector2D & p){x = p.x; y = p.y;}

	/* set only dimension from vector
	 */
	void setDim(const zVector2D & d){w = d.x; h = d.y;}

	/* check whether a given point lies within the rect
	 * @param point the point to test
	 * @return true if point lies inside the rect (or on the very edge of it), else false
	 */
	bool checkPoint(const zVector2D & point)const;

	/* check whether two rects a and b intersect each other
	 * @param a rect a
	 * @param b rect b
	 * @param intersection_rect if not NULL and intersection occurs this rect is set to the resulting rectangular intersection of a and b
	 * @return true if a and b intersect
	 */
	static bool intersect(const zRect & a, const zRect & b, zRect * intersection_rect = NULL);

	/* returns true if a given rect lies inside this rect fully
	 */
	bool contains(const zRect & r, float epsilon = 0.f)const;

	/* x position of center */
	float x;

	/* y position of center */
	float y;

	/* half width (size along x) */
	float w;

	/* half height (size along y) */
	float h;
};

#endif
