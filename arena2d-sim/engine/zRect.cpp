/* author: Cornelius Marx */
#include "zRect.hpp"

bool zRect::checkPoint(const zVector2D & point)const
{
	return point.x <= x+w && point.x >= x-w && point.y <= y+h && point.y >= y-h;
}

bool zRect::contains(const zRect & r, float epsilon)const
{
	return r.x+r.w <= x+w+epsilon && r.x-r.w >= x-w-epsilon &&
			r.y+r.h <= y+h+epsilon && r.y-r.h >= y-h-epsilon;
}


bool zRect::intersect(const zRect & a, const zRect & b, zRect * intersection_rect){
	if(a.x+a.w > b.x-b.w && a.x-a.w < b.x+b.w &&
	   a.y+a.h > b.y-b.h && a.y-a.h < b.y+b.h){
		if(intersection_rect != NULL){
			float left = f_fmax(a.x-a.w, b.x-b.w);
			float right = f_fmin(a.x+a.w, b.x+b.w);
			float down = f_fmax(a.y-a.h, b.y-b.h);
			float up = f_fmin(a.y+a.h, b.y+b.h);
			intersection_rect->set((right+left)/2.f, (up+down)/2.f, (right-left)/2.f, (up-down)/2.f);
		}
		return true;
	}
	else{
		return false;
	}
}
