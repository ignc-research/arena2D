#include "WandererBipedal.hpp"
#include "math.h"

WandererBipedal::WandererBipedal(b2World * w, const b2Vec2 & position,
					float velocity, float change_rate, float stop_rate, float max_angle_change, unsigned int type):
					Wanderer(w, position, velocity, change_rate, stop_rate, max_angle_change, type)
{
	float r = HUMAN_LEG_SIZE/2.f;
	float offset = HUMAN_LEG_DISTANCE/2.0f;
	addCircle(r, b2Vec2(offset+r, 0));
	addCircle(r, b2Vec2(-offset-r, 0));
}
