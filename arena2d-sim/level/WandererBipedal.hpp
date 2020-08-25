#ifndef WANDERERBIPEDAL_H
#define WANDERERBIPEDAL_H

#include "Wanderer.hpp"

#define HUMAN_LEG_SIZE 0.09f
#define HUMAN_LEG_DISTANCE 0.04f

/* human wanderer represented by two circles (legs)
 */
class WandererBipedal : public Wanderer{
public:
	/* constructor */
    WandererBipedal(b2World * w,const b2Vec2 & position, float velocity,
					float change_rate, float stop_rate, float max_angle_change = 60.0f, unsigned int type = 0);

	/* destructor */
	~WandererBipedal(){}

	/* return radius of circle surrounding all fixtures
	 */
	static float getRadius(){return (HUMAN_LEG_DISTANCE+2*HUMAN_LEG_SIZE)/2.f;}
};

#endif
