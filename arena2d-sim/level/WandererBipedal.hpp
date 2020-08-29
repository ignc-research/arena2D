#ifndef WANDERERBIPEDAL_H
#define WANDERERBIPEDAL_H

#include "Wanderer.hpp"
#include <arena/PhysicsWorld.hpp>
#include <engine/zVector2d.hpp>
#include <engine/GlobalSettings.hpp>
#include <ctime>

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

    /* updates velocity accoriding to change/stop rate
     */
    virtual void update();

    /* reset position of wanderer, velocities are set to 0
     * @param position new position of wanderer
     */
    virtual void reset(const b2Vec2 & position);

    /* get position
     * @return position of wanderer
     */
    const b2Vec2& getPosition1(){return _body1->GetTransform().p;}
    const b2Vec2& getPosition2(){return _body2->GetTransform().p;}

    /* get type
     * @return type that was specified on creation
     */
    unsigned int getType(){return _type;}

    /* get Box2D body
     * @return body
     */
    b2Body* getBody(){return _body1;}


    /* get radius
     * @return radius of the circular body
     */
    float getRadius(){return _radius;}

protected:
    /* update velocity
     * this function is called in update() if a randomly sampled value [0, 1] is less than the change rate
     */
    void updateVelocity();

    /* Box2D body */
    b2Body * _body1;
    b2Body * _body2;

    /* constant velocity with which to move if not stopping */
    float _velocity;

    /* [0, 1] how often to change velocity */
    float _changeRate;

    /* [0, 1] how likely velocity is set to 0 on change */
    float _stopRate;

    /* body radius */
    float _radius;

    /* wanderer type (specified by user) */
    unsigned int _type;

    float _counter;

    zVector2D _lastVRot;

    std::time_t last_time;

    static void updateFixtureTask(b2Body *body);

    // Legposture parameter
    double step_frequency_factor;
    double step_width_factor;

};

#endif
