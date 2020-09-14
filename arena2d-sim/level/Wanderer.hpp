#ifndef WANDERER_H
#define WANDERER_H

#include <arena/PhysicsWorld.hpp>
#include <engine/zVector2d.hpp>


/* A Wanderer is a dynamic obstacle that 'wanders' around randomly at a constant speed
 * The userData field of the Box2D body of an object of the Wanderer class will be set to 'this',
 * which may be used to retrieve the corresponding Wanderer-object when checking for collisions etc.
 */
class Wanderer{
public:
	/* constructor
	 * @param w the Box2D world to spawn wanderer in
	 * @param radius the radius of the circular body
	 * @param position spawn position
	 * @param velocity the velocity with which to move
	 * @param change_rate probabililty of changing the velocity on a call of update()
	 * @param stop_rate probability of stopping when changing velocity (-> change_rate*stop_rate is the probability of stopping on a call of update())
	 * @param max_angle_velo maximum +/- angle velocity in deg when changing direction
	 * @param type user defined object type to identify wanderer later
	 */
	Wanderer(b2World * w, const b2Vec2 & position, float velocity,
				float change_rate, float stop_rate, float max_angle_velo = 30.0f, unsigned int type = 0);

	/* destructor, removes body from world
	 */
	virtual ~Wanderer();

	/* updates velocity according to change/stop rate
	 */
	virtual void update(bool chat_flag);

	/* reset position of wanderer, velocities are set to 0
	 * @param position new position of wanderer
	 */
	virtual void reset(const b2Vec2 & position);

	/* get position
	 * @return position of wanderer
	 */
	const b2Vec2& getPosition(){return _body->GetTransform().p;}


	/* set position
     * @param newPosition of wanderer
     */
	void setPosition(b2Vec2 newPosition){
	    _body->SetTransform(newPosition, 0);
	}

	/* get type
	 * @return type that was specified on creation
	 */
	unsigned int getType(){return _type;}

	/* get Box2D body
	 * @return body
	 */
	b2Body* getBody(){return _body;}

	/* add circle fixture to physics body
	 * @param pos relative position of circle center to body origin
	 * @param radius radius of circle
	 */
	void addCircle(float radius, const b2Vec2 & pos = b2Vec2_zero);
protected:

	/* update velocity
	 * this function is called in update() if a randomly sampled value [0, 1] is less than the change rate
	 */
	virtual void updateVelocity();
	
	/* Box2D body */
	b2Body * _body;

	/* constant velocity with which to move if not stopping */
	float _velocity;

	/* [0, 1] how often to change velocity */
	float _changeRate;

	/* */
	float _maxAngleVel;

	/* [0, 1] how likely velocity is set to 0 on change */
	float _stopRate;

	/* wanderer type (specified by user) */
	unsigned int _type;
};

#endif
