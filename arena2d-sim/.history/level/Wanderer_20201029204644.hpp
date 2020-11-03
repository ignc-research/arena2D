#ifndef WANDERER_H
#define WANDERER_H

#include <arena/PhysicsWorld.hpp>
#include <engine/zVector2d.hpp>
#include <vector>

#define NEAR_REGION_DISTANCE 0.05f

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
	Wanderer(b2World * w, const b2Vec2 & position, float velocity, unsigned int type, unsigned int mode=0,
			std::vector<b2Vec2> waypoints={}, int stop_counter_threshold=2, float change_rate=1.0f, float stop_rate=0.1f, float max_angle_velo = 30.0f);

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
	virtual void updateVelocityRandomMode();

	virtual void updateVelocityPathMode();
	
	/* Box2D body */
	b2Body * _body;

	/* Initial poistion of the wanderer*/
	b2Vec2 _initPosition;

	/* constant velocity with which to move if not stopping */
	float _velocity;

	/* wanderer type (specified by user) */
	unsigned int _type;  // 0: human  1: dynamic obstacle(circle, polygon)

	unsigned int _mode;  // 0: radom  1: follow path


	// Path follow *******************************************
	/* Waypoints poistion of the wanderer*/
	std::vector<b2Vec2> _waypoints;

	/*index of way Point*/
	int _indexWaypoint;

	bool _directForward;

	/* Stop counter threshold */
	int _stopCounterThreshold;

	/* Counter for counting stop time*/
	int _stopCounter;

	/* Counter for counting trail times*/
	int _timeOutCounter;

	// Random move *******************************************
	/* [0, 1] how often to change velocity */
	float _changeRate;
	
	/* max Angle Velocity */
	float _maxAngleVel;

	/* [0, 1] how likely velocity is set to 0 on change */
	float _stopRate;

	
};

#endif
