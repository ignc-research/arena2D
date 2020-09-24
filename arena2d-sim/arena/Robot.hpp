/* Author: Cornelius Marx */
#ifndef BURGER_H
#define BURGER_H

#include "PhysicsWorld.hpp"
#include "LidarCast.hpp"
#include <engine/zColor.hpp>
#include <engine/GlobalSettings.hpp>

/* robot constants */
#define BURGER_MOTOR_FORCE_FACTOR 	5 			// robot acceleration multiplier
#define BURGER_LIDAR_POINT_COLOR	0xFF1020FF 	// color of laser scan sample points
#define BURGER_LIDAR_AREA_COLOR		0xFF102040	// color of laser scan area
#define BURGER_LIDAR_POINT_SIZE		4			// size of laser scan sample points
#define BURGER_TRAIL_WIDTH			2			// width of trail line
#define BURGER_TRAIL_COLOR			0x2255FFFF	// color of trail line
#define BURGER_TRAIL_SEGMENT_SIZE 	0.05 		// distance to travel until a new trail segment is created
#define BURGER_TRAIL_BUFFER_SIZE 	1024 		// maximum number of segments in trail buffer

/* Twist describes linear and angular motions */
struct Twist
{
	Twist(){}
	Twist(float _linear, float _angular): linear(_linear), angular(_angular){}
	float linear;	// linear velocity
	float angular; 	// angular velocity around center
};

/* Robot class controlled by the agent */
class Robot
{
public:
	/* discrete actions */
	enum Action{FORWARD, FORWARD_LEFT, FORWARD_RIGHT, FORWARD_STRONG_LEFT, FORWARD_STRONG_RIGHT, BACKWARD, STOP, NUM_ACTIONS};

	/* constructor
	 * @param world Box2D world to create robot in
	 */
	Robot(b2World * world);

	/* destructor
	 */
	~Robot();

	/* render visualization of laser scan
	 * color shader is assumed to be bound
	 * @param area if true whole scan area is rendered, else only sample points are rendered
	 */
	void renderScan(bool area);

	/* render visualiation of path the robot has taken so far
	 * color shader is assumed to be bound
	 */
	void renderTrail();

	/* update visual trail
	 */
	void updateTrail();

	/* reset visual trail buffer
	 */
	void resetTrail();

	/* reset burger position, orientation and velocities
	 * @param position new position of robot
	 * @param angle new angle of robot in rad
	 */
	void reset(const b2Vec2 & position, float angle = 0);

	/* perform discrete action
	 * @param a discrete action (see enum Action definition)
	 */
	void performAction(Action a);

	/* get twist from specified action
	 * @param a discrete action of which to get the corresponding twist
	 * @param t set to the twist of the action
	 */
	static void getActionTwist(Action a, Twist & t);

	/* perform continuous action
	 * @param t linear and angular velocity to apply to the robot
	 */
	void performAction(const Twist & t);

	/* update laser scanner parameters, such as number of samples
	 * called when settings have changed
	 */
	void updateLidar();

	/* perform raycasts to evaluate laser samples
	 */
	void scan();

	/* signal robot contact with obstacle has started
	 * should only be called by environment
	 * @return true on touch event, else false
	 */
	bool beginContact(){_contactCount++; return (_contactCount == 1);}

	/* signal robot contact with obstacle has ended
	 * should only be called by environment
	 * @return true on if all contacts have ended, else false
	 */
	bool endContact(){bool ended = (_contactCount == 1); _contactCount--; if(_contactCount < 0)_contactCount=0; return ended;}


	/* get laser samples
	 * @num_samples is set to the number of samples in the array returned
	 * @return array of laser samples (distances to obstacles)
	 */
	const float* getSamples(int & num_samples){num_samples = _lidar->getNumSamples(); return _lidar->getDistances();}

	/* get angle from index in laser samples
	 * @param i index in laser samples array (as returned by getSamples())
	 * @return angle in radians of given index
	 */
	float getAngleFromSampleIndex(int i){return _lidar->getAngleFromIndex(i);}

	/* get robot rigid body
	 * @return Box2D body
	 */
	b2Body * getBody(){return _base;}

	/* get position of robot
	 * @return position of robot
	 */
	b2Vec2 getPosition(){return _base->GetTransform().p;}

	/* get angle of robot in rad
	 * @return angle of robot (rad)
	 */
	float getAngle(){return _base->GetTransform().q.GetAngle();}

	/* get circlular sensor covering the whole robot
	 * @return Box2D circle fixture with _safeRadius
	 */
	const b2Fixture* getRadiusSensor(){return _safetyRadiusSensor;}

	/* get radius that encloses all robot fixtures
	 * @return radius in meters
	 */
	float getRadius(){return _safeRadius;}

private:
	/* robot rigid body */
	b2Body * _base;

	/* indicating wheel in _wheelPosition */
	enum WheelIndicator{LEFT, RIGHT};

	/* relative position of the wheels center from center of base body */
	b2Vec2 _wheelPosition[2];

	/* distance between both wheel's center */
	float _wheelDistance;

	/* radius describing circle around robot, calculated on construction based on robot dimensions*/
	float _safeRadius;

	/* laser scanner */
	LidarCast * _lidar;

	/* gl buffer storing scanning data for rendering */
	GLuint _lidarBuffer;

	/* gl buffer storing path robot has taken */
	GLuint _trailBuffer;

	/* last position a trail vertex was put in buffer */
	b2Vec2 _lastTrailPosition;

	/* number of trail vertices used in buffer */
	int _trailVertexCount;

	/* how many samples are currently stored in buffer */
	int _lidarBufferCount; 

	/* circular sensor with radius _safeRadius */
	b2Fixture * _safetyRadiusSensor;

	/* keeping track of the number of contacts for reward function */
	int _contactCount;
};

#endif
