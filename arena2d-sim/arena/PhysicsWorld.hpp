#ifndef PHYSICS_WORLD_H
#define PHYSICS_WORLD_H

#include <engine/Renderer.hpp>
#include <engine/zSingleton.hpp>
#include <engine/zVector2d.hpp>
#include <engine/zVector3d.hpp>
#include <engine/zVector4d.hpp>
#include <engine/zMatrix4x4.hpp>
#include <box2d/box2d.h>
#include <engine/zRect.hpp>
#include <engine/zColor.hpp>
#include <list>
#define _PHYSICS PhysicsWorld::get()
#define _PHYSICS_WORLD _PHYSICS->getWorld()

/* debug render flags */
#define PHYSICS_RENDER_SENSORS		0x00000002
#define PHYSICS_RENDER_DYNAMIC		0x00000004
#define PHYSICS_RENDER_STATIC		0x00000008
#define PHYSICS_RENDER_ALL			0xFFFFFFFF

/* collision filter category bits (prefilter before solver) */
#define COLLIDE_CATEGORY_PLAYER				0x0001
#define COLLIDE_CATEGORY_STAGE				0x0002
#define COLLIDE_CATEGORY_GOAL				0x0004
#define COLLIDE_CATEGORY_DONT_RENDER		0x8000 // any fixture having this bit set in category is not rendered when debugDraw() is called

/* useful arrays for iterating over directions
 */
enum PhysicsDirection{RIGHT, UP, LEFT, DOWN};

/* unit vector pointing in the direction indexed with a PhysicsDirection */
extern const b2Vec2 PHYSICS_NEIGHBOUR_MAP[4];

/* unit vector pointing in a 45 degree angle between the given PhysicsDirection index and the next index
 * RIGHT -> RIGHT_UP
 * UP -> UP_LEFT
 * LEFT -> LEFT_DOWN
 * DOWN -> DOWN_RIGHT
 */
extern const b2Vec2 PHYSICS_NEIGHBOUR_MAP_DIAGONAL[4];

/* AABB query to determine visible fixtures */
class PhysicsWorldAABBQuery : public b2QueryCallback
{
public:
	PhysicsWorldAABBQuery(){}
	~PhysicsWorldAABBQuery(){}
	void reset(){_visibleFixtures.clear();}
	bool ReportFixture(b2Fixture * fixture)
	{
		//render sensors last
		if(fixture->IsSensor())
			_visibleFixtures.push_back(fixture);
		else
			_visibleFixtures.push_front(fixture);
		return true;
	}

	std::list<b2Fixture*> _visibleFixtures;
};

/* physics world, providing access to a global Box2D world */
class PhysicsWorld : public zTSingleton<PhysicsWorld>
{
public:
	/* constructor
	 */
	PhysicsWorld();

	/* destructor, remove global world and static body
	 */
	~PhysicsWorld();

	/* initialize global world, create static body
	 */
	void init();
	
	/* set simulation iterations
	 * @param velocity_iterations number of velocity iterations per step
	 * @param position_iterations number of position iterations per step
	 */
	void setIterations(int velocity_iterations, int position_iterations){
		_velocityIterations = velocity_iterations;
		_positionIterations = position_iterations;
	}

	/* perform time step in global world
	 */
	void step(float time_step);

	/* perform AABB query to get all fixtures that are visible in a given area in the global world
	 * @param visible_area the AABB area to perform query with
	 */
	void calculateVisibleFixtures(const zRect & visible_area){calculateVisibleFixturesWorld(_world, visible_area);}

	/* perform AABB query to get all fixtures that are visible in a given area in a given world
	 * @param w Box2D world
	 * @param visible_area the AABB area to perform query with
	 */
	void calculateVisibleFixturesWorld(b2World * w, const zRect & visible_area);

	/* get global world
	 * @return global Box2D world
	 */
	b2World* getWorld(){return _world;}

	/* get static body
	 * @return an always existing static body at world coordinates (0,0)
	 */
	b2Body * getStaticBody(){return _staticBody;}

	/* get visible fixtures from previous call to calculateVisibleFixtures() or calculateVisibleFixturesWorld()
	 * @return list of visible fixtures
	 */
	std::list<b2Fixture*>* getVisibleFixtures(){return &_aabbQuery._visibleFixtures;}


	/* render fixtures of every body of the global world
	 * @param flags control what object type is rendered (see debug render flags defined above) 
	 * @param category_bits control objects of what Box2D filter category are drawn (see categories defined above)
	 * NOTE: calculateVisibleFixtures() needs to be called before
	 */
	void debugDraw(unsigned int flags, uint16 category_bits = 0xFFFF){debugDrawWorld(_world, flags, category_bits);}

	/* render fixtures of every body of a given world
	 * @param w the Box2D world of which to draw the fixtures
	 * @param flags control what is rendered (see debug render flags defined above) 
	 * @param category_bits control objects of what Box2D filter category are drawn (see categories defined above)
	 * NOTE: calculateVisibleFixtures() needs to be called before
	 */
	void debugDrawWorld(b2World * w, unsigned int flags, uint16 category_bits = 0xFFFF);

	/* setting colors for object types
	 * @param c new color for object type
	 */
	void setDynamicColor(const zColor & c){_dynamicColor = c;}
	void setSensorColor(const zColor & c){_sensorColor = c;}
	void setSleepColor(const zColor & c){_sleepColor = c;}
	void setStaticColor(const zColor & c){_staticColor = c;}
	void setCollisionsColor(const zColor & c){_collisionsColor = c;}
	void setFillAlpha(float a){_fillAlpha = a;}

private:
	/* global Box2D world */
	b2World * _world;

	/* position iterations */
	int _positionIterations;

	/* velocity iterations */
	int _velocityIterations;

	/* query visible fixtures */
	PhysicsWorldAABBQuery _aabbQuery;

	/* an empty static body that can be used as anchor joints */
	b2Body * _staticBody;

	/* fixture colors for debug drawing */
	zColor _dynamicColor;
	zColor _sensorColor;
	zColor _sleepColor;
	zColor _staticColor;
	zColor _staticColor2;
	zColor _collisionsColor;
	zColor _kinematicColor;

	/* drawing alpha value, the actual alpha value in zColor is ignored */
	float _fillAlpha;
};

#endif
