#ifndef LEVELMAZE_H
#define LEVELMAZE_H

#include "Level.hpp"
#include "Wanderer.hpp"

#define LEVEL_RANDOM_GOAL_SPAWN_AREA_BLOCK_SIZE 0.1 // maximum size of block when creating quad tree of goal spawn area

class LevelMaze : public Level
{
public:

	/* constructor
	 */
	LevelMaze(const LevelDef & d, bool dynamic = false) : Level(d), _dynamic(dynamic){}

	/* destructor
	 */
	~LevelMaze(){freeWanderers();}

	// copy the functions from levelRandom.hpp to have a try at first
	/* reset
	 */
	void reset(bool robot_position_reset) override;

	/* update
	 */
	void update() override;

	/* render spawn area
	 * overriding to visualize spawn area for dynamic obstacles
	 */
	void renderGoalSpawn() override;

	// from here all the functions are from old version
	//b2Body* generateRandomBody(float min_radius, float max_radius, zRect * aabb);
    b2Body* generateRandomWalls11(int index, zRect * aabb);
   	b2Body* generateRandomWalls22(int index, int numm, zRect * aabb);


private:
	/* free wanderers and clear list
	 */
	void freeWanderers();

	/* if set to true, create dynamic obstacles (wanderers) in addition to static */
	bool _dynamic;

	/* list that stores all wanderers for dynamic level */
	std::list<Wanderer*> _wanderers;

	/* spawn area for dynamic obstacles */
	RectSpawn _dynamicSpawn;

};

#endif
