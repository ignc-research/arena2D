#ifndef LEVELSTATIC_H
#define LEVELSTATIC_H

#include "Level.hpp"
#include "Wanderers.hpp"

#define LEVEL_RANDOM_GOAL_SPAWN_AREA_BLOCK_SIZE 0.1 // maximum size of block when creating quad tree of goal spawn area

/* randomly generated level with static obstacles and optional dynamic obstacles */
class LevelRandom : public Level{

public:
	/* constructor
	 */
	LevelRandom(const LevelDef & d, bool dynamic = false, bool human = false) :Level(d), _dynamic(dynamic), _human(human), wanderers(d){}

	/* destructor
	 */
	~LevelRandom(){}

	/* reset
	 */
	void reset(bool robot_position_reset) override;

	/* update
	 */
	void update() override{
        wanderers.update();
	};

	/* render spawn area
	 * overriding to visualize spawn area for dynamic obstacles
	 */
	void renderGoalSpawn() override;

private:
	/* free wanderers and clear list
	 */
	void freeWanderers();

	/* if set to true, create dynamic obstacles (wanderers) in addition to static */
	bool _dynamic;
    bool _human;

	/* list that stores all wanderers for dynamic level */
    Wanderers wanderers;

	/* spawn area for dynamic obstacles */
	RectSpawn _dynamicSpawn;
};

#endif
