#include "LevelEmpty.hpp"

LevelEmpty::LevelEmpty(const LevelDef & d, bool create_borders): Level(d){
	_border = create_borders;
	const float levelwidth = _SETTINGS->stage.level_size/2.f;
	const float levelheight = _SETTINGS->stage.level_size/2.f;
	const float margin = _SETTINGS->stage.goal_size/2.f;
	if(_border){
		createBorder(levelwidth, levelheight);
	}
	zRect area(0, 0, levelwidth, levelheight);
	_goalSpawnArea.addQuadTree(	area, _levelDef.world, COLLIDE_CATEGORY_STAGE, 0.1, margin);
	_goalSpawnArea.calculateArea();
}


void LevelEmpty::reset(bool robot_position_reset)
{
	if(robot_position_reset){
		resetRobotToCenter();
	}
	randomGoalSpawnUntilValid();
}
