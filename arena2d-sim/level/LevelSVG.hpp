/* author: Cornelius Marx */
#ifndef LEVEL_SVG_H
#define LEVEL_SVG_H 

#include "Level.hpp"
#include <dirent.h>
#include "SVGFile.hpp"

#define LEVEL_SVG_GOAL_SPAWN_AREA_BLOCK_SIZE 0.1 // maximum size of block when creating quad tree of goal spawn area

class LevelSVG : public Level
{
public:
	LevelSVG(const LevelDef &);
	~LevelSVG();

	void reset(bool robot_position_reset) override;

	void renderGoalSpawn()override;
private:
	void resetRobot();
	void loadFile(int index);
	std::vector<SVGFile*> _files;

	/* array of goal spawns for each file */
	RectSpawn * _spawnAreas;
	int _currentFileIndex;
};

#endif
