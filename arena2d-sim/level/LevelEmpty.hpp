/* author: Cornelius Marx */
#ifndef LEVEL_EMPTY_H
#define LEVEL_EMPTY_H

#include "Level.hpp"

/* empty level with optional border */
class LevelEmpty : public Level
{
public:
	LevelEmpty(const LevelDef & d, bool create_borders = true);

	void reset() override{
		randomGoalSpawnUntilValid();
	}

private:

	bool _border;
};

#endif
