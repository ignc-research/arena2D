/* Author: Cornelius Marx */
#include "LevelFactory.hpp"

LevelFactory::LevelFactory()
{
	/*** adding levels to register ***/

	// empty with border
	_levels.registerCommand(&LevelFactory::createLevelT<LevelEmpty>,
								"empty",
								"",
								"Empty level with border");

	// static
	_levels.registerCommand(&LevelFactory::createLevelRandom,
								"random",
								"[--dynamic]",
								"Randomized static Level and optional dynamic obstacles (flag --dynamic)");

	// svg
	_levels.registerCommand(&LevelFactory::createLevelT<LevelSVG>,
								"svg",
								"",
								"Levels loaded from svg-files");
}

Level* LevelFactory::createLevel(const char * level_name, const LevelDef& d, const ConsoleParameters & params)
{
	auto cmd = _levels.getCommand(level_name);
	// command does not exist
	if(cmd == NULL){
		ERROR_F("Unknown level '%s'!", level_name);
		return NULL;
	}

	return cmd->exec(this, d, params);
}
