/* Author: Cornelius Marx */
#ifndef LEVEL_FACTORY_H
#define LEVEL_FACTORY_H

#include <arena/CommandRegister.hpp>
#include <engine/zSingleton.hpp>

// include all level classes
#include <level/Level.hpp>
#include <level/LevelEmpty.hpp>
#include <level/LevelRandom.hpp>
#include <level/LevelSVG.hpp>

// singleton-get macro
#define LEVEL_FACTORY LevelFactory::get()

class LevelFactory : public zTSingleton<LevelFactory>
{
public:
	/* constructor */
	LevelFactory();

	/* destructor */
	~LevelFactory(){}

	/* create generic level by name
	 * @param level_name name of level to be created
	 * @param w Box2D world the level is created in
	 * @param params additional level specific parameters
	 * @return new level or NULL on error
	 * NOTE: level must be deleted manually if not needed anymore (delete level;)
	 */
	Level* createLevel(const char * level_name, const LevelDef& d, const ConsoleParameters & params);

	/* get descriptions for all levels
	 * @param cmd_list contains descriptions for every level after call
	 */
	void getLevelDescriptions(list<const CommandDescription*> & cmd_list){
		_levels.getAllCommandDescriptions(cmd_list);
	}

	/* generic level create function
	 * this template function is called to create a generic level type with no parameters
	 * @param T level class
	 * @param d level initializer
	 * @param params ignored for generic template level
	 */
	template <class T>
	Level* createLevelT(const LevelDef & d, const ConsoleParameters & params)
	{
		return new T(d);
	}

	/* specialized level create functions
	 * these functions are called to create a specific level type that is created in a non generic way or has additional parameters
	 * @param w Box2D world to create the levels in
	 * @param params additional level specific parameters 
	 */

	/* random level */
	Level* createLevelRandom(const LevelDef & d, const ConsoleParameters & params)
	{
		bool level_dynamic = params.getFlag("--dynamic");
		return new LevelRandom(d, level_dynamic);
	}


private:

	/* command register matching level-name -> create-function */
	CommandRegister<LevelFactory, Level*, const LevelDef&, const ConsoleParameters&> _levels;
	
};


#endif
