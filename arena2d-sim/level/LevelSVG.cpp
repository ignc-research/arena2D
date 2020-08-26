/* author: Cornelius Marx */
#include "LevelSVG.hpp"

LevelSVG::LevelSVG(const LevelDef & def) : Level(def), _spawnAreas(NULL)
{
	_currentFileIndex = -1;

	const char * folder_path = _SETTINGS->stage.svg_path.c_str();
	INFO_F("Loading svg levels from \"%s\":", folder_path);
	DIR * d;
	d = opendir(_SETTINGS->stage.svg_path.c_str());
	if(d == NULL){
		ERROR_F("Could not open folder \"%s\"!", folder_path);
		return;
	}

	struct dirent *dir;
	char path[256];
	int base_path_len = strlen(folder_path);
	strcpy(path, folder_path);
	// append '/'
	if(base_path_len > 0 && path[base_path_len-1] != '/')
	{
		path[base_path_len++] = '/';
	}
	
	// read every file in the directory and load it
	int error = 0;
	while((dir = readdir(d)) != NULL)
	{
		if(strcmp(dir->d_name, ".") && strcmp(dir->d_name, "..")){// ignore .. and . folders
			strcpy(path + base_path_len, dir->d_name);
			// check ending .svg
			int file_len = strlen(dir->d_name);
			if(file_len >= 4 &&
				(!strcmp(dir->d_name+file_len-4, ".svg") ||
				!strcmp(dir->d_name+file_len-4, ".SVG")))
			{
				INFO_F("  -> Loading \"%s\"...", path);
				SVGFile * f = new SVGFile(path);
				_files.push_back(f);
				if(f->load()){
					ERROR_F("Failed to load \"%s\"", path);
					error++;
				}
			}
		}
	}
	closedir(d);

	if(_files.size() == 0){
		WARNING("No svg levels were loaded!");
		return;
	}
	
	// create goal spawns for every file
	_spawnAreas = new RectSpawn[_files.size()];
	const float margin = f_fmax(_SETTINGS->stage.goal_size/2.0f, _levelDef.robot->getRadius());
	const float block_size = LEVEL_SVG_GOAL_SPAWN_AREA_BLOCK_SIZE;
	for(unsigned int i = 0; i < _files.size(); i++)
	{
		loadFile(i);
		zRect r;
		_files[i]->getArea(r);
		_spawnAreas[i].addQuadTree(r, _levelDef.world, COLLIDE_CATEGORY_STAGE, block_size, margin);
		_spawnAreas[i].calculateArea();
	}
}

LevelSVG::~LevelSVG()
{
	for(int i = 0; i < (int)_files.size(); i++)
		delete _files[i];
	
	delete[] _spawnAreas;
}

void LevelSVG::resetRobot()
{
	b2Vec2 pos(0,0);
	if(_currentFileIndex >= 0){
		_spawnAreas[_currentFileIndex].getRandomPoint(pos);
	}
	_levelDef.robot->reset(pos, f_frandomRange(0, 2*M_PI));

}

void LevelSVG::reset(bool robot_position_reset)
{
	// no files loaded -> nothing to be done
	if(_currentFileIndex < 0){
		return;
	}

	// select random stage
	int new_stage = f_irandomRange(0, _files.size()-1);
	if(new_stage >= (int)_files.size())// stage has not changed
		new_stage = 0;
	if(new_stage != _currentFileIndex){
		_currentFileIndex = new_stage;
		loadFile(_currentFileIndex);
		robot_position_reset = true;
	}

	// reset robot position to random position
	if(robot_position_reset)
	{
		resetRobot();
	}


	// spawn goal
	randomGoalSpawnUntilValid(&_spawnAreas[_currentFileIndex]);
}

void LevelSVG::loadFile(int index)
{
	// clear old level
	clear();

	// get stage
	SVGFile * f = _files[index];

	// create border
	float half_width = 0.5*f->getWidth();
	float half_height = 0.5*f->getHeight();
	if(half_width*half_height > 0)
		createBorder(half_width, half_height);

	// adding all shapes
	addShape(f->getShapes());

	_currentFileIndex = index;
}


void LevelSVG::renderGoalSpawn()
{
	if(_currentFileIndex < 0 || (unsigned int)_currentFileIndex >= _files.size())
		return;
	
	Z_SHADER->setColor(zColor(LEVEL_GOAL_SPAWN_COLOR));
	_spawnAreas[_currentFileIndex].render();
}
