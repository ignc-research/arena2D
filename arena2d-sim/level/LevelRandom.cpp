#include "LevelRandom.hpp"

void LevelRandom::reset(bool robot_position_reset)
{
	// clear old bodies and spawn area
    clear();
    if(_human )
        wanderers.freeWanderers();
    if(_dynamic)
        wanderers.freeRobotWanderers();

	// get constants
	const float half_width = _SETTINGS->stage.level_size/2.f;
	const float half_height = _SETTINGS->stage.level_size/2.f;
	const float half_goal_size = _SETTINGS->stage.goal_size/2.f;
	const float dynamic_radius = _SETTINGS->stage.dynamic_obstacle_size/2.f;
	const float dynamic_speed = _SETTINGS->stage.obstacle_speed;
	const int num_obstacles = _SETTINGS->stage.num_obstacles;
	const int num_dynamic_obstacles = _SETTINGS->stage.num_dynamic_obstacles;
	const float min_obstacle_radius = _SETTINGS->stage.min_obstacle_size/2;
	const float max_obstacle_radius = _SETTINGS->stage.max_obstacle_size/2;
	const zRect main_rect(0, 0, half_width, half_height);
	const zRect big_main_rect(0, 0, half_width+max_obstacle_radius, half_height+max_obstacle_radius);

	// create border around level
	createBorder(half_width, half_height);

	if(robot_position_reset){
		resetRobotToCenter();
	}

	// calculate spawn area for static obstacles
	RectSpawn static_spawn;
	static_spawn.addCheeseRect(big_main_rect, _levelDef.world, COLLIDE_CATEGORY_PLAYER, max_obstacle_radius);
	static_spawn.calculateArea();

	// create static obstacles
	for(int i = 0; i < num_obstacles; i ++){
		b2Vec2 p;
		static_spawn.getRandomPoint(p);
		zRect aabb;
		addRandomShape(p, min_obstacle_radius, max_obstacle_radius, &aabb);
	}

	// calculating goal spawn area
	_goalSpawnArea.addQuadTree(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE,
								LEVEL_RANDOM_GOAL_SPAWN_AREA_BLOCK_SIZE, half_goal_size);
	_goalSpawnArea.calculateArea();

	// dynamic obstacles
    if(_dynamic || _human){
        _dynamicSpawn.clear();
        _dynamicSpawn.addCheeseRect(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE | COLLIDE_CATEGORY_PLAYER, dynamic_radius);
        _dynamicSpawn.calculateArea();
        wanderers.reset(_dynamicSpawn, _dynamic, _human);
    }

	randomGoalSpawnUntilValid();
}

void LevelRandom::renderGoalSpawn()
{
	Level::renderGoalSpawn();
	Z_SHADER->setColor(zColor(0.1, 0.9, 0.0, 0.5));
	_dynamicSpawn.render();
}

float LevelRandom::getReward()
{
	float reward = 0;
	_closestDistance_old.clear();
	_closestDistance.clear();

	//reward for observed humans inside camera view of robot (number limited by num_obs_humans)
	if(_SETTINGS->training.reward_function == 1 || _SETTINGS->training.reward_function == 4){
		wanderers.get_old_observed_distances(_closestDistance_old);
		wanderers.get_observed_distances(_closestDistance);
	}
	//reward for all humans in the level
	else if(_SETTINGS->training.reward_function == 2 || _SETTINGS->training.reward_function == 3){
		wanderers.get_old_distances(_closestDistance_old);
		wanderers.get_distances(_closestDistance);
	}
	

	for(int i = 0; i < _closestDistance_old.size(); i++){
		float distance_after = _closestDistance[i];
		float distance_before = _closestDistance_old[i];
		// give reward only if current distance is smaller than the safety distance
		if(distance_after < _SETTINGS->training.safety_distance_human){
 			//give reward for distance to human decreased/increased linearly depending on the distance change 
			if(_SETTINGS->training.reward_function == 3 || _SETTINGS->training.reward_function == 4){
				if(distance_after < distance_before){
					reward += _SETTINGS->training.reward_distance_to_human_decreased * (distance_before - distance_after);
				}else if(distance_after > distance_before){
					reward += _SETTINGS->training.reward_distance_to_human_increased * (distance_after - distance_before);
				}
			}
			//give constant reward for distance to human decreased/increased
			else{
				if(distance_after < distance_before){
					reward += _SETTINGS->training.reward_distance_to_human_decreased;
				}else if(distance_after > distance_before){
					reward += _SETTINGS->training.reward_distance_to_human_increased;
				}
			}
		}
	}
	return reward;
}
