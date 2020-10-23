#include "LevelCustom.hpp"
#include "math.h"

void LevelCustom::reset(bool robot_position_reset) {
    // clear old bodies and spawn area
    clear();
    if(_dynamic){
        wanderers.freeWanderers();
    }

    float half_width = _SETTINGS->stage.level_size / 2.f;
    float half_height = _SETTINGS->stage.level_size / 2.f;
    float half_goal_size = _SETTINGS->stage.goal_size / 2.f;
    const float dynamic_radius = WandererBipedal::getRadius();
    const float dynamic_speed = _SETTINGS->stage.obstacle_speed;
    const int num_dynamic_obstacles = _SETTINGS->stage.num_dynamic_obstacles;
    const float min_obstacle_radius = _SETTINGS->stage.min_obstacle_size / 2;
    const float max_obstacle_radius = _SETTINGS->stage.max_obstacle_size / 2;
    const float robot_diameter = _levelDef.robot->getRadius() * 2;
    const zRect main_rect(0, 0, half_width, half_height);
    const zRect big_main_rect(0, 0, half_width + max_obstacle_radius, half_height + max_obstacle_radius);

    int num_obstacles = _SETTINGS->stage.num_obstacles;

    if(robot_position_reset){
        resetRobotToCenter();
    }

    createBorder(half_width, half_height);

    RectSpawn static_spawn;
    static_spawn.addCheeseRect(big_main_rect, _levelDef.world, COLLIDE_CATEGORY_PLAYER, max_obstacle_radius);
    static_spawn.calculateArea();

    std::vector <zRect> robot_hole(1);
    std::vector <zRect> holes(num_obstacles);

    std::list<b2Vec2*> existing_positions;
    std::list<zRect*> existing_boxes;
    for (int i = 0; i < num_obstacles; i++) {
        //static_spawn.getRandomPoint(p);
        b2Vec2 p;
        zRect aabb;
        //makes sure obstacle spawns on free space
        int randomNumber = (rand() % 6);
        bool spawn_valid = obstacleSpawnUntilValid(&static_spawn, existing_boxes, p, randomNumber);
        //printf("obstacle point found\n");
        float random_length;
        float random_width;
        zRect * first_horizontal;
        zRect * second_horizontal;
        zRect * horizontal_box;
        zRect * first_vertical;
        zRect * second_vertical;
        zRect * vertical_box;


        //bool boundary_cond = true;


        // controls the number of occurences of different shapes: random, vertical blocks and horizontal blocks
        switch (randomNumber) {
            case 1:
            case 6:
            case 5:
            case 3:
                //random obstacle is created
                if (!spawn_valid) continue;
                addRandomShape(p, (_SETTINGS->stage.min_obstacle_size / 2),
                               (_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                ;
                //printf("obstacle point added randomShape\n");
                //existing_positions.push_back(new b2Vec2(p.x, p.y));
                existing_boxes.push_back(new zRect(aabb));
                break;
            case 2:
                //  horizontal box is created
                if (!spawn_valid) continue;
                random_length =  f_frandomRange(0.5,2.5);
                random_width = f_frandomRange(2*robot_diameter, 3*robot_diameter);
                generateRandomBodyHorizontal(p, (_SETTINGS->stage.min_obstacle_size / 2),
                                             random_length*(_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                //printf("obstacle point added horizontal 1\n");
                existing_positions.push_back(new b2Vec2(p.x, p.y));
                first_horizontal = new zRect(aabb);
                //existing_boxes.push_back(new zRect(aabb));

                //  corresponding second horizontal box is created to build a corridor with the firts one
                p.x = p.x;//1.5 + _SETTINGS->stage.min_obstacle_size / 2;
                p.y = p.y+ random_width;//0.45;//-1.95;
                generateRandomBodyHorizontal(p, _SETTINGS->stage.min_obstacle_size / 2,
                                             random_length*(_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                existing_positions.push_back(new b2Vec2(p.x, p.y));
                second_horizontal = new zRect(aabb);
                horizontal_box = new zRect(first_horizontal->x, first_horizontal->y + abs(first_horizontal->y - second_horizontal->y)/2.0f,  second_horizontal->w, (abs(first_horizontal->y - second_horizontal->y) + (first_horizontal->h* 2.0f))/2.0f);

                existing_boxes.push_back(horizontal_box);

                break;
            case 4:
                //vertical box is created
                if (!spawn_valid) continue;
                random_length =  f_frandomRange(0.5,2.5);
                random_width = f_frandomRange(2*robot_diameter, 3*robot_diameter);
                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
                                           random_length*(_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                existing_positions.push_back(new b2Vec2(p.x, p.y));

                first_vertical = new zRect(aabb);

                //  corresponding second vertical box is created to build a corridor with the first one

                p.x = p.x - random_width;//- 0.45;
                p.y = p.y;
                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
                                           random_length*(_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                //printf("obstacle point added vertical 2\n");
                existing_positions.push_back(new b2Vec2(p.x, p.y));

                second_vertical = new zRect(aabb);
                vertical_box = new zRect(first_vertical->x - abs(first_vertical->x - second_vertical->x)/2.0f, first_vertical->y, (abs(first_vertical->x - second_vertical->x) + (first_vertical->w * 2.0f))/2.0f, second_vertical->h);
                existing_boxes.push_back(vertical_box);


                break;
        }
    }

    _goalSpawnArea.addQuadTree(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE,
                               LEVEL_CUSTOM_GOAL_SPAWN_AREA_BLOCK_SIZE, half_goal_size);
    _goalSpawnArea.calculateArea();

    if (_dynamic) {
        _dynamicSpawn.clear();
        //printf("addCheeseRect\n");
        _dynamicSpawn.addCheeseRect(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE | COLLIDE_CATEGORY_PLAYER,
                                    dynamic_radius);
        //printf("calculateArea\n");
        _dynamicSpawn.calculateArea();
		wanderers.reset(_dynamicSpawn);
    }

    randomGoalSpawnUntilValid();
}





b2Body *
LevelCustom::generateRandomBodyHorizontal(const b2Vec2 &p, float min_radius, float max_radius, zRect *aabb) {
    int vert_count = 4;//f_irandomRange(3, 6);
    b2PolygonShape shape;
    shape.m_count = vert_count;
    b2Vec2 verts[vert_count];
    b2Vec2 max_v(-0, -0);
    b2Vec2 min_v(50000, 50000);
    //set fixed shape
    float y2 = p.y + min_radius;
    float x2 = p.x;
    float y3 = y2;
    float x3 = x2 - max_radius;
    float y4 = p.y;
    float x4 = x3;
    for (int i = 0; i < vert_count; i++) {
        switch (i) {
            case 0:
                verts[i].Set(p.x, p.y);
                break;
            case 1:
                verts[i].Set(x2, y2);
                break;
            case 2:
                verts[i].Set(x3, y3);
                break;
            case 3:
                verts[i].Set(x4, y4);
                break;
        }

        //prevents spawning outside
        if (verts[i].x > max_v.x) {
            max_v.x = verts[i].x;
        }
        if (verts[i].y > max_v.y) {
            max_v.y = verts[i].y;
        }
        if (verts[i].x < min_v.x) {
            min_v.x = verts[i].x;
        }
        if (verts[i].y < min_v.y) {
            min_v.y = verts[i].y;
        }
    }
    if (aabb != NULL) {
        aabb->x = p.x - max_radius/2.0f;
        aabb->y = p.y + min_radius/2.0f;
        aabb->w = max_radius/2.0f;//(max_v.x - min_v.x);// / 2.0f;
        aabb->h = min_radius/2.0f;//(max_v.y - min_v.y);// / 2.0f;
    }
    shape.Set((b2Vec2 *) verts, vert_count);
    return addShape(&shape);
}

b2Body *
LevelCustom::generateRandomBodyVertical(const b2Vec2 &p, float min_radius, float max_radius, zRect *aabb) {
    int vert_count = 4;//f_irandomRange(3, 6);
    b2PolygonShape shape;
    shape.m_count = vert_count;
    b2Vec2 verts[vert_count];
    b2Vec2 max_v(-0, -0);
    b2Vec2 min_v(50000, 50000);
    float rotation = f_frandomRange(0, 2 * M_PI);

    //set fixed shape
    float x2 = p.x + min_radius;
    float y2 = p.y;
    float x3 = x2;
    float y3 = y2 - max_radius;
    float x4 = p.x;
    float y4 = y3;
    for (int i = 0; i < vert_count; i++) {
        switch (i) {
            case 0:
                verts[i].Set(p.x, p.y);
                break;
            case 1:
                verts[i].Set(x2, y2);
                break;
            case 2:
                verts[i].Set(x3, y3);
                break;
            case 3:
                verts[i].Set(x4, y4);
                break;
        }
        //prevents spawning outside
        if (verts[i].x > max_v.x) {
            max_v.x = verts[i].x;
        }
        if (verts[i].y > max_v.y) {
            max_v.y = verts[i].y;
        }
        if (verts[i].x < min_v.x) {
            min_v.x = verts[i].x;
        }
        if (verts[i].y < min_v.y) {
            min_v.y = verts[i].y;
        }
    }
    if (aabb != NULL) {
        aabb->x = p.x + min_radius/2.0f;
        aabb->y = p.y - max_radius/2.0f;
        aabb->w = min_radius/2.0f;//(max_v.x - min_v.x) / 2.0f;
        aabb->h = max_radius/2.0f;//(max_v.y - min_v.y) / 2.0f;
    }
    shape.Set(verts, vert_count);
    return addShape(&shape);
}

float LevelCustom::getReward()
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

void LevelCustom::renderGoalSpawn() {
    Level::renderGoalSpawn();
    Z_SHADER->setColor(zColor(0.1, 0.9, 0.0, 0.5));
    _dynamicSpawn.render();
}
