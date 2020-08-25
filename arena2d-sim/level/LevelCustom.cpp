#include "LevelCustom.hpp"


void LevelCustom::reset(bool robot_position_reset) {
    // clear old bodies and spawn area
    clear();

    float half_width = _SETTINGS->stage.level_size / 2.f;

    float half_height = _SETTINGS->stage.level_size / 2.f;
    float half_goal_size = _SETTINGS->stage.goal_size / 2.f;
    const float dynamic_radius = WandererBipedal::getRadius();
    const float dynamic_speed = _SETTINGS->stage.obstacle_speed;
    const int num_dynamic_obstacles = _SETTINGS->stage.num_dynamic_obstacles;
    const float min_obstacle_radius = _SETTINGS->stage.min_obstacle_size / 2;
    const float max_obstacle_radius = _SETTINGS->stage.max_obstacle_size / 2;
    const zRect main_rect(0, 0, half_width, half_height);
    const zRect big_main_rect(0, 0, half_width + max_obstacle_radius, half_height + max_obstacle_radius);

    int num_obstacles = 5;

    if(robot_position_reset){
        resetRobotToCenter();
    }

    createBorder(half_width, half_height);

    RectSpawn static_spawn;
    static_spawn.addCheeseRect(big_main_rect, _levelDef.world, COLLIDE_CATEGORY_PLAYER, max_obstacle_radius);
    static_spawn.calculateArea();

    std::vector <zRect> robot_hole(1);
    std::vector <zRect> holes(num_obstacles);

    printf(">> create obstacles << \n");
    for (int i = 0; i < num_obstacles; i++) {
        b2Vec2 p;
        static_spawn.getRandomPoint(p);
        zRect aabb;
        //b2Body *b;
        int randomNumber = (rand() % 6);
        bool boundary_cond = true;
        switch (randomNumber) {
            case 0:
                b2Body *e;
                e = addRandomShape(p, _SETTINGS->stage.min_obstacle_size / 2,
                                   _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                //d->SetTransform(b2Vec2(p.x, p.y), 0);
                break;
//                p.x = p.x;// -0.5;
//                p.y = p.y;//-0.5;
//                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
//                                                            _SETTINGS->stage.max_obstacle_size / 2, &aabb);
//                p.x = p.x-0.45;//-0.95;
//                p.y = p.y;//-0.5;
//                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
//                                                            _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                break;
            case 1:

                b2Body *f;
                f = addRandomShape(p, _SETTINGS->stage.min_obstacle_size / 2,
                                   _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                //b->SetTransform(b2Vec2(p.x, p.y), 0);
                break;
//                p.x;// 1.5;
//                p.y;// = 1.5;
//                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
//                                                            _SETTINGS->stage.max_obstacle_size / 2, &aabb);
//                p.x = p.x + 0.45;
//                p.y ;//1.5;
//                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
//                                                            _SETTINGS->stage.max_obstacle_size / 2, &aabb);
//                p.x = p.x - 0.45 + _SETTINGS->stage.min_obstacle_size / 2;
//                p.y = p.y;
//                generateRandomBodyHorizontal(p, _SETTINGS->stage.min_obstacle_size / 2,
//                                                              _SETTINGS->stage.max_obstacle_size / 2, &aabb);
//                p.x = p.x + _SETTINGS->stage.min_obstacle_size / 2;
//                p.y = p.y + 0.45;//1.95;
//                generateRandomBodyHorizontal(p, _SETTINGS->stage.min_obstacle_size / 2,
//                                                              _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                break;
            case 2:
//                p.x = 1.5 + _SETTINGS->stage.min_obstacle_size / 2;
//                p.y = -1.5;

                while(boundary_cond){
                    if(p.y < 1.55 && p.x > -1.2 ){
                        boundary_cond = false;
                    }else{
                        static_spawn.getRandomPoint(p);
                    }
                }
                generateRandomBodyHorizontal(p, _SETTINGS->stage.min_obstacle_size / 2,
                                                              _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                p.x = p.x;//1.5 + _SETTINGS->stage.min_obstacle_size / 2;
                p.y = p.y+ 0.45;//-1.95;
                generateRandomBodyHorizontal(p, _SETTINGS->stage.min_obstacle_size / 2,
                                                              _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                break;
            case 3:
                b2Body *b;
                b = addRandomShape(p, _SETTINGS->stage.min_obstacle_size / 2,
                                            _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                //b->SetTransform(b2Vec2(p.x, p.y), 0);
                break;
            case 4:
//                p.x = -2.0;
//                p.y = 2.0;
                while(boundary_cond){
                    if(p.y >  -1 && p.x > -1.4 ){
                            boundary_cond = false;
                    }else{
                        static_spawn.getRandomPoint(p);
                    }
                }
                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
                                                            _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                p.x = p.x - 0.45;
                p.y = p.y;
                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
                                                            _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                break;
            case 5:
                b2Body *c;
                c = addRandomShape(p, _SETTINGS->stage.min_obstacle_size / 2,
                                   _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                //c->SetTransform(b2Vec2(p.x, p.y), 0);
                break;
            case 6:
                b2Body *d;
                d = addRandomShape(p, _SETTINGS->stage.min_obstacle_size / 2,
                                   _SETTINGS->stage.max_obstacle_size / 2, &aabb);
                //d->SetTransform(b2Vec2(p.x, p.y), 0);
                break;
        }
    }



    printf(">> goal spawn << \n");
    // spawning dynamic obstacles
    _goalSpawnArea.addQuadTree(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE,
                               LEVEL_CUSTOM_GOAL_SPAWN_AREA_BLOCK_SIZE, half_goal_size);
    _goalSpawnArea.calculateArea();

    printf(">> check dynamic << \n");
    if (_dynamic) {
        _dynamicSpawn.clear();
        _dynamicSpawn.addCheeseRect(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE | COLLIDE_CATEGORY_PLAYER,
                                    dynamic_radius);
        _dynamicSpawn.calculateArea();
        std::vector<b2Vec2> spawn_position;
		for(int i = 0; i < num_dynamic_obstacles; i++){
			b2Vec2 p;
			_dynamicSpawn.getRandomPoint(p);
            printf(">> create wanderer %d << \n", i);
			spawn_position.push_back(p);
		}
		wanderers.reset(spawn_position);
    }

    //printf("STAGE:%f", _SETTINGS->stage.level_size);
    // adding spawn area
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
    float y2 = p.y + min_radius;
    float x2 = p.x;
    float y3 = y2;
    float x3 = x2 - 2 * max_radius;
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
        aabb->x = p.x;
        aabb->y = p.y;
        aabb->w = (max_v.x - min_v.x) / 2.0f;
        aabb->h = (max_v.y - min_v.y) / 2.0f;
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

    float x2 = p.x + min_radius;
    float y2 = p.y;
    float x3 = x2;
    float y3 = y2 - 2 * max_radius;
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
        aabb->x = p.x;
        aabb->y = p.y;
        aabb->w = (max_v.x - min_v.x) / 2.0f;
        aabb->h = (max_v.y - min_v.y) / 2.0f;
    }
    shape.Set(verts, vert_count);
    return addShape(&shape);
}

float LevelCustom::getReward()
{
	float reward = 0;
	_closestDistance_old.clear();
	_closestDistance.clear();
	wanderers.get_old_Distance(_closestDistance_old);
	wanderers.get_Distance(_closestDistance);

	for(int i = 0; i < _closestDistance_old.size(); i++){
		float distance_after = _closestDistance[i];
		float distance_before = _closestDistance_old[i];
		// checking reward for distance to human decreased/increased
		if(distance_after < _SETTINGS->training.safety_distance_human){
			if(distance_after < distance_before){
				reward += _SETTINGS->training.reward_distance_to_human_decreased;
			}
			else if(distance_after > distance_before){
				reward += _SETTINGS->training.reward_distance_to_human_increased;
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
