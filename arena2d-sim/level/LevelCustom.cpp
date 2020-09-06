#include "LevelCustom.hpp"
#include "math.h"

void LevelCustom::reset(bool robot_position_reset) {
    // clear old bodies and spawn area
    clear();
    _dynamic = true;
    if (_dynamic)
        freeWanderers();

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

    int num_obstacles = 8;

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
    for (int i = 0; i < num_obstacles; i++) {
        //static_spawn.getRandomPoint(p);
        b2Vec2 p;
        zRect aabb;
        //b2Body *b;
        obstacleSpawnUntilValid(&static_spawn, existing_positions, p);
        printf("obstacle point found\n");
        float random_length;
        float random_width;
        int randomNumber = (rand() % 6);
        bool boundary_cond = true;
        switch (randomNumber) {
            case 0:
            case 1:
            case 3:
            case 5:
            case 6:
                addRandomShape(p, (_SETTINGS->stage.min_obstacle_size / 2),
                               (_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                ;
                printf("obstacle point added randomShape\n");
                existing_positions.push_back(new b2Vec2(p.x, p.y));
                break;
            case 2:
                random_length =  f_frandomRange(0.4,1.5);
                random_width = f_frandomRange(0.45,0.6);
                generateRandomBodyHorizontal(p, (_SETTINGS->stage.min_obstacle_size / 2),
                                             random_length*(_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                printf("obstacle point added horizontal 1\n");
                existing_positions.push_back(new b2Vec2(p.x, p.y));

                p.x = p.x;//1.5 + _SETTINGS->stage.min_obstacle_size / 2;
                p.y = p.y+ random_width;//0.45;//-1.95;
                generateRandomBodyHorizontal(p, _SETTINGS->stage.min_obstacle_size / 2,
                                             random_length*(_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                printf("obstacle point added horizontal 2\n");
                existing_positions.push_back(new b2Vec2(p.x, p.y));

                break;
            case 4:
                random_length =  f_frandomRange(0.4,1.5);
                random_width = f_frandomRange(0.45,0.6);
                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
                                           random_length*(_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                printf("obstacle point added vertical 1\n");
                existing_positions.push_back(new b2Vec2(p.x, p.y));

                p.x = p.x - random_width;//- 0.45;
                p.y = p.y;
                generateRandomBodyVertical(p, _SETTINGS->stage.min_obstacle_size / 2,
                                           random_length*(_SETTINGS->stage.max_obstacle_size / 2), &aabb);
                printf("obstacle point added vertical 2\n");
                existing_positions.push_back(new b2Vec2(p.x, p.y));
                break;
        }
    }
    printf("all obstacles added\n");
    // spawning dynamic obstacles
    _goalSpawnArea.addQuadTree(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE,
                               LEVEL_CUSTOM_GOAL_SPAWN_AREA_BLOCK_SIZE, half_goal_size);
    _goalSpawnArea.calculateArea();

    printf("spawn wanderers\n");
    if (_dynamic) {
        printf("_dynamicSpawn.clear()\n");
        _dynamicSpawn.clear();
        printf("addCheeseRect\n");
        _dynamicSpawn.addCheeseRect(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE | COLLIDE_CATEGORY_PLAYER,
                                    dynamic_radius);
        printf("calculateArea\n");
        _dynamicSpawn.calculateArea();
        for (int i = 0; i < num_dynamic_obstacles; i++) {
            printf("spawn wanderer\n");
            b2Vec2 wanderer_p;
            _dynamicSpawn.getRandomPoint(wanderer_p);
            WandererBipedal *w = new WandererBipedal(_levelDef.world, wanderer_p, dynamic_speed, 0.1, 0.05);
            _wanderers.push_back(w);

        }
    }
    printf("wanderers spawned\n");
    // adding spawn area
    randomGoalSpawnUntilValid();
    printf("goal spawned\n");
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

void LevelCustom::freeWanderers() {
    for (std::list<WandererBipedal *>::iterator it = _wanderers.begin(); it != _wanderers.end(); it++) {
        delete (*it);
    }
    _wanderers.clear();
}

void LevelCustom::update() {
    b2Vec2 position;
    for (std::list<WandererBipedal *>::iterator it = _wanderers.begin(); it != _wanderers.end(); it++) {
        bool chat_flag = false;
        float radius_check = 0.3;
        float move_on = f_frandomRange(0.1, 2);
        for (std::list<WandererBipedal *>::iterator it2 = _wanderers.begin(); it2 != _wanderers.end(); it2++) {
            if((*it) == (*it2)){
                continue;
            }
            float distance = ((*it)->getPosition() - (*it2)->getPosition()).Length();
            //printf("Radius %f \n", (*it)->getRadius());

            if ( distance < radius_check) {
                chat_flag = true;
                break;
            }
        }

        position = (*it)->getPosition();

        b2Vec2 new_position = position;
        if(abs(position.x) > 2 || abs(position.y) > 2 ) {
            if((position.x) > 2 && abs(position.y) < 2 ) {
                new_position.x = 1.99;
            }else if((position.x) < -2 && abs(position.y) < 2){
                new_position.x = -1.99;
            }else if((position.y) < -2 && abs(position.x) < 2){
                new_position.y = -1.99;
            }else if((position.y) > 2 && abs(position.x) < 2){
                new_position.y = 1.99;
            }else{
                _dynamicSpawn.getRandomPoint(new_position);
            }

            (*it)->setPosition(new_position);







        }
        (*it)->update(chat_flag);

    }

    }



void LevelCustom::renderGoalSpawn() {
    Level::renderGoalSpawn();
    Z_SHADER->setColor(zColor(0.1, 0.9, 0.0, 0.5));
    _dynamicSpawn.render();
}
