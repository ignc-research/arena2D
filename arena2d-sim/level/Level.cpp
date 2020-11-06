/* Author: Cornelius Marx */

#include "Level.hpp"

Level::Level(const LevelDef &d) : _levelDef(d), _goal(NULL) {
}

Level::~Level() {
    clear();
    // free goal
    if (_goal != NULL)
        _levelDef.world->DestroyBody(_goal);
}

void Level::clear() {
    // free all bodies that belong to this level
    for (std::list<b2Body *>::iterator it = _bodyList.begin(); it != _bodyList.end(); it++) {
        _levelDef.world->DestroyBody(*it);
    }
    _bodyList.clear();

    // clear goal spawn
    _goalSpawnArea.clear();
}

void Level::spawnGoal(const b2Vec2 &pos) {
    if (_goal != NULL) {
        _levelDef.world->DestroyBody(_goal);
    }
    b2BodyDef def;
    def.type = b2_staticBody;
    def.position = pos;
    _goal = _levelDef.world->CreateBody(&def);
    b2CircleShape c;
    c.m_p.Set(0, 0);
    c.m_radius = _SETTINGS->stage.goal_size / 2.f;
    b2FixtureDef fix;
    fix.isSensor = true;
    fix.filter.categoryBits = COLLIDE_CATEGORY_GOAL;
    fix.shape = &c;
    _goal->CreateFixture(&fix);
}

b2Body *Level::createBorder(float half_width, float half_height) {
    b2Vec2 v[4];
    v[0].Set(-half_width, half_height);
    v[1].Set(half_width, half_height);
    v[2].Set(half_width, -half_height);
    v[3].Set(-half_width, -half_height);
    b2ChainShape chain;
    chain.CreateLoop(v, 4);
    return addShape(&chain);
}

b2Body *Level::addCircle(const b2Vec2 &pos, float radius) {
    b2CircleShape circle;
    circle.m_p = pos;
    circle.m_radius = radius;
    return addShape(&circle);
}

b2Body *Level::addBox(const b2Vec2 &center_pos, float half_width, float half_height, float angle) {
    b2PolygonShape box;
    box.SetAsBox(half_width, half_height, center_pos, angle);
    return addShape(&box);
}

b2Body *Level::addShape(const b2Shape *s) {
    b2BodyDef b;
    b.type = b2_staticBody;
    b2Body *body = _levelDef.world->CreateBody(&b);
    b2FixtureDef f;
    f.shape = s;
    f.friction = LEVEL_STATIC_FRICTION;
    f.restitution = LEVEL_STATIC_RESTITUTION;
    f.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
    body->CreateFixture(&f);
    _bodyList.push_back(body);
    return body;
}

b2Body *Level::addShape(const std::vector<b2Shape *> shapes) {
    if (shapes.size() == 0)// no shapes to add
        return NULL;

    // create a single body, add multiple fixtures
    b2BodyDef b;
    b.type = b2_staticBody;
    b2Body *body = _levelDef.world->CreateBody(&b);
    b2FixtureDef f;
    f.friction = LEVEL_STATIC_FRICTION;
    f.restitution = LEVEL_STATIC_RESTITUTION;
    f.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
    for (unsigned int i = 0; i < shapes.size(); i++) {
        f.shape = shapes[i];
        body->CreateFixture(&f);
    }
    _bodyList.push_back(body);
    return body;
}

//corridor spawning avoid center spawning at same center like other obstacles
bool Level::obstacleSpawnUntilValid(RectSpawn *static_spawn, const std::list<b2Vec2 *> &existing_positions, b2Vec2 &p) {
    bool spawn_found = false;
    //printf("obstacleSpawnUntilValid\n");
    int count = 0;
    while (!spawn_found && count < 200) {
        count++;
        spawn_found = true;
        static_spawn->getRandomPoint(p);
        bool boundary_cond = true;
        while (boundary_cond) {
            if (p.y < 1.55 && p.x > -1.2 && p.y > -1 && p.x > -1.4) {
                boundary_cond = false;
            } else {
                static_spawn->getRandomPoint(p);
            }
        }
        for (auto &existing_position : existing_positions) {
            b2Vec2 posi(existing_position->x, existing_position->y);
            if ((posi - p).Length() < _SETTINGS->stage.max_obstacle_size + 0.5) {
                spawn_found = false;
                break;
            }
        }
    }
    if (!spawn_found) {
        printf("##################### No place found after 100 TRIES ############\n");
        printf("##################### No place found after 100 TRIES ############\n");
        printf("##################### No place found after 100 TRIES ############\n");
        printf("##################### No place found after 100 TRIES ############\n");
    }
    return spawn_found;
}





bool Level::obstacleSpawnUntilValid(RectSpawn *static_spawn, const std::list<zRect *> &existing_boxes, b2Vec2 &p,
                                    int obstacle_type, float robot_diameter) {
    int scale_factor = 2.5;
    bool spawn_found = false;
    //printf("obstacleSpawnUntilValid\n");
    int count = 0;
    float max_random_size = (_SETTINGS->stage.max_obstacle_size / 2);
    float max_horizontal_width = scale_factor * (_SETTINGS->stage.max_obstacle_size / 2);
    float max_horizontal_height = scale_factor * _levelDef.robot->getRadius() * 2;
    float max_vertial_height = scale_factor * (_SETTINGS->stage.max_obstacle_size / 2);
    float max_vertical_width = scale_factor * _levelDef.robot->getRadius() * 2;
    while (!spawn_found && count < 200) {
        count++;
        spawn_found = true;
        static_spawn->getRandomPoint(p);
        bool boundary_cond = true;
        while (boundary_cond) {
            zRect center;
            if (obstacle_type == 2) {
                center.x = 0 + max_horizontal_width/2.0f;
                center.y = 0;
                center.w = max_horizontal_width/2.0f + 0.4f + robot_diameter;
                center.h = 0.9f + robot_diameter;
            } else if (obstacle_type == 4 ) {
                center.x = 0;
                center.y = 0 + max_vertial_height/2.0f;
                center.w = 0.9f + robot_diameter;
                center.h = max_vertial_height/2.0f + 0.4f + robot_diameter;
            } else {
                center.x = 0;
                center.y = 0;
                center.w = 0.9f+ robot_diameter;
                center.h = 0.9f+ robot_diameter;
            }

            zVector2D pointi;
            pointi.x = p.x;
            pointi.y = p.y;
            if (!center.checkPoint(pointi) &&
            p.y < 1.55 && p.x > -1.2 && p.y > -1 && p.x > -1.4) {
                boundary_cond = false;
            } else {
                static_spawn->getRandomPoint(p);
            }
        }
        //if (obstacle_type == 2 || obstacle_type == 4 )  {
            for (auto &existing_box : existing_boxes) {
                if (obstacle_type == 2) {
                    //horizontal
                    zRect hori;
                    hori.x = p.x - max_horizontal_width/2.0f;
                    hori.y = p.y + max_horizontal_height/2.0f;
                    hori.w = max_horizontal_width;
                    hori.h = max_horizontal_height;
                    if (zRect::intersect(hori, *existing_box)) {
                        spawn_found = false;
                        break;
                    }
                } else if (obstacle_type == 4) {
                    //vertical
                    zRect verti;
                    verti.x = p.x - max_vertical_width/2.0f;
                    verti.y = p.y - max_vertial_height/2.0f;
                    verti.w = max_vertical_width;
                    verti.h = max_vertial_height;
                    if (zRect::intersect(verti, *existing_box)) {
                        spawn_found = false;
                        break;
                    }
                } else {
                    zRect randi;
                    randi.x = p.x + max_random_size/2.0f;
                    randi.y = p.y + max_random_size/2.0f;
                    randi.w = max_random_size;
                    randi.h = max_random_size;
                    if (zRect::intersect(randi, *existing_box)) {
                        spawn_found = false;
                        break;
                    }
                }
//                if (p.x <= existing_box->x + existing_box->w && p.x >= existing_box->x &&
//                    p.y <= existing_box->y + existing_box->h && p.y >= existing_box->y) {
//                    spawn_found = false;
//                    break;
//                }
           // }
        }
    }
    return spawn_found;
}




void Level::randomGoalSpawnUntilValidForMaze(RectSpawn * goal_spawn)
{
	RectSpawn * spawn = &_goalSpawnArea;
	if(goal_spawn != NULL)// use custom goal spawn
	{
		spawn = goal_spawn;
	}

	b2Vec2 robot_position = _levelDef.robot->getPosition();
	// spawn goal at random position
	b2Vec2 spawn_position(0,0);
	int count = 0;
	do{
		spawn->getRandomPoint(spawn_position);
		count++;
	}while(!checkValidGoalSpawn_Walls(robot_position,spawn_position) && count < 100);
	spawnGoal(spawn_position);
}

void Level::randomGoalSpawnUntilValid(RectSpawn * goal_spawn)
{
	RectSpawn * spawn = &_goalSpawnArea;
	if(goal_spawn != NULL)// use custom goal spawn
	{
		spawn = goal_spawn;
	}

	b2Vec2 robot_position = _levelDef.robot->getPosition();
	// spawn goal at random position
	b2Vec2 spawn_position(0,0);
	int count = 0;
	do{
		spawn->getRandomPoint(spawn_position);
		count++;
	}while(!checkValidGoalSpawn(robot_position, spawn_position) && count < 10);
	spawnGoal(spawn_position);

}

b2Body *Level::addRandomShape(const b2Vec2 &position, float min_radius, float max_radius, zRect *aabb) {
    int vert_count = f_irandomRange(3, 6);
    b2PolygonShape shape;
    shape.m_count = vert_count;
    b2Vec2 verts[8];
    float radius_x = f_frandomRange(min_radius, max_radius);
    float radius_y = f_frandomRange(min_radius, max_radius);
    b2Vec2 max_v(-10000, -10000);
    b2Vec2 min_v(10000, 10000);
    float rotation = f_frandomRange(0, 2 * M_PI);
    for (int i = 0; i < vert_count; i++) {
        float angle = M_PI * 2 * (i / static_cast<float>(vert_count));
        verts[i].Set(cos(angle + rotation) * radius_x, sin(angle + rotation) * radius_y);
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
        verts[i] += position;
    }

    if (aabb != NULL) {
        aabb->x = position.x;
        aabb->y = position.y;
        aabb->w = (max_v.x - min_v.x) / 2.0f;
        aabb->h = (max_v.y - min_v.y) / 2.0f;
    }
    shape.Set(verts, vert_count);
    return addShape(&shape);
}



void Level::renderGoalSpawn()
{
	Z_SHADER->setColor(zColor(LEVEL_GOAL_SPAWN_COLOR));
	_goalSpawnArea.render();

}
