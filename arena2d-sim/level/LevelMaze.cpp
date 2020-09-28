/* Author: Deyu Wang( modified by Junhui Li)
 * 1. This code is generated from the old version from gitlab.
 * 2. There will be 5 walls randomly generated in the maze.
 * 3. Because the longest wall will be generated at center of the stage, the robot will spawn in the bottom left,
 * which means the "resetRobotToCenter()" function in "Level.hpp" is changed.
 * The new spawn point now is (-0.5, -0.5)
 * 4. The radius of the statistic obstacles are also decreased, otherwise the way could be blocked.
 */


#include "LevelMaze.hpp"

void LevelMaze::reset()
{
	// clear old bodies and spawn area
	clear();
	if(_dynamic)
		freeWanderers();

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
       /* safe radius of circle from center of robot that encloses all robot parts (base and wheels)*/
	const float Burger_safe_Radius = _levelDef.robot->getRadius();

	// create border around level
	createBorder(half_width, half_height);

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

	//***************************************************** step2. add the 4 normal walls in 4 specific areas ******************************************************************
	for(int i=8; i<12; i++){
		    zRect aabb;
		    generateRandomWalls11(i, &aabb);
		    
		}

	//***************************************************** step3. add a long wall in the mid area *************************************************************************
        int Random_Index = f_irandomRange(0, 1);  //generate a random index
        zRect aabb1;
        generateRandomWalls22(Random_Index, 12, &aabb1);
        

	// calculating goal spawn area
	_goalSpawnArea.addQuadTree(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE,
								LEVEL_RANDOM_GOAL_SPAWN_AREA_BLOCK_SIZE, half_goal_size);
	_goalSpawnArea.calculateArea();


	// dynamic obstacles
	if(_dynamic){
		_dynamicSpawn.clear();
		_dynamicSpawn.addCheeseRect(main_rect, _levelDef.world, COLLIDE_CATEGORY_STAGE | COLLIDE_CATEGORY_PLAYER, dynamic_radius);
		_dynamicSpawn.calculateArea();
		for(int i = 0; i < num_dynamic_obstacles; i++){
			b2Vec2 p;
			_dynamicSpawn.getRandomPoint(p);
			Wanderer * w = new Wanderer(_levelDef.world, dynamic_radius, p, dynamic_speed, 0.1, 0.0, 0);			
			_wanderers.push_back(w);
		}
	}

	randomGoalSpawnUntilValidForMaze();

}

void LevelMaze::freeWanderers()
{
	for(std::list<Wanderer*>::iterator it = _wanderers.begin(); it != _wanderers.end(); it++){
		delete (*it);
	}
	_wanderers.clear();
}

void LevelMaze::update()
{
	for(std::list<Wanderer*>::iterator it = _wanderers.begin(); it != _wanderers.end(); it++){
		(*it)->update();
	}

}

void LevelMaze::renderGoalSpawn()
{
	Level::renderGoalSpawn();
	Z_SHADER->setColor(zColor(0.1, 0.9, 0.0, 0.5));
	_dynamicSpawn.render();
}


// functions to generate four short walls in four specific areas
b2Body* LevelMaze::generateRandomWalls11(int index, zRect * aabb){
	b2BodyDef def;
	def.type = b2_staticBody;
	b2Body * b = _levelDef.world->CreateBody(&def);
	def.allowSleep = false;
	def.linearDamping = 0;
	def.angularDamping = 0;
	index = index - 8;
	int angleRandom_Index = f_irandomRange(0, 1);                   //generate a random coefficient for the angle of the walls
	int moveRandom_Index = f_irandomRange(0, 1);                    //generate a random coefficient for the shift of the walls's center point
	b2Vec2 center_points[4] = {b2Vec2(-1, -1), b2Vec2(1, -1), b2Vec2(-1, 1), b2Vec2(1, 1)};
	b2PolygonShape Wall_shape;
	float rotation[2] = {0, M_PI/2.f};                               // two directions' variants, vertical or horizontal
	float shift[2] = {0.5, -0.5};                                    // two shifts, that will be added to the coordinates(either in x or y direction)
	if(angleRandom_Index==0){                                        // according to the direction of the wall, the final center point will be determined after being shifted
            center_points[index].x += shift[moveRandom_Index];
	}
	else{
	      center_points[index].y += shift[moveRandom_Index];
	}
	Wall_shape.SetAsBox(0.5, 0.025, center_points[index], rotation[angleRandom_Index]);    //create the wall
	if(aabb != NULL){
		aabb->x = center_points[index].x;
		aabb->y = center_points[index].y;


		if(angleRandom_Index==0){
			    aabb->w = 0.5;
			    aabb->h = 0.025;
		}
		else{
			    aabb->w = 0.025;
			    aabb->h = 0.5;
		}

	}

	b2FixtureDef fix;
	fix.shape = &Wall_shape;		
	fix.friction = LEVEL_STATIC_FRICTION;
	fix.restitution = LEVEL_STATIC_RESTITUTION;
	fix.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
	//fix.density = 1;
		
	b->CreateFixture(&fix);
	_bodyList.push_back(b);
	return b;
}


// function to generate a long wall
b2Body* LevelMaze::generateRandomWalls22(int index, int numm, zRect * aabb){
        b2BodyDef def;
	def.type = b2_staticBody;
	b2Body * b = _levelDef.world->CreateBody(&def);
	def.allowSleep = false;
	def.linearDamping = 0;
	def.angularDamping = 0;
	b2PolygonShape Wall_shape;
	b2Vec2 center_point = b2Vec2(0, 0);
        if(index==0){                                              //create a long wall, which is either vertical or horizontal
            Wall_shape.SetAsBox(1.0, 0.025, center_point, 0);
            if(aabb != NULL){
                aabb->x = center_point.x;
                aabb->y = center_point.y;
                aabb->w = 1.0;                                      // the half width is 0.8, here *2
                aabb->h = 0.025;                                     // the half height is 0.025, here *2
            }
	}
        else{                                                         // else if index==1
            Wall_shape.SetAsBox(1.0, 0.025, center_point, M_PI/2.f);
            if(aabb != NULL){
                aabb->x = center_point.x;
                aabb->y = center_point.y;
                aabb->w = 0.025;
                aabb->h = 1.0;
            }
        }
        b2FixtureDef fix;
	fix.shape = &Wall_shape;	
	fix.friction = LEVEL_STATIC_FRICTION;
	fix.restitution = LEVEL_STATIC_RESTITUTION;
	fix.filter.categoryBits = COLLIDE_CATEGORY_STAGE;
	//fix.density = 1;        
	b->CreateFixture(&fix);
	_bodyList.push_back(b);
	return b;
	}
