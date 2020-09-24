/* Author: Cornelius Marx */
#include "Environment.hpp"

// EnvironmentThread
EnvironmentThread::EnvironmentThread(){
	state = WAITING;
	state_mutex = SDL_CreateMutex();
	state_cond = SDL_CreateCond();
	thread = SDL_CreateThread(EnvironmentThread::thread_func, "", (void*)this);
}

EnvironmentThread::~EnvironmentThread(){
	// terminate thread
	SDL_LockMutex(state_mutex);
	state = EXIT;
	SDL_UnlockMutex(state_mutex);
	SDL_CondSignal(state_cond);
	SDL_WaitThread(thread, NULL);

	SDL_DestroyMutex(state_mutex);
	SDL_DestroyCond(state_cond);
}

void EnvironmentThread::init(Environment * _env, int _num_envs, int _env_index, const Twist * _action){
	env = _env;
	num_envs = _num_envs;
	env_index = _env_index;
	action = _action;
}

void EnvironmentThread::step(){
	SDL_LockMutex(state_mutex);
	state = RUNNING;
	SDL_UnlockMutex(state_mutex);
	SDL_CondSignal(state_cond);
}

void EnvironmentThread::wait_finish(){
	SDL_LockMutex(state_mutex);
	while(state == RUNNING){
		SDL_CondWait(state_cond, state_mutex);
	}
	SDL_UnlockMutex(state_mutex);
}

int EnvironmentThread::thread_func(void * d){
	EnvironmentThread * data = (EnvironmentThread*)d;
	while(1){
		SDL_LockMutex(data->state_mutex);
		// wait for signal from main thread
		while(data->state == EnvironmentThread::WAITING){
			SDL_CondWait(data->state_cond, data->state_mutex);
		}

		// exit thread if state has been set to EXIT
		if(data->state == EnvironmentThread::EXIT){
			SDL_UnlockMutex(data->state_mutex);
			break;
		}

		// calculate steps
		for(int i = data->env_index; i < data->env_index+data->num_envs; i++){
			data->env[i].stepAll(data->action[i]);
		}

		// send signal done
		data->state = EnvironmentThread::WAITING;
		SDL_UnlockMutex(data->state_mutex);
		SDL_CondSignal(data->state_cond);
	}
	return 0;
}
/////// EnvironmentThread


Environment::Environment()
{
	_world = new b2World(b2Vec2(0,0));
	_world->SetContactListener((b2ContactListener*)this);
	_level = NULL;
	_robot = new Robot(_world);

	refreshSettings();
}

void Environment::refreshSettings()
{
	// copy training settings to avoid read conflicts on simultanious reads
	_physicsSettings = _SETTINGS->physics;
	_trainingSettings = _SETTINGS->training;

	// update laser scanner
	_robot->updateLidar();
	_robot->scan();
}

Environment::~Environment()
{
	delete _robot;
	delete _level;
	delete _world;
}

int Environment::loadLevel(const char * level_name, const ConsoleParameters & params)
{
	// free old level 
	delete _level;
	_level = LEVEL_FACTORY->createLevel(level_name, LevelDef(_world, _robot), params);
	
	initializeTraining();

	if(_level == NULL){
		return -1;
	}
	return 0;
}

void Environment::initializeTraining()
{
	_totalReward = 0;
	_episodeCount = 0;
	_robot->reset(b2Vec2(0,0));
	reset(true);
}

void Environment::pre_step(const Twist & t)
{
	_reward = 0.0f;
	_action = t;
	getGoalDistance(_distance, _angle);
}

void Environment::step()
{
	if(_episodeState != RUNNING)// episode over, has to be reset first
		return;
	_robot->performAction(_action);
	_world->Step(_physicsSettings.time_step, _physicsSettings.velocity_iterations, _physicsSettings.position_iterations);
	_episodeTime += _physicsSettings.time_step;
	// time's up
	if(	_episodeTime > _trainingSettings.max_time &&
		_trainingSettings.max_time > 0.f && _level != NULL){
		_reward += _SETTINGS->training.reward_time_out;
		_episodeState = NEGATIVE_END;
	}
}

void Environment::stepAll(const Twist & t)
{
	pre_step(t);
	for(int i = 0; i < _physicsSettings.step_iterations; i++){
		step();
	}
	post_step();
}

void Environment::post_step()
{
	_episodeStepCount++;
	float distance_after = 0.f;
	float angle_after = 0.f;
	getGoalDistance(distance_after, angle_after);

	// checking reward for distance to goal decreased/increased
	if(distance_after < _distance){
		_reward += _trainingSettings.reward_towards_goal;
	}
	else{
		_reward += _trainingSettings.reward_away_from_goal;
	}

	// updating level logic
	if(_level != NULL){
		_level->update();
		_reward += _level->getReward();
	}

	// update laser scan
	_robot->scan();


	_totalReward += _reward;
}

void Environment::getGoalDistance(float & l2, float & angle)
{
	if(_level != NULL && _level->getGoal() != NULL){
		b2Vec2 goal_pos = _level->getGoalPosition();
		goal_pos = _robot->getBody()->GetLocalPoint(goal_pos);
		l2 = goal_pos.Length();
		angle = f_deg(zVector2D::signedAngle(zVector2D(0, 1), zVector2D(goal_pos.x, goal_pos.y)));
	}
}

void Environment::reset(bool robot_position_reset)
{
	// reset level
	if(_level != NULL)
		_level->reset(_episodeState == NEGATIVE_END || robot_position_reset);
	
	// reset trail
	if(_SETTINGS->video.enabled)
		_robot->resetTrail();

	_episodeStepCount = 0;
	_robot->scan();// initial observation
	_episodeState = RUNNING;
	_episodeTime = 0.f;
	_totalReward = 0.f;
}

void Environment::BeginContact(b2Contact * contact){
	if(_episodeState != RUNNING)// already episode over -> goal reached, nothing to check
		return;

	b2Fixture * a = contact->GetFixtureA();
	b2Fixture * b = contact->GetFixtureB();
	const b2Fixture * burger = _robot->getRadiusSensor();
	if(_level != NULL){
		b2Fixture * goal = _level->getGoal();
		b2Fixture * other_fix = NULL;
		if(a == burger){
			other_fix = b;
		}
		else if(b == burger){
			other_fix = a;
		}
		if(other_fix != NULL){
			// end of episode
			if(other_fix == goal){// goal reached
				_reward += _SETTINGS->training.reward_goal;
				_episodeState = POSITIVE_END;
			}
			else if(_robot->beginContact()){// wall hit
				_reward += _SETTINGS->training.reward_hit;
				if(_SETTINGS->training.episode_over_on_hit)
					_episodeState = NEGATIVE_END;
			}
		}
	}
}

void Environment::EndContact(b2Contact * contact)
{
	const b2Fixture * burger_sensor = _robot->getRadiusSensor();
	if(burger_sensor == contact->GetFixtureA() || burger_sensor == contact->GetFixtureB()){
		_robot->endContact();
	}
}

void Environment::render(const Camera & c, const zRect & global_aabb)
{
	zRect aabb = global_aabb;
	Camera cam = c;
	// set colorplex shader
	_RENDERER->useColorplexShader();
	_RENDERER->setGLMatrix();
	_RENDERER->resetModelviewMatrix();

	// camera follow
	if( _SETTINGS->gui.camera_follow){
		b2Transform t = _robot->getBody()->GetTransform();
		if(_SETTINGS->gui.camera_follow == 2){
			cam.setRotation(t.q.GetAngle()-M_PI/2.0f);
		}
		cam.setPos(zVector2D(t.p.x, t.p.y));
		cam.refresh();
		cam.upload();
		Quadrangle q;
		_RENDERER->getTransformedGLRect(cam.getInverseMatrix(), &q);
		q.getAABB(&aabb);
	}
	else{
		cam.upload();
	}

	// update burger trail
	_robot->updateTrail();

	_PHYSICS->calculateVisibleFixturesWorld(_world, aabb);


	if(_SETTINGS->gui.show_trail)
		_robot->renderTrail();
	
	uint16 category = 0x0001;
	if(_SETTINGS->gui.show_robot)
		category |= COLLIDE_CATEGORY_PLAYER; 


	_PHYSICS->setDynamicColor(zColor(0x3A87E1FF));
	_PHYSICS->debugDraw(PHYSICS_RENDER_ALL, category);

	category = 0;

	if(_SETTINGS->gui.show_goal){
		category |= COLLIDE_CATEGORY_GOAL;
	}

	if(_SETTINGS->gui.show_stage){
		category |= COLLIDE_CATEGORY_STAGE;
	}

	if(category){
		_PHYSICS->setDynamicColor(zColor(0x5ACB79FF));
		_PHYSICS->debugDrawWorld(_world, PHYSICS_RENDER_ALL, category);
	}

	// use color shader
	_RENDERER->useColorShader();
	_RENDERER->setGLMatrix();
	cam.upload();
	_RENDERER->resetModelviewMatrix();

	//draw laser scanner
	if(_SETTINGS->gui.show_laser){
		_robot->renderScan(_SETTINGS->gui.show_laser > 1);
	}

	// rendering goal spawn area
	if(_SETTINGS->gui.show_goal_spawn && _level != NULL){
		_level->renderGoalSpawn();
	}
}
