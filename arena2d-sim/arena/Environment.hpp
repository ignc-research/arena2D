/* Author: Cornelius Marx */

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <engine/zFramework.hpp>
#include <engine/GlobalSettings.hpp>
#include <engine/Camera.hpp>
#include "Robot.hpp"
#include "MeanBuffer.hpp"
#include "ConsoleParameters.hpp"
#include <level/LevelFactory.hpp>

/* forward declaration */
class Environment;

// EnvironmentThread representing thread that performs steps on multiple Environments
struct EnvironmentThread{
	/* thread state */
	enum State{	WAITING,
				RUNNING,
				EXIT};

	/* constructor */
	EnvironmentThread();

	/* destructor */
	~EnvironmentThread();

	/* thread_function */
	static int thread_func(void * data);

	/* initialize environment thread */
	void init(Environment * _env, int _num_envs, int _env_index, const Twist * _action);

	/* perform step in environments */
	void step();

	/* blocking till all environments finished */
	void wait_finish();

	/* array of all environments */
	Environment * env;

	/* array of actions to be performed in environments */
	const Twist * action;

	/* number of environments this thread is responsible for */
	int num_envs;

	/* index in env array of first environment for this thread */
	int env_index;

	/* thread state */
	State state;

	/* mutex for securing state */
	SDL_mutex * state_mutex;

	/* conditional to signal changing of state */
	SDL_cond * state_cond;

	/* sdl thread */
	SDL_Thread * thread;
};

/* abstraction to allow for multiple instanciations of level and robot */
class Environment : public b2ContactListener
{
public:
	/* episode state */
	enum EpisodeState{	RUNNING,		// episode is still running
						POSITIVE_END,	// episode is over, goal reached
						NEGATIVE_END	// episode is over, timeout or obstacle hit
	};

	/* constructor */
	Environment();

	/* destructor */
	~Environment();

	/* load level with given name
	 * @param level_name name of the level to load
	 * @param params parameter to pass to level specific create function
	 * @return 0 on success, -1 on error
	 */
	int loadLevel(const char* level_name, const ConsoleParameters & params);

	/* perform pre_step, all step iterations and post_step
	 * @param action the action to perform on each step iteration
	 */
	void stepAll(const Twist & action);

	/* override functions from b2ContactListener
	 */
	void BeginContact(b2Contact * contact) override;
	void EndContact(b2Contact * contact) override;

	/* reset environment
	 * @param robot_position_reset passed to level reset function
	 */
	void reset(bool robot_position_reset);

	/* fetch local copy of global training and physics settings 
	 * this is done if user changes settings during runtime 
	 */
	void refreshSettings();

	/* get episode state
	 * @return enum indicating episode state
	 */
	EpisodeState getEpisodeState(){return _episodeState;}

	/* get physics world of this environment
	 * @return Box2D world
	 */
	b2World* getWorld(){return _world;}

	/* render environment
	 * viewport, camera-/projection-matrix must be set
	 */
	void render(const Camera & cam, const zRect & aabb);

	/* get distance to goal
	 * @param l2 is set to the l2 distance from robot to goal
	 * @param angle is set to the angle (degree) from the robot's facing direction to goal
	 */
	void getGoalDistance(float & l2, float & angle);

	/* get scan observation
	 * @param num_samples is set to the number of samples in the returned array
	 * @return array containing laser samples (distance to obstacles)
	 */
	const float* getScan(int & num_samples){return _robot->getSamples(num_samples);}

	/* get reward from current step
	 * @return reward
	 */
	float getReward(){return _reward;}

	/* get total reward from current episode
	 * @return total reward
	 */
	float getTotalReward(){return _totalReward;}

	/* initialize training
	 */
	void initializeTraining();

	/* get level currently loaded in environment
	 * @return current level
	 */
	 Level* getLevel(){return _level;}

private:
	/* prepare simulation step
	 * @param action the action to perform on simulation step
	 */
	void pre_step(const Twist & action);

	/* perform single simulation step
	 */
	void step();// do single step iteration

	/* calculate rewards after simulation step
	 */
	void post_step();

	/* physics world */
	b2World * _world;

	/* copy of global physics settings */
	f_physicsSettings _physicsSettings;

	/* copy of global training settings */
	f_trainingSettings _trainingSettings;// copy of settings to avoid read conflicts

	/* level currently loaded in environment */
	Level * _level;

	/* robot controlled by agent/user */
	Robot * _robot;

	/* action to be performed on next step */
	Twist _action;

	/* distance to goal before step */
	float _distance;

	/* angle to goal before step */
	float _angle;

	/* reward gained from last step */
	float _reward;

	/* accumulated reward for one episode */
	float _totalReward; 

	/* simulated time of current episode in seconds */
	float _episodeTime; 

	/* number of steps performed for current episode */
	int _episodeStepCount;

 	/* number of complete episodes from this environment since training start */
	int _episodeCount;

	/* state of episode */
	EpisodeState _episodeState;
};

#endif

