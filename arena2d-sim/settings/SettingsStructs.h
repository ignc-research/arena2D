//NOTE: This file will be loaded by the settings loader, do not add any definitions, do not add #include directives! Only declaration of structs and basic variables (int, float, string, SDL_Keycode) are allowed! Only use single line comments //!

typedef struct{
	int resolution_w;	// window width
	int resolution_h;	// window height
	int window_x;		// window x position on startup -1 for centered
	int window_y;		// window y position on startup -1 for centered
	int maximized;		// start window maximized 1/0
	int msaa;			// multisampling anti-aliasing 2,4,8,16
	int vsync;			// vertical synchronization 1/0
	int fps; 			// video frames per second
	int fullscreen; 	// fullscreen enabled 1/0
	int enabled; 		// video mode enabled 1/0; if set to 0
}f_videoSettings;

typedef struct{
	int font_size; 				// gui font size
	int show_robot; 			// show/hide the robot
	int show_stage; 			// show/hide the stage
	int show_laser; 			// show/hide the laser
	int show_stats; 			// show/hide the statistics (e.g. #episodes)
	int show_goal;				// show/hide goal
	int show_goal_spawn;		// show/hide the spawn area for the goal
	int show_trail; 			// show/hide robot trail
	int camera_follow;			// if == 1 camera follows robot, (if == 2, rotation is also taken into acount)
	float camera_x; 			// initial position of camera
	float camera_y; 			// initial position of camera
	float camera_zoom; 			// initial zoom of camera (zoom actually means scaling of the view -> camera_zoom < 1 means zooming out)
	float camera_rotation;		// view rotation in degree
	float camera_zoom_factor;	// how does scaling increase/decrease with each zoom step
}f_guiSettings;

typedef struct{
	SDL_Keycode up; 	// key for moving forward
	SDL_Keycode left;	// key for moving left
	SDL_Keycode down;	// key for moving backward
	SDL_Keycode right;	// key for moving right
	SDL_Keycode reset;	// key for resetting robot
	SDL_Keycode play_pause_simulation; // key for playing/pausing simulation
}f_keymapSettings;

typedef struct{
	float time_step;		// physics time step
	int step_iterations;	// how often to perform a physics update per step
	int fps;				// how many times per second a simulation step is performed with step_iterations sub steps
	int position_iterations;// position iterations for each time step (higher value increases simulation accuracy)
	int velocity_iterations;// velocity iterations for each time step (higher value increases simulation accuracy)
}f_physicsSettings;

typedef struct{
	int random_seed; 			// seed for pseudo random generator
	string initial_level; 		// level loaded on startup (-1 for none)
	float level_size; 			// width and height of default levels
	float max_obstacle_size;	// maximum diameter of static obstacles
	float min_obstacle_size;	// minimum diameter of static obstacles
	int num_obstacles; 			// number of static obstacles
	float dynamic_obstacle_size;// size of dynamic obstacle
	int num_dynamic_obstacles;	// number of dynamic obstacles in static_dynamic level
	float obstacle_speed;		// in m/s for dynamic obstacles
	float goal_size;			// diameter of circular goal to reach
	string svg_path;			// path to folder where svg files are stored
}f_stage;

typedef struct{
	float max_time;				// maximum time per episode (actual time, so physics.time_step influences maximum number of steps per episode)
	int episode_over_on_hit; 	// if set to 1 episode ends if an obstacle is hit
	float reward_goal;			// reward for reaching goal
	float reward_towards_goal;	// reward when distance to goal decreases
	float reward_away_from_goal;// reward when distance to goal increases
	float reward_hit;			// reward for hitting obstacle
	float reward_time_out;		// reward when episode timed out (after max_time seconds)
	int num_envs;				// number of parallel environments
	int num_threads;			// number of threads to run in parallel, if set to -1 number of cpu cores will be detected automatically
	string agent_class;			// name of class in agent python script
}f_trainingSettings;

typedef struct{
	float linear;
	float angular;
} f_twist;

typedef struct{
	float x;
	float y;
}f_vec2;

typedef struct{
	float laser_noise; 			// random, uniformly distributed offset with a maximum of +/- laser_noise*distance_measured (a value of 0 means perfect laser data -> no noise)
	float laser_max_distance; 	// maximum distance the laser can recognize
	float laser_start_angle; 	// angle in degree of first sample
	float laser_end_angle; 		// angle in degree of last sample
	int laser_num_samples;		// number of laser samples
	f_vec2 laser_offset;		// offset of laser from base center
	f_vec2 base_size;			// width(x) and height(y) of robot base
	f_vec2 wheel_size;			// width(x) and height(y) of robot wheels
	f_vec2 wheel_offset;		// offset of wheels from edge of base
	f_vec2 bevel_size;			// size of bevel along x/y axis at the base corners
	f_twist forward_speed;		// velocity for forward action
	f_twist left_speed;			// velocity for left action
	f_twist right_speed;		// velocity for right action
	f_twist strong_left_speed;	// velocity for strong left action
	f_twist strong_right_speed; // velocity for strong right action
	f_twist backward_speed;		// velocity for backward action
}f_robotSettings;


typedef struct{
	f_videoSettings video;
	f_guiSettings gui;
	f_keymapSettings keys;
	f_physicsSettings physics;
	f_trainingSettings training;
	f_robotSettings robot;
	f_stage stage;
}f_settings;
