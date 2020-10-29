video{
	resolution_w = 800   # window width
	resolution_h = 600   # window height
	window_x = -1   # window x position on startup -1 for centered
	window_y = -1   # window y position on startup -1 for centered
	maximized = 0   # start window maximized 1/0
	msaa = 4   # multisampling anti-aliasing 2,4,8,16
	vsync = 0   # vertical synchronization 1/0
	fps = 60   # video frames per second
	fullscreen = 0   # fullscreen enabled 1/0
	enabled = 1   # video mode enabled 1/0; if set to 0
}

gui{
	font_size = 16   # gui font size
	show_robot = 1   # show/hide the robot
	show_stage = 1   # show/hide the stage
	show_laser = 1   # show/hide the laser
	show_stats = 1   # show/hide the statistics (e.g. #episodes)
	show_goal = 1   # show/hide goal
	show_goal_spawn = 0   # show/hide the spawn area for the goal
	show_trail = 1   # show/hide robot trail
	camera_follow = 0   # if == 1 camera follows robot, (if == 2, rotation is also taken into acount)
	camera_x = 0.0   # initial position of camera
	camera_y = 0.0   # initial position of camera
	camera_zoom = 0.45   # initial zoom of camera (zoom actually means scaling of the view -> camera_zoom < 1 means zooming out)
	camera_rotation = 0.0   # view rotation in degree
	camera_zoom_factor = 1.3   # how does scaling increase/decrease with each zoom step
}

keys{
	up = KEY_UP   # key for moving forward
	left = KEY_LEFT   # key for moving left
	down = KEY_DOWN   # key for moving backward
	right = KEY_RIGHT   # key for moving right
	reset = KEY_r   # key for resetting robot
	play_pause_simulation = KEY_SPACE   # key for playing/pausing simulation
}

physics{
	time_step = 0.1   # physics time step
	step_iterations = 1   # how often to perform a physics update per step
	fps = 60   # how many times per second a simulation step is performed with step_iterations sub steps
	position_iterations = 8   # position iterations for each time step (higher value increases simulation accuracy)
	velocity_iterations = 12   # velocity iterations for each time step (higher value increases simulation accuracy)
}

training{
	max_time = 100.0   # maximum time per episode (actual time, so physics.time_step influences maximum number of steps per episode)
	episode_over_on_hit = 0   # if set to 1 episode ends if an obstacle is hit
	reward_goal = 100.0   # reward for reaching goal
	reward_towards_goal = 0.1   # reward when distance to goal decreases
	reward_away_from_goal = -0.2   # reward when distance to goal increases
	reward_hit = -10.0   # reward for hitting obstacle
	reward_time_out = 0.0   # reward when episode timed out (after max_time seconds)
	num_envs = 4   # number of parallel environments
	num_threads = -1   # number of threads to run in parallel, if set to -1 number of cpu cores will be detected automatically
	agent_class = "Agent"   # name of class in agent python script
	reward_human = -100.0   # reward for hitting a human
	safety_distance_human = 0.5   # safty distance to human, which should be always fullfilled
	reward_distance_to_human_decreased = -0.3   # reward when distance to human decreases
	reward_distance_to_human_increased = 0.3   # reward when distance to human increases
	num_obs_humans = 3   # maximum number of humans the agent can observe inside the camera view
	reward_function = 1   # choose between different reward functions (1 = distance rewards only for observed humans, 2 = distance rewards for all humans in level)
}

robot{
	laser_noise = 0.0   # random, uniformly distributed offset with a maximum of +/- laser_noise*distance_measured (a value of 0 means perfect laser data -> no noise)
	laser_max_distance = 3.5   # maximum distance the laser can recognize
	laser_start_angle = 0.0   # angle in degree of first sample
	laser_end_angle = 359.0   # angle in degree of last sample
	laser_num_samples = 360   # number of laser samples
	laser_offset{   # offset of laser from base center
		x = 0.0
		y = 0.0
	}
	base_size{   # width(x) and height(y) of robot base
		x = 0.13
		y = 0.13
	}
	wheel_size{   # width(x) and height(y) of robot wheels
		x = 0.018
		y = 0.064
	}
	wheel_offset{   # offset of wheels from edge of base
		x = 0.0
		y = 0.034
	}
	bevel_size{   # size of bevel along x/y axis at the base corners
		x = 0.025
		y = 0.025
	}
	forward_speed{   # velocity for forward action
		linear = 0.2
		angular = 0.0
	}
	left_speed{   # velocity for left action
		linear = 0.15
		angular = 0.75
	}
	right_speed{   # velocity for right action
		linear = 0.15
		angular = -0.75
	}
	strong_left_speed{   # velocity for strong left action
		linear = 0.0
		angular = 1.5
	}
	strong_right_speed{   # velocity for strong right action
		linear = 0.0
		angular = -1.5
	}
	backward_speed{   # velocity for backward action
		linear = -0.1
		angular = 0.0
	}
	camera_angle = 90.0   # camera view in degree
}

stage{
	random_seed = 0   # seed for pseudo random generator
	initial_level = "random"   # level loaded on startup (-1 for none)
	level_size = 4.0   # width and height of default levels
	max_obstacle_size = 1.0   # maximum diameter of static obstacles
	min_obstacle_size = 0.1   # minimum diameter of static obstacles
	num_obstacles = 8   # number of static obstacles
	dynamic_obstacle_size = 0.3   # size of dynamic obstacle
	num_dynamic_obstacles = 1   # number of dynamic obstacles in static_dynamic level
	obstacle_speed = 0.08   # in m/s for dynamic obstacles
	max_time_chatting = 50.0   # maximum time for chatting between two wanderers
	goal_size = 0.1   # diameter of circular goal to reach
	svg_path = "svg_levels/"   # path to folder where svg files are stored
	static_map_ros_service_name = "/static_map"   #name of map service provided ros map server.
}

