#include <engine/GlobalSettings.hpp>

void GlobalSettings::setToDefault()
{
	// video
	_settings.video.resolution_w = 800;
	_settings.video.resolution_h = 600;
	_settings.video.window_x = -1;
	_settings.video.window_y = -1;
	_settings.video.maximized = 0;
	_settings.video.fps = 60;
	_settings.video.fullscreen = 0;
	_settings.video.msaa = 4;
	_settings.video.enabled = 1;

	// gui
	_settings.gui.font_size = 16;
	_settings.gui.show_stage = 1;
	_settings.gui.show_robot = 1;
	_settings.gui.show_goal = 1;
	_settings.gui.show_laser = 1;
	_settings.gui.show_stats = 1;
	_settings.gui.show_goal_spawn = 0;
	_settings.gui.show_trail = 1;
	_settings.gui.camera_follow = 0;
	_settings.gui.camera_x = 0;
	_settings.gui.camera_y = 0;
	_settings.gui.camera_zoom = 0.45;
	_settings.gui.camera_rotation = 0;
	_settings.gui.camera_zoom_factor = 1.3;

	// physics
	_settings.physics.time_step = 0.1f;
	_settings.physics.step_iterations = 1;
	_settings.physics.fps = 60;
	_settings.physics.position_iterations = 8;
	_settings.physics.velocity_iterations = 12;

	// keys
	_settings.keys.up = SDLK_UP;
	_settings.keys.down = SDLK_DOWN;
	_settings.keys.left = SDLK_LEFT;
	_settings.keys.right = SDLK_RIGHT;
	_settings.keys.reset = SDLK_r;
	_settings.keys.play_pause_simulation = SDLK_SPACE;

	// training
	_settings.training.max_time = 100.0f;
	_settings.training.episode_over_on_hit = 0;
	_settings.training.reward_goal = 100.0f;
	_settings.training.reward_towards_goal = 0.1f;
	_settings.training.reward_away_from_goal = -0.2f;
	_settings.training.reward_hit = -10.0f;
	_settings.training.reward_time_out = 0.f;
	_settings.training.num_envs = 4;
	_settings.training.num_threads = -1;
	_settings.training.agent_class = "Agent";

	// stage
	_settings.stage.random_seed = 0;
	_settings.stage.initial_level = "random";
	_settings.stage.level_size = 4;
	_settings.stage.obstacle_speed = 0.08;
	_settings.stage.dynamic_obstacle_size = 0.3;
	_settings.stage.num_dynamic_obstacles = 4;
	_settings.stage.num_obstacles = 8;
	_settings.stage.min_obstacle_size = 0.1;
	_settings.stage.max_obstacle_size = 1.0;
	_settings.stage.goal_size = 0.1;
	_settings.stage.svg_path = "svg_levels/";

	// robot
	_settings.robot.laser_noise = 0;
	_settings.robot.laser_max_distance = 3.5;
	_settings.robot.laser_start_angle = 0.0f;
	_settings.robot.laser_end_angle = 359.0f;
	_settings.robot.laser_num_samples = 360;	
	_settings.robot.laser_offset.x = 0;	
	_settings.robot.laser_offset.y = 0;	
	_settings.robot.base_size.x = 0.13;
	_settings.robot.base_size.y = 0.13;
	_settings.robot.wheel_size.x = 0.018;
	_settings.robot.wheel_size.y = 0.064;
	_settings.robot.wheel_offset.x = 0.f;
	_settings.robot.wheel_offset.y = 0.034f;
	_settings.robot.bevel_size.x = 0.025f;
	_settings.robot.bevel_size.y = 0.025f;
	_settings.robot.forward_speed.linear = 0.20;
	_settings.robot.forward_speed.angular = 0.0;
	_settings.robot.backward_speed.linear = -0.1;
	_settings.robot.backward_speed.angular = 0.0;
	_settings.robot.left_speed.linear = 0.15;
	_settings.robot.left_speed.angular = 0.75;
	_settings.robot.right_speed.linear = 0.15;
	_settings.robot.right_speed.angular = -0.75;
	_settings.robot.strong_left_speed.linear = 0.0;
	_settings.robot.strong_left_speed.angular = 1.5;
	_settings.robot.strong_right_speed.linear = 0.0;
	_settings.robot.strong_right_speed.angular = -1.5;

}
