import random

NUM_ACTIONS = 6

class Agent:
	### init ###
	# called once before training starts
	# @param device_name string defining the device to use for training ('cpu' or 'cuda')
	# @param model_name path of model to load from file and to initialize the net with, set to None if not specified by user
	# @param num_observations number of scalar values per environment passed to pre_step/post_step
	# @param num_envs number of parallel environments in the simulator
	# @param num_threads number of threads (cpu cores) to be used for parallel training
	# @param training_data_path path to a folder (ending with '/') were all the training data is stored (plots, settings, etc.)
	#			it is encouraged to store the model weights at this location
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		print("Agent init !")
		self.num_envs = num_envs

	### pre_step ###
	# this function is called before the simulation step
	# @param observations is a list containing the observations [distance, angle, laser0, ..., laserN-1, additional_data0, ..., additional_dataN-1];
	#						if more than one environment is used, observations is a list containing a list for each environment
	# @return single action that is performed accross all environments or a list of actions (dedicated action for each environment)
	# NOTE: an action is either a single integer referring to a specific action type: 0->FORWARD, 1->LEFT, 2->RIGHT, 3->STRONG_LEFT, 4->STRONG_RIGHT, 5->BACKWARD, 6->STOP (discrete action space)
	# 		or a tuple containing two float numbers (linear_velocity, angular_velocity) (continuous action space)
	def pre_step(self, observations):
		actions = []
		for i in range(self.num_envs):
			actions.append(random.randrange(0, NUM_ACTIONS))
		self.last_action = actions[0]
		return actions

	### post_step ###
	# this function is called after simulation step has been performed
	# @param new_observations is a list containing the new observation (see function pre_step())
	# @param rewards are the rewards that have been received in each environment, single value (one environment) or list (multiple environments)
	# @param dones 1 if episode is over, 0 if episode is not finished yet, single value (one environment) or list (multiple environments) 
	# @param mean_reward scalar value, mean reward from last 100 episodes accross all environments
	# @param mean_success scalar value, mean success rate from last 100 episodes accross all environments
	# @return 0 to continue training normally, 1 to stop training, -1 to continue training but environments are not reset if episodes are done
	def post_step(self, new_observations, rewards, dones, mean_reward, mean_success):
		return 0
	
	### get_stats ###
	# this function is called on every end of an episode and can be used to return metrics to be recorded as training data
	# @return a list of tuples (name, value), an empty list or None
	#			the value must be of type float or int, the name must be a string
	# NOTE: do not use ',' in the name for as this symbol is used as delimiter in the output csv file
	#		make sure to not change the order of returned metrics in the list between multiple calls to get_stats()
	def get_stats(self):
		return [("Action chosen", self.last_action), ("Pi", 3.14159)]
	
	### stop ###
	# called when training has been stopped by the user in the simulator
	# @return optionally a string can be returned to be written to the results-file
	def stop(self):
		pass
