from dqn_models import fc 

import torch
import pathlib
import os.path
import shutil

NUM_ACTIONS = 7
STOP_TESTING = 10000		# number of episodes used for testing

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads,trash):
		assert(num_envs == 1)
		### test stuff ###
		self.episode_idx = 0
		if model_name is None:
			print("WARNING: No Model given! Add \'--model <model_name>\' to specifie a path to a model file to be loaded as initial model weights by the agent.")
		self.model_name = model_name
		### --- ###
		self.device = torch.device(device_name)
		self.net = fc.FC_DQN(num_observations, NUM_ACTIONS)
		self.net.train(False)# set training mode to false to deactivate dropout layer
		if model_name != None:
			self.net.load_state_dict(torch.load(model_name, map_location=self.device))
		self.net.to(self.device)

	def pre_step(self, observation):
		# passing observation through net
		state_v = torch.FloatTensor([observation]).to(self.device)
		q_vals_v = self.net(state_v)
		# select action with max q value
		_, act_v = torch.max(q_vals_v, dim=1)
		action = int(act_v.item())

		return action

	def post_step(self, new_observation, reward, is_done, mean_reward, mean_success):
	# this function is called after simulation step has been performed
	# return 0 to continue, 1 to stop training
		if is_done != 0: # episode done
			self.episode_idx += 1
			if self.episode_idx >= STOP_TESTING:
				return 1 #stop testing
		return 0 # keep on testing

	def get_stats(self):
		return []

	def stop(self):
		#copy evaluation.py script to training folder of given model
		evaluation_path_target = os.path.dirname(self.model_name) + '/evaluation.py'
		evaluation_path_orignal = '../arena2d-sim/scripts/evaluation.py'
		print(evaluation_path_target)
		print(evaluation_path_orignal)
		shutil.copyfile(evaluation_path_orignal, evaluation_path_target)
		
