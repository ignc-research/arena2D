import policy_value_fc
import torch
import torch.nn as nn
import numpy
import pathlib
import os.path
import shutil

NUM_ACTIONS = 6 # discrete actions
STOP_TESTING = 10000		# number of episodes used for testing

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		assert(num_envs == 1)
		### test stuff ###
		self.episode_idx = 0
		if model_name is None:
			print("WARNING: No Model given! Add \'--model <model_name>\' to specifie a path to a model file to be loaded as initial model weights by the agent.")
		self.model_name = model_name
		### --- ###
		self.device = torch.device(device_name)
		self.num_envs = num_envs
		self.net = policy_value_fc.PolicyValueFC(num_observations, NUM_ACTIONS)
		self.net.to(self.device)
		if model_name != None:
			self.net.load_state_dict(torch.load(model_name, map_location=self.device))

	def pre_step(self, observations):
		obs_v = torch.FloatTensor(observations).to(self.device)
		if self.num_envs == 1: # only 1 environment -> treat as batch of size 1
			obs_v = obs_v.view(1, -1)
		policy_v = self.net.get_policy(obs_v)
		probs = nn.functional.softmax(policy_v, dim=1).data.cpu().numpy()
		actions = []
		for i in range(self.num_envs):
			actions.append(numpy.random.choice(NUM_ACTIONS, p=probs[i]))

		return actions 

	def post_step(self, new_observations, rewards, dones, mean_reward, mean_success):
		for d in range(self.num_envs):
			if dones[d]: # episode done
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