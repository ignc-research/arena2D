import policy_value_fc
import torch
import torch.nn as nn
import numpy

NUM_ACTIONS = 6 # discrete actions

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
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
		return 0

	def get_stats(self):
		return []

	def stop(self):
		pass
