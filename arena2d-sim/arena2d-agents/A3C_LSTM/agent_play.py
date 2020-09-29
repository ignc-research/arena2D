import policy_value_lstm
import torch
import torch.nn as nn
import numpy

NUM_ACTIONS = 6 # discrete actions

class Agent:
	def __init__(self, device_name, model_name, num_observations, num_envs, num_threads, training_data_path):
		self.device = torch.device(device_name)
		self.num_envs = num_envs
		self.net = policy_value_lstm.PolicyValueLSTM(num_observations, NUM_ACTIONS)
		self.net.to(self.device)
		self.net.train(False)
		if model_name != None:
			self.net.load_state_dict(torch.load(model_name, map_location=self.device))

		self.reset_all_hidden()
		
	def pre_step(self, observations):
		obs_v = torch.FloatTensor(observations).to(self.device)
		if self.num_envs == 1: # only 1 environment -> treat as batch of size 1
			obs_v = obs_v.view(1, -1)

		policy_v, _, (self.last_h, self.last_c) = self.net.forward_hidden(obs_v, (self.last_h, self.last_c))
		probs = nn.functional.softmax(policy_v, dim=1).data.cpu().numpy()
		actions = []
		for i in range(self.num_envs):
			actions.append(numpy.random.choice(NUM_ACTIONS, p=probs[i]))

		return actions;

	def post_step(self, new_observations, rewards, dones, mean_reward, mean_success):
		for d in range(self.num_envs):
			if dones[d]:
				self.reset_hidden(d)
		return 0
	
	def get_stats(self):
		return []

	def reset_all_hidden(self):
		(self.last_h, self.last_c) = self.net.get_initial_hidden(self.num_envs)
		self.last_h.to(self.device)
		self.last_c.to(self.device)

	def reset_hidden(self, env):
		self.last_h[0][env].zero_()
		self.last_c[0][env].zero_()
